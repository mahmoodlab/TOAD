import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM, CLAM_Simple
from models.model_attention_mil import MIL_Attention_fc_mtl
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type in ['clam', 'clam_simple'] and args.subtyping:
        model_dict.update({'subtyping': True})
    
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam', 'clam_simple']:
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam':
            model = CLAM(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            model = CLAM_Simple(**model_dict, instance_loss_fn=instance_loss_fn)

    elif args.model_type =='attention_mil':
        model = MIL_Attention_fc_mtl(**model_dict)
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, collate_fn='MIL_mtl')
    val_loader = get_split_loader(val_split, collate_fn='MIL_mtl')
    test_loader = get_split_loader(test_split, collate_fn='MIL_mtl')
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam', 'clam_new']:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, cls_val_error, cls_val_auc, site_val_error, site_val_auc, _= summary(model, val_loader, args.n_classes)
    print('Cls Val error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_val_error, cls_val_auc) + 
        ' Site Val error: {:.4f}, Site ROC AUC: {:.4f}'.format(site_val_error, site_val_auc))

    results_dict, cls_test_error, cls_test_auc, site_test_error, site_test_auc, acc_loggers= summary(model, test_loader, args.n_classes)
    print('Cls Test error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_test_error, cls_test_auc) + 
        ' Site Test error: {:.4f}, Site ROC AUC: {:.4f}'.format(site_test_error, site_test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_loggers[0].get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    for i in range(2):
        acc, correct, count = acc_loggers[1].get_summary(i)
        print('site {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_site_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/cls_val_error', cls_val_error, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/site_val_error', site_val_error, 0)
        writer.add_scalar('final/site_val_auc', site_val_auc, 0)
        writer.add_scalar('final/cls_test_error', cls_test_error, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)

        writer.add_scalar('final/site_test_error', site_test_error, 0)
        writer.add_scalar('final/site_test_auc', site_test_auc, 0)
    
    writer.close()
    return results_dict, cls_test_auc, cls_val_auc, 1-cls_test_error, 1-cls_val_error, site_test_auc, site_val_auc, 1-site_test_error, 1-site_val_error 


# def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.train()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     inst_logger = Accuracy_Logger(n_classes=n_classes)
    
#     train_loss = 0.
#     train_error = 0.
#     train_inst_loss = 0.
#     inst_count = 0

#     sample_size = model.k_sample

#     print('\n')
#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

#         acc_logger.log(Y_hat, label)
#         loss = loss_fn(logits, label)
#         loss_value = loss.item()

#         instance_loss = instance_dict['instance_loss']
#         inst_count+=1
#         instance_loss_value = instance_loss.item()
#         train_inst_loss += instance_loss_value

#         instance_loss = instance_dict['instance_loss']
#         inst_count+=1
#         instance_loss_value = instance_loss.item()
#         train_inst_loss += instance_loss_value

        
#         total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

#         p_acc = instance_dict['p_acc']
#         n_acc = instance_dict['n_acc']
#         inst_logger.log_batch(sample_size, int(p_acc * sample_size), 1)
#         inst_logger.log_batch(sample_size, int(n_acc * sample_size), 0)

#         train_loss += loss_value
#         if (batch_idx + 1) % 5 == 0:
#             print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
#                 'inst_p_acc: {:.4f}, inst_n_acc: {:.4f}, label: {}, bag_size: {}'.format(p_acc, n_acc, label.item(), data.size(0)))

#         error = calculate_error(Y_hat, label)
#         train_error += error
        
#         # backward pass
#         total_loss.backward()
#         # step
#         optimizer.step()
#         optimizer.zero_grad()

#     # calculate loss and error for epoch
#     train_loss /= len(loader)
#     train_error /= len(loader)
    
#     if inst_count > 0:
#         train_inst_loss /= inst_count
#         print('\n')
#         for i in range(2):
#             acc, correct, count = inst_logger.get_summary(i)
#             print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
        
#             if writer:
#                 writer.add_scalar('train/inst_class_{}_acc'.format(i), acc, epoch)

#     print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

#     if writer:
#         writer.add_scalar('train/loss', train_loss, epoch)
#         writer.add_scalar('train/error', train_error, epoch)
#         writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    site_logger = Accuracy_Logger(n_classes=2)
    cls_train_error = 0.
    cls_train_loss = 0.
    site_train_error = 0.
    site_train_loss = 0.
    print('\n')
    for batch_idx, (data, label, site) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        site = site.to(device)
            
        results_dict = model(data)
        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        site_logits, site_prob, site_hat = results_dict['site_logits'], results_dict['site_prob'], results_dict['site_hat']
        
        cls_logger.log(Y_hat, label)
        site_logger.log(site_hat, site)
        
        cls_loss =  loss_fn(logits, label) 
        site_loss = loss_fn(site_logits, site)
        loss = cls_loss * 0.75 + site_loss * 0.25
        cls_loss_value = cls_loss.item()
        site_loss_value = site_loss.item()
        
        cls_train_loss += cls_loss_value
        site_train_loss+=site_loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, cls loss: {:.4f}, site loss: {:.4f} '.format(batch_idx, cls_loss_value, site_loss_value) + 
                'label: {}, site: {}, bag_size: {}'.format(label.item(), site.item(), data.size(0)))
           
        cls_error = calculate_error(Y_hat, label)
        cls_train_error += cls_error
        site_error = calculate_error(site_hat, site)
        site_train_error += site_error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    cls_train_loss /= len(loader)
    cls_train_error /= len(loader)
    site_train_loss /= len(loader)
    site_train_error /= len(loader)

    print('Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}'.format(epoch, cls_train_loss, cls_train_error))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    for i in range(2):
        acc, correct, count = site_logger.get_summary(i)
        print('site {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/site_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/cls_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_error', cls_train_error, epoch)
        writer.add_scalar('train/site_loss', site_train_loss, epoch)
        writer.add_scalar('train/site_error', site_train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    site_logger = Accuracy_Logger(n_classes=2)
    # loader.dataset.update_mode(True)
    cls_val_error = 0.
    cls_val_loss = 0.
    site_val_error = 0.
    site_val_loss = 0.
    
    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))
    site_probs = np.zeros((len(loader), 2))
    site_labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, site) in enumerate(loader):
            data =  data.to(device)
            label = label.to(device)
            site = site.to(device)

            results_dict = model(data)
            logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            site_logits, site_prob, site_hat = results_dict['site_logits'], results_dict['site_prob'], results_dict['site_hat']
            del results_dict

            cls_logger.log(Y_hat, label)
            site_logger.log(site_hat, site)
            
            cls_loss =  loss_fn(logits, label) 
            site_loss = loss_fn(site_logits, site)
            loss = cls_loss * 0.5 + site_loss * 0.5
            cls_loss_value = cls_loss.item()
            site_loss_value = site_loss.item()

            cls_probs[batch_idx] = Y_prob.cpu().numpy()
            cls_labels[batch_idx] = label.item()

            site_probs[batch_idx] = site_prob.cpu().numpy()
            site_labels[batch_idx] = site.item()
            
            cls_val_loss += cls_loss_value
            site_val_loss+= site_loss_value
            cls_error = calculate_error(Y_hat, label)
            cls_val_error += cls_error
            site_error = calculate_error(site_hat, site)
            site_val_error += site_error
            

    cls_val_error /= len(loader)
    cls_val_loss /= len(loader)
    site_val_error /= len(loader)
    site_val_loss /= len(loader)

    # if n_classes == 2:
    #     cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        
    # else:
    #     cls_auc = roc_auc_score(cls_labels, cls_probs, multi_class='ovr')


    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        cls_aucs = []
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        cls_auc = np.nanmean(np.array(cls_aucs))
    
    site_auc = roc_auc_score(site_labels, site_probs[:, 1])
    
    if writer:
        writer.add_scalar('val/cls_loss', cls_val_loss, epoch)
        writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_error', cls_val_error, epoch)
        writer.add_scalar('val/site_loss', site_val_loss, epoch)
        writer.add_scalar('val/site_auc', site_auc, epoch)
        writer.add_scalar('val/site_error', site_val_error, epoch)

    print('\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_val_error, cls_auc) + 
        ' site val_loss: {:.4f}, site val_error: {:.4f}, site auc: {:.4f}'.format(site_val_loss, site_val_error, site_auc))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    
    for i in range(2):
        acc, correct, count = site_logger.get_summary(i)
        print('site {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/site_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, cls_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

# def validate_clam(cur, epoch, model, loader, n_classes,  early_stopping = None, writer = None, loss_fn = None, results_dir = None):
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     acc_logger = Accuracy_Logger(n_classes=n_classes)
#     inst_logger = Accuracy_Logger(n_classes=n_classes)
#     val_loss = 0.
#     val_error = 0.

#     val_inst_loss = 0.
#     val_inst_acc = 0.
#     inst_count=0
    
#     prob = np.zeros((len(loader), n_classes))
#     labels = np.zeros(len(loader))
#     sample_size = model.k_sample
#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(loader):
#             data, label = data.to(device), label.to(device)      
#             logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
#             acc_logger.log(Y_hat, label)
            
#             loss = loss_fn(logits, label)

#             val_loss += loss.item()

#             instance_loss = instance_dict['instance_loss']
            
#             inst_count+=1
#             instance_loss_value = instance_loss.item()
#             val_inst_loss += instance_loss_value
#             p_acc = instance_dict['p_acc']
#             n_acc = instance_dict['n_acc']
#             inst_logger.log_batch(sample_size, int(p_acc * sample_size), 1)
#             inst_logger.log_batch(sample_size, int(n_acc * sample_size), 0)

#             prob[batch_idx] = Y_prob.cpu().numpy()
#             labels[batch_idx] = label.item()
            
#             error = calculate_error(Y_hat, label)
#             val_error += error

#     val_error /= len(loader)
#     val_loss /= len(loader)

#     if n_classes == 2:
#         auc = roc_auc_score(labels, prob[:, 1])
    
#     else:
#         auc = roc_auc_score(labels, prob, multi_class='ovr')

#     print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
#     if inst_count > 0:
#         val_inst_loss /= inst_count
#         for i in range(2):
#             acc, correct, count = inst_logger.get_summary(i)
#             print('Clustering: class {}, acc {}, correct {}/{}'.format(i, acc, correct, count))
        
#             if writer:
#                 writer.add_scalar('val/inst_class_{}_acc'.format(i), acc, epoch)
    
    
#     if writer:
#         writer.add_scalar('val/loss', val_loss, epoch)
#         writer.add_scalar('val/auc', auc, epoch)
#         writer.add_scalar('val/error', val_error, epoch)
#         writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


#     for i in range(n_classes):
#         acc, correct, count = acc_logger.get_summary(i)
#         print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
#         if writer:
#             writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

#     if early_stopping:
#         assert results_dir
#         early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
#         if early_stopping.early_stop:
#             print("Early stopping")
#             return True

#     return False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    site_logger = Accuracy_Logger(n_classes=2)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.
    site_test_error = 0.
    site_test_loss = 0.

    all_cls_probs = np.zeros((len(loader), n_classes))
    all_cls_labels = np.zeros(len(loader))
    all_site_probs = np.zeros((len(loader), 2))
    all_site_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, site) in enumerate(loader):
        data =  data.to(device)
        label = label.to(device)
        site = site.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data)

        logits, Y_prob, Y_hat  = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
        site_logits, site_prob, site_hat = results_dict['site_logits'], results_dict['site_prob'], results_dict['site_hat']
        del results_dict

        cls_logger.log(Y_hat, label)
        site_logger.log(site_hat, site)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        site_probs = site_prob.cpu().numpy()
        all_site_probs[batch_idx] = site_probs
        all_site_labels[batch_idx] = site.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item(), 
                                'site_prob': site_probs, 'site_label': site.item()}})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error
        site_error = calculate_error(site_hat, site)
        site_test_error += site_error

    cls_test_error /= len(loader)
    site_test_error /= len(loader)

    if n_classes == 2:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
        
    else:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs, multi_class='ovr')
    
    site_auc = roc_auc_score(all_site_labels, all_site_probs[:, 1])

    return patient_results, cls_test_error, cls_auc, site_test_error, site_auc, (cls_logger, site_logger)
