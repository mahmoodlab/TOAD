import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np
"""
A Modified Implementation of Deep Attention MIL
"""


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
Attention MIL with 1 additional hidden fc-layer after CNN feature extractor
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class MIL_Attention_fc_mtl(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(MIL_Attention_fc_mtl, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 2)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 2)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        self.site_classifier = nn.Linear(size[1], 2)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)
        self.site_classifier = self.site_classifier.to(device)
        
    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 

        logits  = self.classifier(M[0].unsqueeze(0)) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        site_logits  = self.site_classifier(M[1].unsqueeze(0)) 
        site_hat = torch.topk(site_logits, 1, dim = 1)[1]
        site_prob = F.softmax(site_logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 
                'site_logits': site_logits, 'site_prob': site_prob, 'site_hat': site_hat, 'A': A_raw})

        return results_dict

class MIL_Attention_fc_mtl_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(MIL_Attention_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 2)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 2)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1]+1, n_classes)
        self.site_classifier = nn.Linear(size[1]+1, 2)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)
        self.site_classifier = self.site_classifier.to(device)
        
    def forward(self, h, sex=None, return_features=False, attention_only=False, return_topk=0):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        M = torch.cat([M, sex.repeat(M.size(0),1)], dim=1) 

        logits  = self.classifier(M[0].unsqueeze(0)) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        site_logits  = self.site_classifier(M[1].unsqueeze(0)) 
        site_hat = torch.topk(site_logits, 1, dim = 1)[1]
        site_prob = F.softmax(site_logits, dim = 1)

        results_dict = {}
        if return_features:
            if return_topk <= 0:
                results_dict.update({'features': M})
            else:
                # pdb.set_trace()
                # h = torch.cat([h, sex.repeat(h.size(0),1)], dim=1) 
                top_p_ids = torch.topk(A[0], return_topk)[1]
                top_p = torch.index_select(h, dim=0, index=top_p_ids)
                results_dict.update({'cls_features': top_p.clone()})
                top_p_ids = torch.topk(A[1], return_topk)[1]
                top_p = torch.index_select(h, dim=0, index=top_p_ids)
                results_dict.update({'site_features': top_p.clone()})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 
                'site_logits': site_logits, 'site_prob': site_prob, 'site_hat': site_hat, 'A': A_raw})

        return results_dict

class MIL_Attention_fc_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 1):
        super(MIL_Attention_fc_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1]+1, n_classes)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)
        
    def forward(self, h, sex=None, return_features=False, attention_only=False, return_topk=0):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)

        M = torch.cat([M, sex.repeat(M.size(0),1)], dim=1) 

        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            if return_topk <= 0:
                results_dict.update({'features': M})
            else:
                top_p_ids = torch.topk(A.view(-1), return_topk)[1]
                top_p = torch.index_select(h, dim=0, index=top_p_ids)
                results_dict.update({'features': top_p.clone()})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict

class MIL_Attention_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2):
        super(MIL_Attention_fc, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

        
    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict
