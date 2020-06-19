# python imports
from math import floor
import pdb
import os

# external libraries
import pandas as pd    
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy import interp
import seaborn as sns

# internal imports
from utils.file_utils import save_pkl, load_pkl

def font_prop(size = 40, font_dir='figures/fonts', fname='HelveticaNeue.ttf'):
	return fm.FontProperties(size = size, fname=os.path.join(font_dir, fname))

def configure_font(label_size = 50, font_dir='figures/fonts', fname='HelveticaNeue.ttf'):
	font_args = dict()
	font_args['ax_label_fprop']= font_prop(label_size, font_dir, fname)
	font_args['ax_tick_label_fprop'] = font_prop(int(label_size * 0.9), font_dir, fname)
	font_args['title_fprop'] = font_prop(label_size, font_dir, fname)
	font_args['zoom_ax_tick_label_fprop'] = font_prop(int(label_size * 0.6), font_dir, fname)
	font_args['legend_fprop'] = font_prop(int(label_size * 0.65), font_dir, fname)
	return font_args


def init_roc_curve(title='roc curve', zoom_in=True, zoom_limits=(0, 0.17, 0.85, 1.01), 
	font_args=None, **zoom_kwargs):
	
	if font_args is None:
		font_args = configure_font()

	fig = plt.figure(figsize=(8, 8))
	# ax = fig.add_axes([0.14, 0.18, 0.65, 0.65])
	ax = fig.add_axes([0.2, 0.2, 0.75, 0.75])
	axins=None
#     ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Chance')
	ax.set_xlim([-0.05, 1.02])
	ax.set_ylim([-0.05, 1.02])
	
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
	ax.set_xlabel('1-Specificity', fontproperties=font_args['ax_label_fprop'])
	ax.set_ylabel('Sensitivity', fontproperties=font_args['ax_label_fprop'])
	ax.set_title(title, fontproperties=font_args['title_fprop'])
	plt.setp(ax.get_xticklabels(), fontproperties =font_args['ax_tick_label_fprop'])
	plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
#     ax.set_yticks(fontproperties = font_prop(size=60))
#     ax.set_xticks(fontproperties = font_prop(size=60))
	# fig.tight_layout()
	
	if zoom_in:
		axins = zoomed_inset_axes(ax, **zoom_kwargs, loc='center', bbox_transform=ax.transAxes) 
		# zoom-factor: 2.5, location: upper-left
		x1, x2, y1, y2 = zoom_limits # specify the limits
		axins.set_xlim(x1-0.02, x2+0.02) # apply the x-limits
		axins.set_ylim(y1-0.02, y2+0.02) # apply the y-limits
		axins.set_xticks(np.linspace(x1, x2, num=5))
		axins.set_yticks(np.linspace(y1, y2, num=5))
		axins.xaxis.set_visible('False')
		axins.yaxis.set_visible('False')
		axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
		plt.setp(axins.get_xticklabels(), fontproperties = font_args['zoom_ax_tick_label_fprop'])
		plt.setp(axins.get_yticklabels(), fontproperties = font_args['zoom_ax_tick_label_fprop'])
#         axins.xaxis.set_major_locator(plt.MaxNLocator(5))
#         axins.yaxis.set_major_locator(plt.MaxNLocator(5))
	
	return fig, ax, axins

   
def finalize_roc_curve(fig, ax, font_args, ensemble=False):
	# ax.legend(loc="lower right", prop=font_args['legend_fprop']) frameon=False,
	if not ensemble:
		legend = ax.legend(*map(reversed, ax.get_legend_handles_labels()), loc="lower right", prop=font_args['legend_fprop'])

	else:
		legend= ax.legend(handlelength=0, handletextpad=0, loc="lower right", prop=font_args['legend_fprop'])
		for item in legend.legendHandles[:1]:
			item.set_visible(False)

	legend.get_frame().set_linewidth(1)
	legend.get_frame().set_edgecolor((0, 0, 0))
	plt.gca().set_aspect('equal', adjustable='box')
	return fig, ax
	
def plot_roc_curve_avg_w_zoom(all_Y, all_Y_probs, fig, ax, label, lw=2.5, axins=None, alpha=.2, color=None):
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 101)
	
	for i in range(len(all_Y)):
		fpr, tpr, _ = roc_curve(all_Y[i].astype(int), all_Y_probs[i]) #calculate fpr, tpr for each set of predictions
		roc_auc = auc(fpr, tpr) # calculate auc of curve 
		interp_tpr = interp(mean_fpr, fpr, tpr) #interpolate tpr at fixed values of fpr
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(roc_auc)
			 
	mean_tpr = np.mean(tprs, axis=0)
#     mean_auc = auc(mean_fpr, mean_tpr)
	mean_auc = np.mean(aucs)
	std_auc = np.std(aucs)

	plt_kwargs = {'lw': lw, 'alpha': 0.65}
	fill_kwargs = {'alpha': 0.15}

	if color:
		plt_kwargs.update({'color': color})
		fill_kwargs.update({'color': color})
	
	ax.plot(mean_fpr, mean_tpr,
		label='{}: AUC = {:.3f} $\pm$ {:.3f}'.format(label, mean_auc, std_auc),
		**plt_kwargs)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, **fill_kwargs)
	
	if axins:
		axins.plot(mean_fpr, mean_tpr,
		label='{}: AUC = {:.3f} $\pm$ {:.3f}'.format(label, mean_auc, std_auc),
		**plt_kwargs)
		axins.fill_between(mean_fpr, tprs_lower, tprs_upper, **fill_kwargs)
	
	return fig, ax, axins


def plot_clusters(df, title=None, label_dict=None, mode='tsne', font_args=None, c_col='c_probs_1', 
	cbar_title='positive cluster probability'):
	fig = plt.figure(figsize=(20, 20))
	ax = fig.add_axes([0.14, 0.18, 0.8, 0.8])
	#fig, ax = plt.subplots(figsize=(20,20))
	if label_dict is None:
		label_dict = {'0': 'normal', '1': 'tumor'}
	if mode == 'tsne':
		x_label = "t-SNE1"
		y_label = "t-SNE2"
	elif mode == 'pca':
		x_label = "PC1"
		y_label = "PC2"

	if font_args is None:
		font_args = configure_font()
	
	points = ax.scatter(df["comp1"], df["comp2"],
					 c=df[c_col], s=350, cmap="coolwarm", alpha=0.4)
	
	# cbar = plt.colorbar(points, ax=ax)
	# # cbar.ax.tick_params(axis='y', which='major', pad=10)
	# plt.setp(cbar.ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
	# cbar.set_label(cbar_title, rotation=270, fontproperties=font_args['ax_tick_label_fprop'], labelpad=75)
	
	# if x_lims is not None:
	#     x1, x2 = x_lims # specify the limits
	#     ax.set_xlim(x1-0.2, x2+0.2) # apply the x-limits
	#     ax.set_xticks(np.linspace(x1, x2, num=3))
	# if y_lims is not None:
	#     y1, y2 = y_lims # specify the limits
	#     ax.set_ylim(y1-0.2, y2+0.2)
	#     ax.set_yticks(np.linspace(y1, y2, num=3))

	
	x_max, x_min = df["comp1"].max(), df["comp1"].min()
	x_range = x_max - x_min
	x_multiple = x_range / 3 
	y_max, y_min = df["comp2"].max(), df["comp2"].min()
	y_range = (y_max - y_min)
	y_multiple = y_range / 3 
	
	ax.set_ylim(y_min-y_range*0.1, y_max+y_range*0.1)
	ax.set_yticks(np.linspace(y_min, y_max, num=3))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	
	ax.set_xlim(x_min-x_range*0.1, x_max+x_range*0.1)
	ax.set_xticks(np.linspace(x_min, x_max, num=3))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

	ax.set_xlabel(x_label, fontproperties=font_args['ax_label_fprop'])
	ax.set_ylabel(y_label, fontproperties=font_args['ax_label_fprop'])
	plt.setp(ax.get_xticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
	plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
	
	if title:
		ax.set_title(title, fontproperties=font_args['title_fprop'])

	# labels = [label_dict[label]handles, labels = ax.get_legend_handles_labels()
	# legend_title = labels[0]
	# legend_handle = handles[0]
	# labels = labels[1:] for label in labels]
	# legend = ax.legend(handles[1:], labels, prop=font_args['legend_fprop'], markerscale=4)
	# legend.get_frame().set_linewidth(1.75)
	# legend.get_frame().set_edgecolor((0, 0, 0))
#     for legend_handle in legend.legendHandles:
#         legend_handle._legmarker.set_markersize(9)
#     fig.tight_layout()
	return fig

def plot_clusters_attention(df, title=None, label_dict=None, palette=None, mode='tsne', font_args=None, hue_order=None):
	fig = plt.figure(figsize=(20, 20))
	ax = fig.add_axes([0.14, 0.18, 0.8, 0.8])
	#fig, ax = plt.subplots(figsize=(20,20))
	if label_dict is None:
		label_dict = {'0': 'normal', '1': 'tumor'}
	if mode == 'tsne':
		x_label = "t-SNE1"
		y_label = "t-SNE2"
	elif mode == 'pca':
		x_label = "PC1"
		y_label = "PC2"

	# {'1': 'Chromophobe', '2': 'Clear Cell', '3': 'Papillary', '0': 'Agnostic'}
	g = sns.scatterplot(
		x="comp1", y="comp2",
		hue="Y",
		data=df,
		legend='full',
		palette=palette,
		hue_order = hue_order,
		alpha=0.45,
		ax=ax,
		s=400, 
	)

	if font_args is None:
		font_args = configure_font()
	
	x_max, x_min = df["comp1"].max(), df["comp1"].min()
	x_range = x_max - x_min
	x_multiple = x_range / 3 
	y_max, y_min = df["comp2"].max(), df["comp2"].min()
	y_range = (y_max - y_min)
	y_multiple = y_range / 3 
	
	ax.set_ylim(y_min-y_range*0.1, y_max+y_range*0.1)
	ax.set_yticks(np.linspace(y_min, y_max, num=3))
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	
	ax.set_xlim(x_min-x_range*0.1, x_max+x_range*0.1)
	ax.set_xticks(np.linspace(x_min, x_max, num=3))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

	ax.set_xlabel(x_label, fontproperties=font_args['ax_label_fprop'])
	ax.set_ylabel(y_label, fontproperties=font_args['ax_label_fprop'])
	plt.setp(ax.get_xticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
	plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
	
	if title:
		ax.set_title(title, fontproperties=font_args['title_fprop'])

	handles, labels = ax.get_legend_handles_labels()
	legend_title = labels[0]
	legend_handle = handles[0]
	labels = labels[1:]
	labels = [label_dict[label] for label in labels]
	# legend = ax.legend(*map(reversed, (handles[1:], labels)), loc="upper right", prop=font_args['legend_fprop'],markerscale=4)
	legend = ax.legend(handles[1:], labels, prop=font_args['legend_fprop'], loc="upper right", markerscale=4)
	legend.get_frame().set_linewidth(1.75)
	legend.get_frame().set_edgecolor((0, 0, 0))
	# fig.tight_layout()
	return fig

def plot_roc_curve_avg_mc_w_zoom(all_Y, all_Y_probs, fig, ax, label, lw=2.5, axins=None, color=None, micro_avg=False, macro_avg=True):
	micro_tprs = []
	micro_aucs = []
	macro_tprs = []
	macro_aucs = []
	mean_fpr = np.linspace(0, 1, 101)
	n_classes = all_Y[0].shape[1]
	
	for i in range(len(all_Y)):
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for c in range(n_classes):
			fpr[c], tpr[c], _ = roc_curve(all_Y[i][:, c].astype(int), all_Y_probs[i][:, c])
			roc_auc[c] = auc(fpr[c], tpr[c])

		fpr["micro"], tpr["micro"], _ = roc_curve(all_Y[i].ravel(), all_Y_probs[i].ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
		micro_interp_tpr = interp(mean_fpr, fpr['micro'], tpr['micro'])
		micro_interp_tpr[0] = 0.0
		micro_tprs.append(micro_interp_tpr)
		micro_aucs.append(roc_auc['micro'])

		# macro average
		all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
		# Then interpolate all ROC curves at these points
		macro_mean_tpr = np.zeros_like(all_fpr)
		for c in range(n_classes):
			macro_mean_tpr += interp(all_fpr, fpr[c], tpr[c])

		# Finally average it and compute AUC
		macro_mean_tpr /= n_classes
		fpr["macro"] = all_fpr
		tpr["macro"] = macro_mean_tpr
		# fpr['macro'][0] = 0
		# tpr['macro'][0] = 0
		# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
		roc_auc['macro'] = roc_auc_score(np.argmax(all_Y[i], axis=1), all_Y_probs[i], multi_class='ovr')
		macro_interp_tpr = interp(mean_fpr, fpr['macro'], tpr['macro'])
		macro_interp_tpr[0] = 0.0
		macro_tprs.append(macro_interp_tpr)
		macro_aucs.append(roc_auc['macro'])

	print(macro_aucs)
	micro_mean_tpr = np.mean(micro_tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
	micro_mean_auc = np.mean(micro_aucs)
	micro_std_auc = np.std(micro_aucs)

	macro_mean_tpr = np.mean(macro_tprs, axis=0)
	macro_mean_auc = np.mean(macro_aucs)
	macro_std_auc = np.std(macro_aucs)

	plt_kwargs = {'lw': lw, 'alpha': .65}
	fill_kwargs = {'alpha': .15}
	if color:
		plt_kwargs.update({'color': color})
		fill_kwargs.update({'color': color})
	
	if micro_avg:
		ax.plot(mean_fpr, micro_mean_tpr,
			label='{}: Avg AUC = {:.3f} $\pm$ {:.3f}'.format(label, micro_mean_auc, micro_std_auc),
			**plt_kwargs)

		micro_std_tpr = np.std(micro_tprs, axis=0)
		micro_tprs_upper = np.minimum(micro_mean_tpr + micro_std_tpr, 1)
		micro_tprs_lower = np.maximum(micro_mean_tpr - micro_std_tpr, 0)
		ax.fill_between(mean_fpr, micro_tprs_lower, micro_tprs_upper, alpha=.2)


	if macro_avg:
		ax.plot(mean_fpr, macro_mean_tpr,
			label='{}: AUC = {:.3f} $\pm$ {:.3f}'.format(label, macro_mean_auc, macro_std_auc),
			**plt_kwargs)

		macro_std_tpr = np.std(macro_tprs, axis=0)
		macro_tprs_upper = np.minimum(macro_mean_tpr + macro_std_tpr, 1)
		macro_tprs_lower = np.maximum(macro_mean_tpr - macro_std_tpr, 0)
		ax.fill_between(mean_fpr, macro_tprs_lower, macro_tprs_upper, **fill_kwargs)
	
	if axins:
		if micro_avg:
			axins.plot(mean_fpr, micro_mean_tpr, **plt_kwargs)
			axins.fill_between(mean_fpr, micro_tprs_lower, micro_tprs_upper, **fill_kwargs)
		if macro_avg:
			axins.plot(mean_fpr, macro_mean_tpr, **plt_kwargs)
			axins.fill_between(mean_fpr, macro_tprs_lower, macro_tprs_upper, **fill_kwargs)
	
	return fig, ax, axins

def plot_roc_curve_ensemble_mc_w_zoom(all_Y, all_Y_probs, fig, ax, label, lw=2.5, axins=None, color=None, micro_avg=False, macro_avg=True, ci={}):
	micro_tprs = []
	micro_aucs = []
	macro_tprs = []
	macro_aucs = []
	mean_fpr = np.linspace(0, 1, 101)
	n_classes = all_Y.shape[1]
	
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for c in range(n_classes):
		fpr[c], tpr[c], _ = roc_curve(all_Y[:, c].astype(int), all_Y_probs[:, c])
		roc_auc[c] = auc(fpr[c], tpr[c])

	fpr["micro"], tpr["micro"], _ = roc_curve(all_Y.ravel(), all_Y_probs.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# macro average
	all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
	# Then interpolate all ROC curves at these points
	macro_mean_tpr = np.zeros_like(all_fpr)
	for c in range(n_classes):
		macro_mean_tpr += interp(all_fpr, fpr[c], tpr[c])
	# Finally average it and compute AUC
	macro_mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = macro_mean_tpr
	fpr['macro'][0] = 0
	tpr['macro'][0] = 0
	roc_auc['macro'] = roc_auc_score(np.argmax(all_Y, axis=1), all_Y_probs, multi_class='ovr')

	plt_kwargs = {'lw': lw, 'alpha': .65}
	fill_kwargs = {'alpha': .15}
	if color:
		plt_kwargs.update({'color': color})
	
	if micro_avg:
		ax.plot(fpr['micro'], tpr['micro'],
				label='{} \nMicro-Average AUC: {:.3f}'.format(label, roc_auc['micro']) + 
				  '\nCRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[0]['auc'], ci[0]['ci_low'], ci[0]['ci_high']) +
				  '\nCCRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[1]['auc'], ci[1]['ci_low'], ci[1]['ci_high']) + 
				  '\nPRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[2]['auc'], ci[2]['ci_low'], ci[2]['ci_high']),
				**plt_kwargs)

	if macro_avg:
		ax.plot(fpr['macro'], tpr['macro'],
				label='{} \nMacro-Averaged AUC: {:.3f}'.format(label, roc_auc['macro']) + 
				  '\nCRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[0]['auc'], ci[0]['ci_low'], ci[0]['ci_high']) +
				  '\nCCRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[1]['auc'], ci[1]['ci_low'], ci[1]['ci_high']) + 
				  '\nPRCC AUC: {:.3f}, 95% CI: {:.3f}-{:.3f}'.format(ci[2]['auc'], ci[2]['ci_low'], ci[2]['ci_high']),
				**plt_kwargs)
	
	ax.plot([0, 1], [0, 1], color='black', linestyle='--', lw=3, alpha=0.5)
	if axins:
		if micro_avg:
			axins.plot(mean_fpr, micro_mean_tpr, **plt_kwargs)

		if macro_avg:
			axins.plot(mean_fpr, macro_mean_tpr, **plt_kwargs)
	
	return fig, ax, axins

def plot_roc_curve_per_class_avg_mc_w_zoom(all_Y, all_Y_probs, cls_idx, fig, ax, label, lw=2.5, axins=None, color=None):
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 101)
	n_classes = all_Y[0].shape[1]
	for i in range(len(all_Y)):
		fpr = dict()
		tpr = dict()
		fpr, tpr, _ = roc_curve(all_Y[i][:, cls_idx].astype(int), all_Y_probs[i][:, cls_idx])
		roc_auc = auc(fpr, tpr)

		interp_tpr = interp(mean_fpr, fpr, tpr) #interpolate tpr at fixed values of fpr
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(roc_auc)

	print('class {} AUC: {:.3f}'.format(cls_idx, roc_auc))

	mean_tpr = np.mean(tprs, axis=0)
	mean_auc = np.mean(aucs)
	std_auc = np.std(aucs)

	plt_kwargs = {'lw': lw, 'alpha': 0.65}
	fill_kwargs = {'alpha': 0.15}

	if color:
		plt_kwargs.update({'color': color})
		fill_kwargs.update({'color': color})
	
	ax.plot(mean_fpr, mean_tpr,
		label='{}: AUC = {:.3f} $\pm$ {:.3f}'.format(label, mean_auc, std_auc),
		**plt_kwargs)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, **fill_kwargs)
	
	if axins:
		axins.plot(mean_fpr, mean_tpr,
		label='{}: AUC = {:.3f} $\pm$ {:.3f}'.format(label, mean_auc, std_auc),
		**plt_kwargs)
		axins.fill_between(mean_fpr, tprs_lower, tprs_upper, **fill_kwargs)
	
	return fig, ax, axins

def plot_roc_curve_per_class_w_zoom(all_Y, all_Y_probs, cls_idx, fig, ax, label, axins=None, **kwargs):
	fpr, tpr, _ = roc_curve(all_Y[:, cls_idx].astype(int), all_Y_probs[:, cls_idx])
	roc_auc = auc(fpr, tpr)

	# auc_score = roc_auc_score(all_Y.ravel(), all_Y_probs.ravel())

	print('class {} AUC: {:.3f}'.format(cls_idx, roc_auc))
	ax.plot(fpr, tpr, label='{} (AUC: {:0.3f})'.format(label, roc_auc), **kwargs)
		
	if axins:
		axins.plot(fpr, tpr, label='{} (AUC: {:0.3f})'.format(label, roc_auc), **kwargs)
	
	return fig, ax, axins

def plot_roc_curve_w_zoom(all_Y, all_Y_probs, fig, ax, label, axins=None, ci={}, **kwargs):
	fpr, tpr, _ = roc_curve(all_Y.astype(int), all_Y_probs)
	roc_auc = auc(fpr, tpr)

	# tpr[0] = fpr[0]
	if label is not None:
		if ci is not None:
			label = '{} AUC: {:0.3f} \n95% CI: {:.3f}-{:.3f}'.format(label, roc_auc, ci['auc_ci_low'], ci['auc_ci_high'])
		else:
			label = '{} AUC: {:0.3f}'.format(label, roc_auc)
	else:
		if ci is not None:
			label = 'AUC: {:0.3f} \n95% CI: {:.3f}-{:.3f}'.format(roc_auc, ci['auc_ci_low'], ci['auc_ci_high']) 
		else:
			label = 'AUC: {:0.3f}'.format(roc_auc)

	ax.plot(fpr, tpr, label=label, **kwargs)

	if axins:
		axins.plot(fpr, tpr, label='{} (AUC: {:0.3f})'.format(label, roc_auc), **kwargs)
		
	return fig, ax, axins

# def plot_roc_curve_ensemble_w_zoom(all_Y, all_Y_probs, fig, ax, label, axins=None, ci={}, **kwargs):
# 	fpr, tpr, _ = roc_curve(all_Y.astype(int), all_Y_probs[:,1])
# 	roc_auc = auc(fpr, tpr)
# 	# tpr[0] = fpr[0]
# 	ax.plot(fpr, tpr, label='{} \nAUC: {:.3f} \n95% CI: {:.3f}-{:.3f}'.format(label, roc_auc, ci[0]['ci_low'], ci[0]['ci_high']), **kwargs)
# 	ax.plot([0, 1], [0, 1], color='black', linestyle='--', lw=3, alpha=0.5)
# 	if axins:
# 		axins.plot(fpr, tpr, **kwargs)
		
# 	return fig, ax, axins

def plot_roc_curve_w_zoom_mc(all_Y, all_Y_probs, fig, ax, label, n_classes=3, axins=None, averaging='micro', ci={}, **kwargs):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	ovr_aucs = []
	n_classes = all_Y.shape[1]
	
	valid_classes = []
	for i in range(n_classes):
		ovr_targets = all_Y[:, i].astype(int)
		ovr_probs = all_Y_probs[:, i]
		if len(np.unique(ovr_targets)) > 1:
			fpr[i], tpr[i], _ = roc_curve(ovr_targets, ovr_probs)
			roc_auc[i] = auc(fpr[i], tpr[i])
			valid_classes.append(i)
	
	# micro average
	fpr["micro"], tpr["micro"], _ = roc_curve(all_Y.ravel(), all_Y_probs.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	
	if 'micro' in averaging:
		if label is not None:
			if ci is not None:
				label = '{} AUC: {:0.3f} \n95% CI: {:.3f}-{:.3f}'.format(label, roc_auc["micro"], ci['auc_ci_low'], ci['auc_ci_high'])
			else:
				label = '{} AUC: {:0.3f}'.format(label, roc_auc["micro"])
		else:
			if ci is not None:
				label = 'AUC: {:0.3f} \n95% CI: {:.3f}-{:.3f}'.format(roc_auc["micro"], ci['auc_ci_low'], ci['auc_ci_high']) 
			else:
				label = 'AUC: {:0.3f}'.format(roc_auc["micro"])
		ax.plot(fpr["micro"], tpr["micro"], label=label, **kwargs)
		# ax.fill_between(ci['sp'], ci['se_ci_low'], ci['se_ci_high'], alpha=0.15)
	
	# macro average 
	# need to select only classes for which the support is > 0 
	all_Y = np.take(all_Y, valid_classes, axis=1)
	all_Y_probs = np.take(all_Y_probs, valid_classes, axis=1)
	# take union of all sets of fpr points considered
	all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_classes]))
	# Then interpolate all ROC curves at these fpr points
	mean_tpr = np.zeros_like(all_fpr)
	for i in valid_classes:
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= len(valid_classes)
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	fpr['macro'][0] = 0
	tpr['macro'][0] = 0

	roc_auc['macro'] = np.mean([roc_auc[i] for i in valid_classes])
	
	if 'macro' in averaging:
		if label is not None:
			label = '{} (AUC: {:0.3f})'.format(label, roc_auc["macro"])
		else:
			label = 'AUC: {:0.3f}'.format(roc_auc["macro"])
		ax.plot(fpr["macro"], tpr["macro"], label=label, **kwargs)
	
	print(roc_auc)
	if axins:
		if 'micro' in averaging:
			axins.plot(fpr["micro"], tpr["micro"], **kwargs)
			# if len(ci) > 0:
			# 	axins.fill_between(1-ci['sp'], ci['se_ci_low'], ci['se_ci_high'], alpha=0.15)
		if 'macro' in averaging:
			axins.plot(fpr["macro"], tpr["macro"], **kwargs)
	return fig, ax, axins


def compile_results(exp_paths, num_splits=10, n_classes=2, use_site=False):
	if n_classes > 2:
		all_Y_probs, all_Y, all_test_aucs = compile_eval_results_mc(exp_paths, num_splits=num_splits, n_classes=n_classes, use_site=use_site)
	else:
		all_Y_probs, all_Y, all_test_aucs = compile_eval_results(exp_paths, num_splits=num_splits, n_classes=2, use_site=use_site)

	return all_Y_probs, all_Y, all_test_aucs



def draw_complete_roc_curve(all_Y_probs, all_Y, colors=[], plot_args=None, font_args=None, ci=None):
	averaging = plot_args.mc_averaging
	print('use multi-class averaging: ', averaging)

	# best_ids = [np.argmax(test_aucs) for test_aucs in all_test_aucs]
	# if not average:
	# 	all_Y_probs = [all_Y_probs[i][best_ids[i]] for i in range(len(labels))]
	# 	all_Y = [all_Y[i][best_ids[i]] for i in range(len(labels))]
	# 	test_set_size = all_Y_probs[0].shape[0]

	# elif not average:
	# 	print(all_test_aucs)
	# 	best_id = np.argmax(all_test_aucs[-1]) 
	# 	all_Y_probs = [all_Y_probs[i][best_id] for i in range(len(labels))]
	# 	all_Y = [all_Y[i][best_id] for i in range(len(labels))]
	# 	test_set_size = all_Y_probs[0].shape[0]
	# else:
	# 	print(all_test_aucs)
	# 	total_test_num = 0
	# 	for fold in all_Y[0]:
	# 		total_test_num += fold.shape[0]
	# 	test_set_size = int(total_test_num / len(all_Y[0])) 
	plot_title = None

	if font_args is None:
		font_args = configure_font()

	fig, ax, axins = init_roc_curve(title=plot_title, zoom_in=plot_args.zoom_in, zoom=2, 
									zoom_limits = plot_args.zoom_limits,
								   	bbox_to_anchor= plot_args.bbox_to_anchor,
								   	font_args=font_args)
	
	for exp_idx in range(len(all_Y_probs)):
		if len(colors) > 1:
			color = colors[exp_idx]
		else:
			color = None

		targets = all_Y[exp_idx][0]
		probs = all_Y_probs[exp_idx][0]
		if len(plot_args.labels) > 0:
			label = plot_args.labels[exp_idx]
		else:
			label = None 
		
		pdb.set_trace()
		if len(probs.shape) > 1:
			fig, ax, axins = plot_roc_curve_w_zoom_mc(targets, probs, 
					  								  fig, ax, label, lw=plot_args.roc_lw, 
					  								  axins=axins, alpha=plot_args.roc_alpha, ci=ci[exp_idx])
		else:
			fig, ax, axins = plot_roc_curve_w_zoom(targets, probs, 
		  								  fig, ax, label, lw=plot_args.roc_lw, 
		  								  axins=axins, alpha=plot_args.roc_alpha, ci=ci[exp_idx])
	fig, ax =  finalize_roc_curve(fig, ax, font_args)

	return fig, ax

def compile_eval_results(exp_paths, num_splits = 10, n_classes=2, use_site=False):
	# assert n_classes == 2
	all_aucs = []
	all_Y_probs = [[] for i in range(len(exp_paths))]
	all_Y = [[] for i in range(len(exp_paths))]
	for exp_path_idx in range(len(exp_paths)):
		# summary_df = pd.read_csv(os.path.join(exp_paths[exp_path_idx], 'summary.csv'))
		# all_test_aucs = summary_df['test_auc'].values
		# all_aucs.append(all_test_aucs)
		for split_idx in range(num_splits):
			fold_df = pd.read_csv(os.path.join(exp_paths[exp_path_idx], 'fold_{}.csv'.format(split_idx)))
			if use_site:
				all_targets = fold_df['site'].values
				all_probs = fold_df['site_p'].values
			else:
				all_targets = fold_df['Y'].values
				probs = fold_df[['p_{}'.format(i) for i in range(n_classes)]].values

				if n_classes == 2:
					all_probs = probs[:, 1]
				else:
					all_targets = label_binarize(all_targets, classes=[i for i in range(n_classes)])
					
					valid_classes = np.where(np.any(all_targets, axis=0))[0]
					all_targets = all_targets[:, valid_classes]
					all_probs = probs[:, valid_classes]


			all_Y_probs[exp_path_idx].append(all_probs)
			all_Y[exp_path_idx].append(all_targets)
	
	return all_Y_probs, all_Y

# def compile_eval_results_mc(exp_paths, num_splits = 10, n_classes=3):
# 	all_aucs = []
# 	all_Y_probs = [[] for i in range(len(exp_paths))]
# 	all_Y = [[] for i in range(len(exp_paths))]
# 	for exp_path_idx in range(len(exp_paths)):
# 		summary_df = pd.read_csv(os.path.join(exp_paths[exp_path_idx], 'summary.csv'))
# 		all_test_aucs = summary_df['test_auc'].values
# 		all_aucs.append(all_test_aucs)
# 		for split_idx in range(num_splits):
# 			fold_df = pd.read_csv(os.path.join(exp_paths[exp_path_idx], 'fold_{}.csv'.format(split_idx)))
# 			all_targets = fold_df['Y'].values
# 			all_probs = np.zeros((len(all_targets), n_classes))
# 			for idx in range(n_classes):
# 				all_probs[:, idx] = fold_df['p_{}'.format(idx)].values
# 			all_targets = label_binarize(all_targets, classes=[i for i in range(n_classes)])
# 			all_Y_probs[exp_path_idx].append(all_probs)
# 			all_Y[exp_path_idx].append(all_targets)
	
# 	return all_Y_probs, all_Y, all_aucs

def compile_eval_results_ensemble(exp_paths, n_classes=3):
	ci = [{} for i in range(len(exp_paths))]
	all_Y_probs = []
	all_Y = []
	for exp_path_idx in range(len(exp_paths)):
		fold_df = pd.read_csv(os.path.join(exp_paths[exp_path_idx], 'ensembled_results.csv'))
		all_targets = fold_df['Y'].values
		all_probs = np.zeros((len(all_targets), n_classes))

		if n_classes > 2:
			all_targets = label_binarize(all_targets, classes=[i for i in range(n_classes)])
			for idx in range(n_classes):
				all_probs[:, idx] = fold_df['p_{}'.format(idx)].values
				ci[exp_path_idx].update({idx:{'auc': fold_df.loc[idx, 'AUC'], 
										'ci_low':fold_df.loc[idx, 'CI_low'],
										'ci_high':fold_df.loc[idx, 'CI_high']}})
		else:
			for idx in range(n_classes):
				all_probs[:, idx] = fold_df['p_{}'.format(idx)].values
			ci[exp_path_idx].update({0:{'auc': fold_df.loc[0, 'AUC'], 
									'ci_low':fold_df.loc[0, 'CI_low'],
									'ci_high':fold_df.loc[0, 'CI_high']}})
		# all_targets = label_binarize(all_targets, classes=[i for i in range(n_classes)])
		all_Y_probs.append(all_probs)
		all_Y.append(all_targets)
		
	return all_Y_probs, all_Y, ci

def compile_multi_splits_w_descriptor(num_splits, exp_paths, descriptor_paths, dict_keys, n_classes=2):
	all_r_df = {}
	for exp_idx in range(len(exp_paths)):
		dict_key = dict_keys[exp_idx]
		results_path = exp_paths[exp_idx]
		descriptor_path = descriptor_paths[exp_idx]
		columns = ['train_num_cls_{}'.format(i) for i in range(n_classes)]
		columns.extend(['test_auc', 'test_acc'])
		index = ['split {}'.format(i) for i in range(num_splits)]
		r_df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.float32), index=index, columns= columns)
		summary_df = pd.read_csv(os.path.join(results_path, 'summary.csv'))
		all_test_aucs = summary_df['test_auc'].values
		all_test_accs = summary_df['test_acc'].values
#         all_aucs.append(all_test_aucs)
		for i in range(num_splits):
			desc_df = pd.read_csv(os.path.join(descriptor_path, 'splits_{}_descriptor.csv'.format(i)))
			fold_df = pd.read_csv(os.path.join(exp_paths[exp_idx], 'fold_{}.csv'.format(i)))
			r_df.loc[index[i], 'test_auc'] = all_test_aucs[i]
			r_df.loc[index[i], 'test_acc'] = all_test_accs[i]
			for c_idx in range(n_classes):
				r_df.loc[index[i], 'train_num_cls_{}'.format(c_idx)] = desc_df.loc[desc_df.index[c_idx], 'train']

		if dict_key not in all_r_df.keys():
			all_r_df[dict_key] = []
		all_r_df[dict_key].append(r_df.copy())
	return all_r_df





