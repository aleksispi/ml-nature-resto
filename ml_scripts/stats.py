import torch
from ml_scripts import Network
from ml_scripts.training import prepare_targets, logits_to_classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, classification_report
import itertools
from functions.plotting_functions import plot_rgb, plot_image_grid
import os

def get_targs_preds_loss(dataloader, model, loss_fn=None):
    all_preds = []
    all_targs = []

    model.eval()
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_raw = y.to(device)

            # prepare targets for loss
            y_prepared = prepare_targets(y_raw, num_classes=model.num_classes)

            logits = model(X)

            # LOSS
            if loss_fn is not None:
                loss += loss_fn(logits, y_prepared).item() * len(X)

            # PREDICTIONS (THE KEY FIX)
            pred = logits_to_classes(logits, num_classes=model.num_classes)

            all_preds.extend(pred.cpu().numpy())
            all_targs.extend(y_raw.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targs = np.array(all_targs)

    return all_targs, all_preds, loss

def stats_classification(
    model,
    dataloader,
    *,
    label: str,
    loss_fn=None,
    print_stats=True,
    plot_samples=False,
    save_dir='.',
    num_classes=1,
    train_on_timeseries=False 
):
    
    if train_on_timeseries:
        all_targs, all_preds, loss = get_targs_preds_loss_timeseries(
            dataloader, model, loss_fn
        )
    else:
        all_targs, all_preds, loss = get_targs_preds_loss(
            dataloader, model, loss_fn
        )
    

    acc = (all_preds == all_targs).mean()

    stats = {
        "Accuracy": acc
    }

    # ✅ Binary-specific metrics only when needed
    if num_classes == 1:
        t = all_targs.astype(bool)
        p = all_preds.astype(bool)

        tp = np.logical_and(p, t).sum()
        fn = np.logical_and(~p, t).sum()
        tn = np.logical_and(~p, ~t).sum()
        fp = np.logical_and(p, ~t).sum()

        stats["Sensitivity"] = tp / (tp + fn + 1e-8)
        stats["Specificity"] = tn / (tn + fp + 1e-8)

    if loss_fn is not None:
        stats["Loss"] = loss / len(dataloader.dataset)

    if print_stats:
        print(f"*** STATISTICS for {label} Data ***")
        for l, v in stats.items():
            print(f"{l:15} {v:.4f}")
        print()

    return stats

#def stats_classification(model : Network, dataloader : DataLoader,     
#                         *, label : str, loss_fn = None, print_stats = True, plot_samples = False, save_dir =  '.', train_on_timeseries = False):
#    """ 
#    Print classification statistics.
#
#    Args:
#        model (Network): The model.
#        dataloader (DataLoader): Batch DataLoader.
#        label (str): Training, test etc.
#        loss (optional): Loss function.
#
#    Returns:
#        None.
#    """
#    if train_on_timeseries:
#        X0,seq_lens0,y0 = next(iter(dataloader))      # First batch
#    else:
#        X0,y0 = next(iter(dataloader))      # First batch      
# 
#   
#    if y0.dim() == 1:
#        # Binary classification
#        if train_on_timeseries:
#            all_targs, all_preds, loss = get_targs_preds_loss_timeseries(dataloader, model, loss_fn)
#        else:
#            all_targs, all_preds, loss = get_targs_preds_loss(dataloader, model, loss_fn)    
#    
#        nof_p, tp, tn = [k.sum() for k in [all_targs, all_preds[all_targs], ~all_preds[~all_targs]]]
#        stats = {'Accuracy': (tp + tn) / len(all_targs),
#                 'Sensitivity': tp / nof_p,
#                 'Specificity': tn / (len(all_targs) - nof_p)}
#        if plot_samples:
#            plot_samples_with_preds(dataloader,label,all_preds,save_dir)
#
#    else:
#        # TODO: fix this
#        # One-hot
#        pred = pred.argmax(axis=1)
#        targ = targ.argmax(axis=1)
#        stats = {'Accuracy': (pred == targ).sum() / len(targ)}
#        
#
#    if loss_fn is not None:
#        stats['Loss'] = loss/len(dataloader.dataset)  
#
#    if print_stats:
#        print(f"*** STATISTICS for {label} Data ***")
#        for l, v in stats.items(): 
#            print(f'{l:15} {v:.4f}')
#        print()
# 
#    return stats

def plot_samples_with_preds(dataloader,dataset_label,preds,save_dir,n_samples=10,timeseries_sample=False):
    """
    Plots some samples along with the model prediction.
    """
    sample_dir = save_dir/f"{dataset_label} sample predictions"
    os.makedirs(sample_dir, exist_ok=True)

    idxs = np.random.permutation(len(preds))
    idxs = idxs[:n_samples]  

    for idx in idxs:
        img, label = dataloader.dataset[idx]
        if timeseries_sample:
            fig = plot_image_grid(img)      # img is actually an im series
            fig.text(0.01, 0.99, f"Sample number {idx}", ha="left", va="top",fontsize=36)
            fig.text(0.99, 0.99, f'True label {label} \n Predicted label: {preds[idx][0].astype(int)}', ha="right", va="top",fontsize=36)
        else:
            fig, ax = plot_rgb(img,title='')
            plt.title(f'Sample number {idx}', loc = "left")
            plt.title(f'True label {label} \n Predicted label: {preds[idx][0].astype(int)}', loc = "right")
        plt.savefig(sample_dir/f"sample_{idx}_true_{label}_pred_{preds[idx][0].astype(int)}.png") 

def plot_confusion_matrix(cm, target_names, savedir, file_name, title='Confusion matrix',
                          cmap=None, normalize=True):
    "Plot a confusion matrix"

    plt.rcParams.update({'font.size': 14})
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8)) 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\n'
               f'accuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    
    plt.tight_layout()
    plt.savefig(savedir/file_name)
    plt.close()
    # plt.show()

def make_cm_plot(
    model,
    dataloader: DataLoader,
    savedir,
    file_name="Confusion_matrix",
    print_stats=True,
    label="Test data"
):
    """
    Compute and plot the confusion matrix for both binary and multiclass.
    """

    if print_stats:
        print(f'*** Result for {label} ***')

    # ✅ Get predictions & targets (already fixed function)
    all_targs, all_preds, _ = get_targs_preds_loss(dataloader, model)

    # ✅ Ensure integer arrays
    all_targs = all_targs.astype(int)
    all_preds = all_preds.astype(int)

    # ✅ Determine number of classes dynamically
    num_classes = max(all_targs.max(), all_preds.max()) + 1

    # ✅ Basic metrics
    acc = (all_preds == all_targs).mean()

    if print_stats:
        print(f'accuracy:   {acc:.4f}\n')

    # ✅ Class names
    class_names = [f'class {i}' for i in range(num_classes)]

    # ✅ Classification report (safe for multiclass)
    if print_stats:
        print(classification_report(all_targs, all_preds, target_names=class_names))

    # ✅ Confusion matrix
    cm = confusion_matrix(all_targs, all_preds)

    # ✅ Plot
    plot_confusion_matrix(
        cm=cm,
        normalize=False,
        target_names=class_names,
        savedir=savedir,
        file_name=file_name,
        title=f"Confusion Matrix, {label}"
    )

#def make_cm_plot(model, dataloader : DataLoader, savedir, file_name = "Confusion_matrix", print_stats = True, label='Test data'):
#    """
#    Compute and plot the confusion matrix
#    """
#    if print_stats:
#        print(f'*** Result for {label} ***')
#    all_targs, all_preds, __ = get_targs_preds_loss(dataloader, model)
#    #num_classes = np.shape(all_targs)[0]
#    #y = model.predict(inp)
#    d_class = all_targs
#    y_class = all_preds
#    num_classes = 2         # Binary classification
#
#    if print_stats:
#        print(f'log_loss:   {log_loss(all_targs, all_preds):.4f}')
#    #print(f'log_loss:   {log_loss(trg, y):.4f}')
#    #d_class = trg.argmax(axis=1)
#    #y_class = y.argmax(axis=1)
#    acc = (y_class==d_class).mean()
#
#    if print_stats:
#        print(f'accuracy:   {acc:.4f}\n')
#
#    class_names = [f'class {i}' for i in range(num_classes)]
#
#    if print_stats:
#        print(classification_report(d_class, y_class, target_names=class_names))
#
#    confuTst = confusion_matrix(d_class, y_class)
#    plot_confusion_matrix(cm           = confuTst,
#                          normalize    = False,
#                          target_names = class_names,
#                          savedir = savedir,
#                          file_name = file_name,
#                          title        = f"Confusion Matrix, {label}")
    
# def get_targs_preds_loss_timeseries(dataloader: DataLoader, model : Network, loss_fn = None):
#     """
#     Loops through the dataloader and gathers all targets, predictions and total loss.
#     """
#     all_preds=[]
#     all_targs=[]
#     model.eval()
#     total_loss = 0.0
#     total_samples = 0

#     with torch.no_grad():
#         for X, seq_lens, y in dataloader:
#             X, y = X.to(device), y.to(device).long()
                                           
#             logits = model(X, seq_lens)
            
#             # Predicted class indices
#             preds = logits.argmax(dim=1)  # (B,)

#             # Accumulate predictions and targets on CPU
#             all_preds.append(preds.cpu())
#             all_targs.append(y.cpu())

#             # Loss accumulation (optional)
#             if loss_fn is not None:
#                 loss = loss_fn(logits, y)
#                 total_loss += loss.item() * X.size(0)
#                 total_samples += X.size(0)

#     all_preds = torch.cat(all_preds).numpy()
#     all_targs = torch.cat(all_targs).numpy()

#     avg_loss = None
#     if loss_fn is not None and total_samples > 0:
#         avg_loss = total_loss / total_samples

#     return all_targs, all_preds, avg_loss


#def get_targs_preds_loss_timeseries(dataloader: DataLoader, model : Network, loss_fn = None):
#    """
#    Loops through the dataloader and gathers all targets, predictions and total loss.
#    """
#    all_preds=[]
#    all_targs=[]
#    model.eval()
#    loss = 0
#    with torch.no_grad():
#        for X, seq_lens, y in dataloader:
#            X, y = X.to(device), y.to(device)
#            y = y.unsqueeze(1).float()                                                  
#            pred = model(X,seq_lens)
#            all_preds.extend((pred >= .0).cpu().numpy())        # Move to CPU and make into np array
#            all_targs.extend((y >= .5).cpu().numpy())
#            if loss_fn is not None:
#                loss += loss_fn(pred, y).item() * X.size(0)
#    
#    all_preds = np.array(all_preds)
#    all_targs = np.array(all_targs)
#
#    return all_targs, all_preds, loss    


def get_targs_preds_loss_timeseries(dataloader, model, loss_fn=None):
    all_preds = []
    all_targs = []
    loss = 0

    model.eval()
    with torch.no_grad():
        for X, seq_lens, y in dataloader:
            X = X.to(device)
            y_labels = y.to(device)

            y_loss = prepare_targets(y_labels, model.num_classes)

            logits = model(X, seq_lens)

            if loss_fn is not None:
                loss += loss_fn(logits, y_loss).item() * X.size(0)

            pred = logits_to_classes(logits, model.num_classes)

            all_preds.extend(pred.cpu().numpy())
            all_targs.extend(y_labels.cpu().numpy())

    return np.array(all_targs), np.array(all_preds), loss
