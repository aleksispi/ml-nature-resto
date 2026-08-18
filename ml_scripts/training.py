import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
from ml_scripts import Network
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
import os


def prepare_targets(y, num_classes=1):
    if num_classes == 1:
        return y.float().unsqueeze(1)   # (B,)  to (B,1)
    else:
        return y.long()                # (B,)

def logits_to_classes(logits, num_classes=1):
    if num_classes == 1:
        return (logits > 0).long().squeeze(1)  # (B,1) to (B,)
    else:
        return logits.argmax(dim=1)     




def test(*, model : Network, dataloader : DataLoader, loss_fn, metrics=[]):
    """
    Test a model on a set of data.

    Args:
        model (Network): The network.
        dataloader (DataLoader): DataLoader with data to test.
        loss_fn (Loss): Loss function, e.g. nn.MSELoss.
        metrics (iterable): Additional metrics to update.

    Returns:
        loss (float): Mean error over all batches.
    """

    model.eval()
    loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_label=y
            y = prepare_targets(y,num_classes=model.num_classes)                                              # OBS! This is added

            logits = model(X)
         
            loss += loss_fn(logits, y).item() * len(X)

            pred=logits_to_classes(logits,num_classes=model.num_classes)
 
            for m in metrics:
                m.update(pred, y_label)
        
    return loss / len(dataloader.dataset)


def train_loop(*, model : Network, train_dataloader : DataLoader,
               val_dataloader : DataLoader = None, loss_fn,
               optimizer : torch.optim.Optimizer, epochs : int, scheduler=None, 
               print_every:int = 100, metrics=None, print_final=True, train_on_timeseries = False):
    """
    Train and optionally test a model.

    Args:
        model (Network): The network.
        train_dataloader (DataLoader): Training data.
        val_dataloader (DataLoader, optional): Validation data.
        loss_fn (Loss): Loss function, e.g. nn.MSELoss.
        optimizer (Optimizer): An optimizer from torch.optim.
        epochs (int): Number of epochs to train for.
        print_every (int, optional): Print loss every so many epochs. Defaults to 100.
        metrics (dict(name: metric), optional): Record/print these additional metrics.
        print_final(bool, optional): Print final metrics. Defaults to True.

    Returns:
        train_losses (list(float)): Training loss during each epoch.
        val_losses (list(float)): Validation loss after each epoch.
        metrics_res (dict(name: list(float))): Values of metrics after each epoch.
    """ 
    train_losses = []
    val_losses = []
    val_loss = np.nan

    # Move metrics to CPU/GPU and prepare for their output
    metrics = {name: m.to(device) for name, m in (metrics or {}).items()}
    metrics_res = {name+"-t": [] for name in metrics.keys()}
    metrics_res.update({name+"-v": [] for name in metrics.keys()})

    for t in range(epochs):
        for m in metrics.values():
            m.reset()
        if train_on_timeseries:                                                # ADDED 23/4 (forgot to git commit before change, I'm sorry)
            train_loss = train_epoch_timeseries(model=model, dataloader=train_dataloader,
                           loss_fn=loss_fn, optimizer=optimizer,
                           metrics=metrics.values())
        else:    
            train_loss = train_epoch(model=model, dataloader=train_dataloader,
                            loss_fn=loss_fn, optimizer=optimizer,
                            metrics=metrics.values())
            
        train_losses.append(train_loss)
        for name, m in metrics.items():
            metrics_res[name+"-t"].append(m.compute().cpu())

        if val_dataloader is not None:
            for m in metrics.values():
                m.reset()
            if train_on_timeseries:                                                # ADDED 23/4 (forgot to git commit before change, I'm sorry)
                val_loss = test_timeseries(dataloader=val_dataloader, model=model,
                        loss_fn=loss_fn, metrics=metrics.values())
            else:
                val_loss = test(dataloader=val_dataloader, model=model,
                                loss_fn=loss_fn, metrics=metrics.values())
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            val_losses.append(val_loss)
            for name, m in metrics.items():
                metrics_res[name+"-v"].append(m.compute().cpu())

        if (print_every > 0 and t % print_every == 0) or (
                print_every >= 0 and t + 1 == epochs):
            extras = [f" {n} {v[-1]:<7f}" if torch.isreal(v[-1])
                      else f" {n} {v[-1]}"
                      for n, v in metrics_res.items()]
            print(f"Epoch {t+1:<7d} train {train_loss:<7f} "
                  f" validation {val_loss:<7f}", "".join(extras))
    if print_final:
        print("\n** Validation metrics after training **\n"
              f"Loss {val_losses[-1]:<7g}")
        for n, v in metrics_res.items():
            if torch.isreal(v[-1]):
                print(f"{n} {v[-1]:<7g}")
            else:
                print(f"{n}:")
                print(v[-1])
        print()
    return train_losses, val_losses, metrics_res

def train_epoch(*, model : Network, dataloader : DataLoader,
                loss_fn, optimizer : torch.optim.Optimizer, metrics=[]):
    """
    Train a model for a single epoch.

    Args:
        model (Network): The network.
        dataloader (DataLoader): Batch DataLoader with training data.
        loss_fn (Loss): Loss function, e.g. nn.MSELoss.
        optimizer (Optimizer): The optimizer used to update the network.
        metrics (iterable): Additional metrics to update.

    Returns:
        train_loss (float): Training error over all batches.
    """
    model.train()
    train_loss = 0
    
    for X, y in dataloader:
        X, y = X.to(device,non_blocking=True), y.to(device, non_blocking=True)   # Move data to GPU if necessary
        y_label=y
        y = prepare_targets(y,num_classes=model.num_classes)                                                 # OBS! This is added
        optimizer.zero_grad()   # Reset the gradients

        # Compute prediction error
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item() * len(X)
        # loss = loss + model.regularization_loss()

        pred=logits_to_classes(logits,num_classes=model.num_classes)
        for m in metrics:
            m.update(pred.detach(), y_label)

        # Backpropagation
        loss.backward()
        
        optimizer.step()

    return train_loss / len(dataloader.dataset)

def plot_training(train_loss, val_loss, save_dir,filename = f"training_loss_history.png", metrics_res={},title="Training loss history"):
    "Plot the training history"
    fig = plt.figure(figsize=(13,7))
    fig.suptitle(title, fontsize=14)

    ax = fig.add_subplot(1,2,1)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    ax.grid(True)
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    plt.legend(loc='best')

    ax = fig.add_subplot(1,2,2)
    plt.ylabel('Metric')
    plt.xlabel('Epoch')
    for name, res in metrics_res.items():
        if torch.isreal(res[0]):
            plt.plot(res, label=name)
    ax.grid(True)
    #ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_yticks(np.linspace(0.4, 1, 13))  # Should be 0.4, 0.45, ... etc
    plt.legend(loc='best')

    plt.savefig(save_dir/filename) 
    plt.close()

def train_epoch_timeseries(*, model : Network, dataloader : DataLoader,
                loss_fn, optimizer : torch.optim.Optimizer, metrics=[]):
    """
    Train a model for a single epoch.

    Args:
        model (CNNLSTM): The model.
        dataloader (DataLoader): Batch DataLoader with training data.
        loss_fn (Loss): Loss function, e.g. nn.MSELoss.
        optimizer (Optimizer): The optimizer used to update the network.
        metrics (iterable): Additional metrics to update.

    Returns:
        train_loss (float): Training error over all batches.
    """
    model.train()
    train_loss = 0
    
    for X, seq_lens, y in dataloader:
        X, y = X.to(device,non_blocking=True), y.to(device, non_blocking=True).long()   # Move data to GPU if necessary
        y_label=y
        y = prepare_targets(y,num_classes=model.num_classes)
        optimizer.zero_grad()   # Reset the gradients

        # Compute prediction error
        logits = model(X, seq_lens)
        loss = loss_fn(logits, y)
        train_loss += loss.item() * X.size(0)

        pred=logits_to_classes(logits,num_classes=model.num_classes)

        for m in metrics:
            m.update(pred.detach(), y_label)



        # # Compute prediction error
        # logits = model(X, seq_lens)

        # # Sanity checks (development / debugging)       --> Added
        # assert logits.dim() == 2              # (B, num_classes)
        # assert y.dim() == 1                   # (B,)
        # assert logits.size(0) == y.size(0)

        # loss = loss_fn(logits, y)
        # train_loss += loss.item() * X.size(0)       # X.size(0) is batch size?

        # for m in metrics:
        #     m.update(logits.detach(), y)

        # Backpropagation
        loss.backward()
        
        optimizer.step()

    return train_loss / len(dataloader.dataset)

def test_timeseries(*, model : Network, dataloader : DataLoader, loss_fn, metrics=[]):
    """
    Test a model on a set of data.

    Args:
        model (Network): The network.
        dataloader (DataLoader): DataLoader with data to test.
        loss_fn (Loss): Loss function, e.g. nn.MSELoss.
        metrics (iterable): Additional metrics to update.

    Returns:
        loss (float): Mean error over all batches.
    """

    model.eval()
    loss = 0
    with torch.no_grad():
        for X, seq_lens, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_label=y
            y = prepare_targets(y,num_classes=model.num_classes)

            logits = model(X, seq_lens)
         
            loss += loss_fn(logits, y).item() * X.size(0)

            pred= logits_to_classes(logits, num_classes=model.num_classes)

            for m in metrics:
                m.update(pred, y_label)

            # logits = model(X, seq_lens)
         
            # loss += loss_fn(logits, y).item() * X.size(0) 
 
            # for m in metrics:
            #     m.update(logits, y)
        
    return loss / len(dataloader.dataset)

