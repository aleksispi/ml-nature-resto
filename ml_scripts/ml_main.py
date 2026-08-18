import torch
import time
from torch.utils.data import DataLoader
from ml_scripts.training import logits_to_classes
from paths import GPKGS, ZARR, ROOT, CLOUD_MASKS, DATA
import numpy as np
#from sklearn.model_selection import train_test_split
#import geopandas as gpd
#from classes import zarr_class, zarr_dataset_class
import ml_scripts.load_data as ld
# Needed to define and train CNN:
from collections import OrderedDict
from torch import nn, Tensor
from ml_scripts.Network import Network ,ResidualDownsample, ResidualBlock,AdversarialGenesis
from torchvision import models      # Models such as resnet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ml_scripts.metrics import BinaryAccuracyMetric, BinaryRecallMetric,BinaryPrecisionMetric,BinarySpecificityMetric,BinaryNPVMetric, MulticlassAccuracyMetric, MulticlassPrecisionMetric, MulticlassF1Metric
from ml_scripts.training import train_loop, plot_training
from ml_scripts.stats import stats_classification, make_cm_plot
from classes import zarr_class
import torch.nn.functional as F
from pathlib import Path
from ml_scripts.registry import MODEL_BUILDERS, BUILDERS, SAMPLE_METHODS
from ml_scripts.registry import register_model
from matplotlib.colors import ListedColormap
import geopandas as gpd
from collections import defaultdict
import classes.zarr_dataset_class as zdc
import matplotlib.colors as mcolors  
import pandas as pd   
import folium  
from matplotlib.ticker import PercentFormatter
import random
from collections import Counter
# PLOTTING:
from functions.utils import make_savedir
from functions.plotting_functions import plot_rgb
import os, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


"""
This module contains methods for training and evaluating a 2D CNN. Scroll to the bottom for usage examples. 

This module is large and therefore divided into sections:
- "main" function
- Most important models
- Run variants from the project
- Methods for saving network info, results and visualization of samples.
- Tester class to test model performance
- Misc models etc
- Actual __main__ with some examples usage
"""
def main(run_info, opt_method, learning_rate, loss_fn, number_epochs, minibatch_size, sample_method, builder, model_and_layers, cube, num_classes=1, scheduler=None, weight_decay = None, metrics={"accuracy": BinaryAccuracyMetric()}, flips=False,save_preset=True, n_plotted_samples=10, num_workers=6,ps=None):
    """
    Method that is called on to start training a 2D CNN model with specified input, model and hyperparameters. 
    metrics = on the form {'accuracy': BinaryAccuracyMetric()}
    builder: Method of extracting data from the samples (e.g. raw image, composite)
    sample_method: Method for picking out samples (e.g. all polygons first and last year)
    """
    t0=time.time()
    run_info = run_info 
    # Make directory for results, statistics, sample plots etc
    save_dir = make_savedir('ml_runs', path = ROOT, additional_info=run_info)
    train_loader, val_loader, test_loader = ld.load_data(cube,sample_method,builder,minibatch_size,ps=ps,num_workers=num_workers,flips=flips)
    print("Training size: ", len(train_loader.dataset))   

    # Peek at the very first batch to confirm batch dimension
    first_batch = next(iter(train_loader))
    X0, y0 = first_batch
    print("First batch shapes:", getattr(X0, "shape", type(X0)), getattr(y0, "shape", type(y0)))
    in_channels = X0.shape[1]
  
    # Plotting some samples
    try:
        plot_samples(train_loader,save_dir,n_samples=n_plotted_samples,randomize=True)
    except Exception:
        pass

    # Choosing a model
    model, layers = model_and_layers(in_channels, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)

    builder_name = getattr(builder, "__name__", builder.__class__.__name__)
    sample_name = getattr(sample_method, "__name__", sample_method.__class__.__name__)
    model_name = getattr(model_and_layers, "__name__", model_and_layers.__class__.__name__)

    # Save specified hyperparameters and ML structure
    save_network_info(save_dir, opt_method, learning_rate, loss_fn, number_epochs, minibatch_size, weight_decay,
                    layers, train_loader, val_loader, sample_method, builder,total_params)
    
    # Using a preset model
    if save_preset:
        np.savez(save_dir/"model_preset.npz",layers=layers,opt_method=opt_method.__name__,learning_rate=learning_rate,loss_fn=loss_fn.__class__.__name__,number_epochs=number_epochs,minibatch_size=minibatch_size,in_channels=in_channels,model_and_layers=model_name,builder=builder_name,sample_method=sample_name,num_classes=num_classes)                                           # Save network preset 
    
    # Set up the optimizer
    if weight_decay is None:
        optimizer = opt_method(model.parameters(), lr=learning_rate)
    else:    
        optimizer = opt_method(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",        # because we monitor validation loss
            factor=0.8,        # decrease the learning rate
            patience=5,        # epochs without improvement before reducing LR
            threshold=1e-3,
            verbose=True
        )

    # Train the network and print the progress
    train_loss, val_loss, metrics_res = train_loop(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        print_every=5,
        epochs=number_epochs,
        scheduler=scheduler)

    run_time=time.time()-t0
    print("RUNTIME: ", run_time/60, "minutes")
    
    # Plot the training history
    plot_training(train_loss, val_loss, save_dir, metrics_res=metrics_res,title=run_info)

    # Calculate accuracy, sensitivity, etc
    stats_train = stats_classification(model, train_loader, loss_fn=loss_fn,num_classes=num_classes ,label="Training", print_stats = False)
    stats_val = stats_classification(model, val_loader, loss_fn=loss_fn,num_classes=num_classes ,label="Validation", print_stats = False, plot_samples=True, save_dir=save_dir)
    
    #Save model
    torch.save(model.state_dict(),save_dir/"model_weights.pth")

    # Make a confusion matrix
    make_cm_plot(model, val_loader, save_dir, file_name='Confusion_matrix_val', print_stats = True, label='Validation data')
    final_metrics = {k: float(v[-1]) for k, v in metrics_res.items()}
    
    print("\nFINAL METRICS (validation):")
    for k, v in final_metrics.items():
        if k.endswith("-v"):
            print(f"{k.replace('-v',''):15}: {v:.4f}")

    save_results(
        save_dir,
        final_metrics,
        stats_train,
        stats_val,
        num_classes=num_classes
    )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Most important models. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_model 
def build_genesis(in_channels,num_classes=1):
    """
    Final custom model used in project.
    """
    layers = OrderedDict()
    layers["stem"] = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(32)
    layers["res2"] = ResidualBlock(32)

    layers["down1"] = ResidualDownsample(32, 64)
    layers["res3"] = ResidualBlock(64)

    layers["res4"] = ResidualBlock(64)

    layers["gap"]  = nn.AdaptiveAvgPool2d(1)
    layers["flat"] = nn.Flatten()

    layers["fc1"]  = nn.Linear(64, 128) #REMOVED
    layers["actf"] = nn.LeakyReLU()
    layers["out"]  = nn.Linear(128, num_classes)

    model = Network(layers, l2regularization=0).to(device)
    model.num_classes=num_classes
    return model, layers

@register_model
def build_resnet50(in_channels, num_classes: int = 1):
    """
    Builds a ResNet-50 that accepts arbitrary-channel input.
    """

    # Load pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # --- Modify input conv layer ---
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    old_conv = model.conv1
    
    # Create a new conv with same params but different input channels
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Initialize new weights (e.g. copy mean of original RGB weights) <----- DID something new here! Something about not removing pretrained weights in deeper layers
    with torch.no_grad():   
        # Sentinel-2 is [B, G, R]
        model.conv1.weight[:, 1] = old_conv.weight[:, 2]  # B
        model.conv1.weight[:, 2] = old_conv.weight[:, 1]  # G
        model.conv1.weight[:, 3] = old_conv.weight[:, 0]  # R

        # Initialize remaining channels
        for ch in range(in_channels):
            if ch not in [1, 2, 3]:
                nn.init.kaiming_normal_(model.conv1.weight[:, ch:ch+1])

    # --- Modify output layer ---
    # Original: Linear(2048, 1000)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.num_classes=num_classes
    model = model.to(device)        # Move to device
    return model, OrderedDict(model.named_modules())  

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Different run variants used in the project. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

VARIANTS={
    "baseline":{
        "sample_method":ld.first_and_last_year_balanced,
        "builder":ld.build_global_normalized_masked_composite,
        "is_stack":False
    },
    "random image":{
        "sample_method":ld.first_and_last_year,
        "builder":ld.build_random_global_normalized_masked_img,
        "is_stack":False
    },
    "vegetation indices":{
        "sample_method":ld.first_and_last_year_balanced,
        "builder":ld.build_bands_and_veg_inds_comp,
        "is_stack":False
    },
    "2nd last year":{
        "sample_method":ld.first_and_2ndlast_year,
        "builder":ld.build_global_normalized_masked_composite,
        "is_stack":False
    },
    "pasture norm":{
        "sample_method":ld.first_and_last_year_balanced,
        "builder":ld.build_masked_composite_pasture_norm,
        "is_stack":False
    },
    "monthly stack":{
        "sample_method":ld.first_and_last_year_monthly,
        "builder":ld.build_masked_composite_monthly_global_normalization,
        "is_stack":True
    },
    "monthly stack pasture norm":{
        "sample_method":ld.first_and_last_year_monthly,
        "builder":ld.build_masked_composite_monthly_pasture_normalization,
        "is_stack":True,
        "weight_decay":0.0005,
        "epochs": 50
    },
}

MODELS = {
    "genesis": build_genesis,
    "resnet": build_resnet50
}

DEFAULT_WD=0.0
DEFAULT_EPOCHS=100

def get_model_builder(model_name, cfg):
    # Only Genesis needs special handling
    if model_name == "genesis" and cfg["is_stack"]:
        return build_genesis_stack
    return MODELS[model_name]

def run_variants(cube,models=None,variants=None,overrides=None):
    if isinstance(models, str):
        models = [models]
    if isinstance(variants, str):
        variants = [variants]

    overrides = overrides or {}
    selected_models = models if models is not None else MODELS.keys()
    selected_variants = variants if variants is not None else VARIANTS.keys()
    for model_name in selected_models:
        for variant_name in selected_variants:
            cfg = VARIANTS[variant_name]
            weight_decay = cfg.get("weight_decay", DEFAULT_WD)
            epochs=cfg.get("epochs", DEFAULT_EPOCHS)
            model_builder = get_model_builder(model_name, cfg)

            #Apply overrides
            weight_decay = overrides.get("weight_decay", weight_decay)
            epochs = overrides.get("epochs", epochs)
            builder = overrides.get("builder", cfg["builder"])
            sample_method = overrides.get("sample_method", cfg["sample_method"])

            run_name = f"{model_name}_{variant_name}_wd{weight_decay}"
            print(f"\n=== Running {run_name} ===")

            main(
                run_name,
                opt_method=torch.optim.AdamW,
                learning_rate=1e-4,
                weight_decay=weight_decay,
                scheduler=None,
                cube=cube,
                loss_fn=nn.BCEWithLogitsLoss(),
                number_epochs=epochs,
                minibatch_size=64,
                sample_method=sample_method,
                builder=builder,
                model_and_layers=model_builder,
                flips=True,
            )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Methods for saving network info, results and plotting samples. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_samples(dataloader : DataLoader, save_dir,n_samples=None,randomize=True):
    """
    Creates a directory, and plots samples from dataloader. Plots all samples if n_samples is unspecified.
    """
    sample_dir = save_dir/'Input data samples'
    os.makedirs(sample_dir, exist_ok=True)
    if n_samples == None:
        n_samples = len(dataloader.dataset)

    if randomize:
        idxs = np.random.permutation(len(dataloader.dataset))
        idxs = idxs[:n_samples]
    else:
        idxs = range(n_samples)

    for idx in idxs:
        img, label = dataloader.dataset[idx]
        fig, ax = plot_rgb(img,title='')
        plt.title(f'Sample number {idx}', loc = "left")
        plt.title(f'Label {label}', loc = "right")
        plt.savefig(sample_dir/f"sample_{idx}_label_{label}.png") 
        plt.close()

def save_network_info(save_dir, opt_method,learning_rate, loss_fn,number_epochs, minibatch_size, weight_decay,
                      layers, train_loader, val_loader, sample_method, builder,tot_params):  
    """
    Saves the information about layer structure and hyperparameters to a text file in save_dir.
    """     
    file_path = save_dir/"network_info.txt"

    with open(file_path, "w") as f:
        f.write("NETWORK INFORMATION \n")
        f.write("Data: \n")
        f.write(f"Number of train samples: {len(train_loader.dataset)} \n")
        f.write(f"Number of validation samples: {len(val_loader.dataset)} \n")
        
        method_name = getattr(sample_method, "__name__", sample_method.__class__.__name__)
        f.write(f"Sample method: {method_name}\n")

        builder_name = getattr(builder, "__name__", builder.__class__.__name__)
        f.write(f"Builder: {builder_name} \n")

        f.write("\n")
        f.write("Hyperparameters: \n")
        f.write(f"Optimizer method: {opt_method.__name__} \n")
        f.write(f"Learning rate: {learning_rate} \n")
        f.write(f"Loss function: {loss_fn} \n")
        f.write(f"Number of epochs: {number_epochs} \n")
        f.write(f"Minibatch size: {minibatch_size} \n")
        f.write(f"Weight decay: {weight_decay} \n")

        f.write("\n")
        f.write(f"CNN layers: \n")
        for name, layer in layers.items():
            f.write(f"  {name}: {layer}\n")
        f.write(f"Total number of parameters: {tot_params} \n")    

def save_results(save_dir, final_metrics, stats_train, stats_val, num_classes=1):
    file_path = save_dir / "network_info.txt"

    with open(file_path, "a") as f:
        f.write("\n")
        f.write("FINAL RESULTS\n\n")

        # Training metrics from metrics_res
        f.write("Training metrics:\n")
        for k, v in final_metrics.items():
            if k.endswith("-t"):
                f.write(f"{k.replace('-t',''):15}: {v:.4f}\n")

        f.write(f"Loss            : {stats_train.get('Loss', np.nan):.4f}\n")

        if num_classes == 1:
            f.write(f"Sensitivity     : {stats_train.get('Sensitivity', np.nan):.4f}\n")
            f.write(f"Specificity     : {stats_train.get('Specificity', np.nan):.4f}\n")

        f.write("\n")

        # Validation metrics
        f.write("Validation metrics:\n")
        for k, v in final_metrics.items():
            if k.endswith("-v"):
                f.write(f"{k.replace('-v',''):15}: {v:.4f}\n")

        f.write(f"Loss            : {stats_val.get('Loss', np.nan):.4f}\n")
        if num_classes == 1:
            f.write(f"Sensitivity     : {stats_val.get('Sensitivity', np.nan):.4f}\n")
            f.write(f"Specificity     : {stats_val.get('Specificity', np.nan):.4f}\n")

def load_preset(file_path):
    """
    Load network layers and hyperparameters from specified file.
    """
    data = np.load(file_path, allow_pickle=True)
    layers = data["layers"].item()

    optimizer_name = data["opt_method"].item()  # 'Adam' e.g.
    OPTIMIZERS = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    opt_method = OPTIMIZERS[optimizer_name]

    lr = float(data["learning_rate"])

    loss_fn_name = data["loss_fn"].item()  # 'BCEWithLogitsLoss' e.g.
    LOSS_FNS = {
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()
    }
    loss_fn = LOSS_FNS[loss_fn_name]

    n_epochs = data["number_epochs"]
    minib = data["minibatch_size"]
    model = Network(layers, l2regularization=0).to(device)

    return model,layers, opt_method,lr,loss_fn,n_epochs,minib

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Class to test model performance (e.g. on val or test set)
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Test_runs():
    """
    With a path to the folder for the run, loads the trained model and performs tests
    """
    def __init__(self,run_path,num_classes=1):
        matplotlib.rcParams.update({'font.size': 14})
        self.run_path=Path(run_path)
        model_data=np.load(self.run_path/"model_preset.npz", allow_pickle=True)
        self.num_classes = num_classes
        self.model=self._get_model(model_data)
        self.model.load_state_dict(torch.load(self.run_path/'model_weights.pth'))
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if self.num_classes == 1:
            self.metrics = {
                "accuracy": BinaryAccuracyMetric(),
                "recall": BinaryRecallMetric(),
                "precision": BinaryPrecisionMetric(),
                "specificity": BinarySpecificityMetric(),
                "NPV": BinaryNPVMetric()
            }
        else:
            self.metrics = {
                "accuracy": MulticlassAccuracyMetric(),
                "precision": MulticlassPrecisionMetric(self.num_classes),
                "f1": MulticlassF1Metric(self.num_classes)
            }
        sample_name = model_data["sample_method"].item()

        if callable(sample_name):
            sample_name = sample_name.__name__

        self.sample_method = SAMPLE_METHODS[sample_name]
        builder_name = model_data["builder"].item()
        if callable(builder_name):
            builder_name = builder_name.__name__

        self.builder = BUILDERS[builder_name]
        self.minibatch_size=int(model_data["minibatch_size"])
        self.cube=self.set_cube()

    def _get_model(self,model_data):
        model_name = model_data["model_and_layers"].item()
        in_channels = int(model_data["in_channels"])
        model_and_layers = MODEL_BUILDERS[model_name]
        model,layers=model_and_layers(in_channels,num_classes=self.num_classes)
        return model
    
    def test(self,dataloader):
        # All preds etc expects shuffling to be fasle
        device=self.device
        all_preds=[]
        all_targs=[]
        self.model.eval()

        # Reset metrics
        for m in self.metrics.values():
            m.reset()

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                
                y_labels = y.to(device)

                logits = self.model(X)

                pred = logits_to_classes(logits, num_classes=self.num_classes)

                all_preds.extend(pred.cpu().numpy())
                all_targs.extend(y_labels.cpu().numpy())

                for m in self.metrics.values():
                    m.update(pred.cpu(), y_labels.cpu())

        all_preds = np.array(all_preds)
        all_targs = np.array(all_targs)
        samples = dataloader.dataset.samples

        # Safety check
        assert len(samples) == len(all_preds), "Mismatch between samples and predictions!"

        for i in range(len(samples)):
            samples[i]["predicted"] = int(all_preds[i])

        ps=self.get_poly_ids_from_loader(dataloader)
        self.report()
        return samples
    
    def validation_set_test(self):
        """
        Run standard evaluation on the validation set.

        What this does:
        - Loads train/val/test loaders using the same split as during training
        - Runs inference on the validation set
        - Produces:
            • correctness map (spatial visualization of predictions vs truth)
            • class-wise performance metrics (saved to disk)

        Purpose:
        - Primary sanity check after training
        - Used to verify that the model performs as expected on held-out validation data
        - Also useful for inspecting spatial patterns of errors
        """
        _, val_loader, _ = self.get_train_val_test_loaders()
        samples = self.test(val_loader)
        self.plot_correctness_map(samples, self.run_path, filename="correctness_map.png")
        self.plot_and_save_class_performance(samples,set_loader="val")

    def test_set_test(self):
        """
        Similar to validation_set_test, but on test data instead (also, does not call
        the plot_correctness_map method by default in this case).
        """
        _, _, test_loader = self.get_train_val_test_loaders()
        samples = self.test(test_loader)
        self.plot_and_save_class_performance(samples,set_loader="test")

    def special_validation_set_test(self):
        """
        Run validation-set evaluation using a bias-focused sampling strategy.

        What this does:
        - Uses `get_class_loader(set_loader="val")`, which:
            • overrides standard sampling
            • uses `first_and_last_year_req_imgs`
            • uses `multi_year_bias_builder`
        - Runs inference on this specially constructed dataset

        Purpose:
        - Designed for analyzing temporal bias and multi-year behavior
        - Constructs samples explicitly across years (e.g. first vs last year)
        - Helps evaluate whether the model is relying on temporal artifacts

        Notes:
        - This is NOT the same distribution as used during standard training/evaluation
        - No plots or metrics are automatically saved here
        - Typically used together with custom analysis functions (e.g. bias or time-based tests)
        """
        loader = self.get_class_loader(set_loader="val")
        samples = self.test(loader)

    def special_test_set_test(self):
        """
        Run test-set evaluation using a bias-focused sampling strategy.

        What this does:
        - Uses `get_class_loader(set_loader="test")`
        - Applies same special sampling as validation version:
            • multi-year bias builder
            • controlled temporal sampling
        - Runs inference on this dataset

        Purpose:
        - Evaluate model robustness under controlled temporal conditions
        - Useful for detecting:
            • reliance on year-specific artifacts
            • temporal shortcut learning

        Notes:
        - Distribution differs from standard test set
        - No automatic plotting or saving
        - Intended for advanced evaluation workflows (bias analysis)
        """
        loader = self.get_class_loader(set_loader="test")
        samples = self.test(loader)

    def pasture_bias_test_stats(self, og_builder, set_loader="val"):
        """
        Compute temporal bias statistics across years since restoration start.

        What this does:
        - Iterates over years (0–5) relative to the start of restoration
        - For each year:
            • Builds a dataset using `get_pasture_loader(year, ...)`
            • Runs model inference on that dataset
            • Separates samples into positive (label=1) and negative (label=0)
            • Computes prediction rates:
                - pos_rate: fraction of positive samples predicted as positive
                - neg_rate: fraction of negative samples predicted correctly

        Purpose:
        - To analyze whether the model’s predictions depend on temporal position
        (e.g. years since restoration)
        - Helps detect temporal biases or shortcut learning
        (e.g. model distinguishing "year" rather than ecological changes)

        Input:
        - og_builder: specifies original input construction ("stack" or "composite")
        - set_loader:
            • "val" (default) → uses validation set
            • "test" → uses test set

        Notes:
        - This is NOT a standard performance metric (e.g. accuracy or AUROC)
        - It is a diagnostic tool for analyzing temporal behaviour of the model
        - The year=0 case uses negative samples as a baseline reference
        - Output is a dictionary:
            results[year] = {
                "pos_rate": ...,
                "neg_rate": ...,
                "n": number of pastures
            }

        Interpretation:
        - If performance varies strongly across years, this may indicate:
            • temporal bias
            • reliance on year-specific artifacts
        """

        # OBS: Needs og_builder as "stack" or "composite"
        results = {}
        for year in np.arange(0,6):
            print("Year after start: ", year)
            pasture_loader=self.get_pasture_loader(year,og_builder=og_builder,set_loader=set_loader)
            samples=self.test(pasture_loader)

            # Separate positive samples (year k)
            pos_samples = [s for s in samples if s["label"] == 1]
            neg_samples = [s for s in samples if s["label"] == 0]
            if year==0:
                pos_samples=neg_samples
    
            # % predicted positive for each group
            if len(pos_samples) > 0:
                pos_rate = np.mean([s["predicted"] for s in pos_samples])
            else:
                pos_rate = np.nan
    
            if len(neg_samples) > 0:
                neg_rate = 1-np.mean([s["predicted"] for s in neg_samples])
            else:
                neg_rate = np.nan
    
            # Number of pastures
            pastures = set(s["p"] for s in samples)
    
            results[year] = {
                "pos_rate": pos_rate,
                "neg_rate": neg_rate,
                "n": len(pastures)
            }
    
        return results
    
    def plot_pasture_bias(self, results,set_loader="test"):
        years = sorted(results.keys())
        pos_vals = [results[y]["pos_rate"] for y in years]
        neg_vals = [results[y]["neg_rate"] for y in years]
        n_vals   = [results[y]["n"] for y in years]

        # --- Positive plot ---
        plt.figure()
        plt.plot(years, pos_vals, color="tab:green", marker='o', label=f"% restored")
        plt.xticks(years)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.xlabel("Years after restoration start")
        plt.ylabel("% predicted restored")
        #plt.title(f"Predicted restored on the last year of years included in normalization")

        plt.grid(True)
        plt.legend()

        for x, n in zip(years, n_vals):
            plt.text(x, 0.05, f"n={n}", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.run_path / f"pasture_bias_positive_{set_loader}_set.png",bbox_inches='tight')
        plt.close()

        # --- Negative plot ---
        plt.figure()
        plt.plot(years, neg_vals, color="tab:orange", marker='o', label="Negative accuracy")
        
        if set_loader=="test":
            neg_acc_glob=0.67
        else:
            neg_acc_glob=0.69
        plt.axhline(neg_acc_glob, color='red', linestyle=':', linewidth=1.5,
                    label="global norm negative accuracy")

        plt.xticks(years)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.xlabel("Years after restoration start")
        plt.ylabel("Accuracy on negative samples")
        #plt.title(f"Accuracy on negative samples depending on normalization years")
        plt.grid(True)
        plt.legend(loc="center")

        for x, n in zip(years, n_vals):
            plt.text(x, 0.05, f"n={n}", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.run_path / f"pasture_bias_negative_{set_loader}_set.png",bbox_inches='tight')
        plt.close()

    def when_restored_test_stats(self, og_builder="stack", set_loader="val"):
        """
        Analyze when (in time) the model predicts restoration.

        What this does:
        - Iterates over years within a restoration sequence (0 → rest_len-1)
        - For each year:
            • Constructs a dataset containing samples at that specific year
            using `get_single_year_loader(...)`
            • Runs model inference
            • Computes the average predicted probability for that year

        Purpose:
        - To estimate how the model's predictions evolve over time
        - Answers the question:
            "At which point in the restoration timeline does the model
            start classifying a pasture as restored?"

        Inputs:
        - og_builder:
            • Specifies how input data is constructed ("stack", "composite", etc.)
        - set_loader:
            • "val" (default) → validation set
            • "test" → test set

        Output:
        - Dictionary of the form:
            all_results[rest_len] = {
                "curve": {
                    year_0: mean prediction,
                    year_1: mean prediction,
                    ...
                },
                "n": number of unique pastures
            }

        Interpretation:
        - The "curve" represents model confidence over time
        - Ideal behavior:
            • low predictions early in restoration
            • gradually increasing predictions over time
        - Deviations may indicate:
            • temporal bias
            • reliance on year-specific artifacts
            • inability to capture gradual ecological change

        Notes:
        - This is NOT a standard classification metric
        - It is a diagnostic tool for temporal model behavior
        - Currently uses rest_len = 6 (fixed)
        """

        all_results = {}
        rest_lens=np.arange(6,7)

        for rest_len in rest_lens:
            year_results = {}
            pasture_ids = set()
            years=np.arange(0,rest_len)
            for year in years:
                loader=self.get_single_year_loader(rest_len=rest_len,year=year,og_builder=og_builder,set_loader=set_loader)
                samples=self.test(loader)
                preds = [s["predicted"] for s in samples]
                pasture_ids.update([s["p"] for s in samples])
                if len(preds) > 0:
                    year_results[year] = np.mean(preds)
                else:
                    year_results[year] = np.nan

            all_results[rest_len] = {
                    "curve": year_results,
                    "n": len(pasture_ids)
                }
        return all_results
    
    def plot_when_restored(self, all_results,set_loader="val"):
        for rest_len, data in all_results.items():
            
            year_results = data["curve"]
            n_pastures = data["n"]

            years = sorted(year_results.keys())
            values = [year_results[y] for y in years]

            plt.plot(years, values, marker='o',color="tab:green", label=f"% Restored")
            plt.xlabel("Years after restoration start")
            plt.ylabel("% predicted restored")
            #plt.title(f"Predicted restored for pastures with restoration time {rest_len-2}")
            plt.yticks(np.arange(0, 1.1, 0.1))

            plt.xticks(years)
            plt.ylim(0, 1)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.run_path/f"when_restored_{rest_len}_{set_loader}_set.png",bbox_inches='tight')
            plt.close()

    def res_durs_test_stats(self, og_builder="stack", set_loader="val"):
        all_results={}
        rest_lens=np.arange(2,8)

        for rest_len in rest_lens:
            loader=self.get_res_durs_loader(rest_len=rest_len,og_builder=og_builder,set_loader=set_loader)
            samples=self.test(loader)

            pos_samples = [s for s in samples if s["label"] == 1]
            neg_samples = [s for s in samples if s["label"] == 0]

            if len(pos_samples) > 0:
                pos_acc = np.mean([s["predicted"] == 1 for s in pos_samples])
            else:
                pos_acc = np.nan
    
            if len(neg_samples) > 0:
                neg_acc = np.mean([s["predicted"] == 0 for s in neg_samples])
            else:
                neg_acc = np.nan
    
            all_results[rest_len] = {
                "pos_acc": pos_acc,
                "neg_acc": neg_acc,
                "n": len(samples)
            }
    
        return all_results
    
    def plot_res_durations(self, results, og_builder="stack",set_loader="val"):
        rest_lens = sorted(results.keys())

        # x-axis = duration - 2
        x_vals = [r - 2 for r in rest_lens]

        pos_vals = [results[r]["pos_acc"] for r in rest_lens]
        neg_vals = [results[r]["neg_acc"] for r in rest_lens]
        n_vals  = [results[r]["n"] for r in rest_lens]

        x = np.arange(len(x_vals))
        width = 0.35
        all_samples = []
        for r in rest_lens:
            loader = self.get_res_durs_loader(rest_len=r, og_builder=og_builder,set_loader=set_loader)
            all_samples.extend(self.test(loader))
    
        pos_mean = np.mean([s["predicted"] == 1 for s in all_samples if s["label"] == 1])
        neg_mean = np.mean([s["predicted"] == 0 for s in all_samples if s["label"] == 0])

        plt.figure(figsize=(10, 5))
        plt.axhline(
            pos_mean,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label="Avg positive"
        )

        plt.axhline(
            neg_mean,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Avg negative")

        
        bars_pos = plt.bar(
            x - width/2, pos_vals,
            width,
            color="tab:green",
            edgecolor="black",
            label="Positive accuracy"
        )

        bars_neg = plt.bar(
            x + width/2, neg_vals,
            width,
            color="tab:orange",
            edgecolor="black",
            label="Negative accuracy"
        )

        plt.xticks(x, x_vals)
        plt.xlabel("Restoration duration")
        plt.ylabel("Accuracy")

        plt.ylim(0, 1.3)
        plt.axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="100%")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))

        plt.grid(axis="y", linestyle=":", linewidth=1,alpha=0.8)
        plt.legend(fontsize=10)

        # Annotate sample counts
        for i, n in enumerate(n_vals):
            plt.text(x[i], 0.05, f"n={int(n/2)}", ha="center", fontsize=12)

        plt.tight_layout()
        plt.savefig(self.run_path / f"res_duration_{set_loader}_set.png")
        plt.close()

    def plot_combined_restoration(self, pasture_results, when_results, set_loader="val"):
        plt.figure()

        # --- Pasture bias ---
        years_pb = sorted(pasture_results.keys())
        pb_vals = [pasture_results[y]["pos_rate"] for y in years_pb]

        plt.plot(
            years_pb,
            pb_vals,
            marker='o',
            color="tab:cyan",
            label="Pasture bias (% restored)"
        )

        # --- When restored ---
        for rest_len, data in when_results.items():
            years_wr = sorted(data["curve"].keys())
            wr_vals = [data["curve"][y] for y in years_wr]

            plt.plot(
                years_wr,
                wr_vals,
                marker='o',
                color="tab:green",
                label=f"When restored)"
            )

        # --- Formatting ---
        plt.xlabel("Years after restoration start")
        plt.ylabel("% predicted restored")

        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

        plt.grid(axis="y", linestyle=":", linewidth=1)
    
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # remove duplicates
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)

        plt.tight_layout()
        plt.savefig(self.run_path / f"combined_restoration_{set_loader}.png")
        plt.close()

    def report(self):
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value
            print(f"{name}: {value:.4f}")
        return results

    def get_train_val_test_loaders(self):
        train_loader, val_loader, test_loader = ld.load_data(cube=self.cube,method=self.sample_method,
                                                             builder=self.builder,batch_size=self.minibatch_size)
        return train_loader, val_loader, test_loader
    
    def get_pasture_loader(self,year,og_builder="stack",set_loader="test"):
        """Returns the pasture loader of a year which is "year after start year"
        """
        if set_loader=="val":
            _,test_loader,_=self.get_train_val_test_loaders()
        elif set_loader=="test":
            _,_,test_loader=self.get_train_val_test_loaders()
        test_ps=self.get_poly_ids_from_loader(test_loader)
        sample_method=ld.pasture_bias_sample_method(year,og_builder=og_builder)
        test_samples = ld.idx_to_samples(test_ps,self.cube,sample_method,self.builder)
        test_ds  = zdc.ZarrDataset(self.cube, test_samples)
        test_loader=DataLoader(test_ds, batch_size=self.minibatch_size, shuffle=False, num_workers=0)

        return test_loader

    def get_single_year_loader(self,rest_len,year,og_builder,set_loader="val"):
        _,val_loader,test_loader=self.get_train_val_test_loaders()

        if set_loader=="val":
            ps=self.get_poly_ids_from_loader(val_loader)
        elif set_loader=="test":
            ps=self.get_poly_ids_from_loader(test_loader)
        
        sample_method=ld.single_year_sample_method(rest_len,year,og_builder=og_builder)
        samples=ld.idx_to_samples(ps,self.cube,sample_method,self.builder)
        ds=zdc.ZarrDataset(self.cube,samples)
        loader=DataLoader(ds, batch_size=self.minibatch_size, shuffle=False, num_workers=0)

        return loader
    
    def get_class_loader(self,set_loader="val"):
        """
        Constructs a special DataLoader for bias / temporal analysis.

        Key differences from standard loaders:
        - Uses `first_and_last_year_req_imgs` sampling:
            → ensures specific temporal coverage (e.g. early vs late years)
        - Uses `multi_year_bias_builder`:
            → constructs inputs to explicitly probe temporal bias

        Purpose:
        - To analyze whether the model learns true signals vs temporal artifacts
        - Used by "special_*" evaluation methods

        Note:
        - This does NOT reflect the natural training distribution
        - It is intended purely for controlled evaluation experiments
        """
        if set_loader=="val":
            _,loader,_=self.get_train_val_test_loaders()
        elif set_loader=="test":
            _,_,loader=self.get_train_val_test_loaders()
        
        ps=self.get_poly_ids_from_loader(loader)
        sample_method=ld.first_and_last_year_req_imgs
        builder=ld.multi_year_bias_builder(masking="normal",normalization="pasture",builder_type="random_stack",random_img=True)

        samples=ld.idx_to_samples(ps,self.cube,sample_method,builder)
        ds=zdc.ZarrDataset(self.cube,samples)
        loader=DataLoader(ds, batch_size=self.minibatch_size,shuffle=False, num_workers=0)
        return loader
        
    def get_res_durs_loader(self,rest_len,og_builder,set_loader="val"):
        
        if set_loader=="val":
            _,loader,_=self.get_train_val_test_loaders()
        elif set_loader=="test":
            _,_,loader=self.get_train_val_test_loaders()

        ps=self.get_poly_ids_from_loader(loader)

        sample_method=ld.res_duration_sample_method(rest_len=rest_len,og_builder=og_builder)

        samples=ld.idx_to_samples(ps,self.cube,sample_method,self.builder)
        ds=zdc.ZarrDataset(self.cube,samples)
        loader=DataLoader(ds, batch_size=self.minibatch_size,shuffle=False, num_workers=0)
        return loader

    def get_poly_ids_from_loader(self,loader):
        """Returns unique ps in a loader
        """
        ps = set()
        for i in range(len(loader.dataset)):
            ps.add(loader.dataset.samples[i]["p"])
        return np.array(list(ps))

    def aggregate_by_year(self,samples):
        stats = defaultdict(lambda: {0: {"correct": 0, "total": 0},
                                     1: {"correct": 0, "total": 0}})
        for s in samples:
            
            times = self._flatten_times(s["times"])       
            t0 = times[0]  # or min(times), depending on intent

            year = self.cube.timestamps[t0].year
            label = s["label"]
            pred = s["predicted"]

            stats[year][label]["total"] += 1
            if pred == label:
                stats[year][label]["correct"] += 1

        return stats
    
    def _flatten_times(self,times):
        # If already flat (e.g. [1,2,3])
        if len(times) > 0 and not isinstance(times[0], (list, tuple)):
            return times

        # If nested (e.g. [[1,2],[3]])
        return [t for group in times for t in group]

    def prepare_plot_data(self,samples, label):
        years = sorted(samples.keys())
        acc = []
        counts = []

        for y in years:
            total = samples[y][label]["total"]
            correct = samples[y][label]["correct"]

            if total > 0:
                acc.append(correct / total * 100)
            else:
                acc.append(0)

            counts.append(total)

        return years, acc, counts
    
    def plot_class_performance(self, years, acc, counts, title):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(years))
        bars = ax1.bar(x, acc, edgecolor="black")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 110)  # allow space above 100
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.set_yticks(np.arange(0, 111, 10))
        ax1.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

        # Add red line at 100%
        ax1.axhline(100, color='red', linestyle='--', linewidth=1)

        # Annotate counts (always above bars now)
        for bar, n in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2,
                     f"n={n}",
                     ha='center', va='bottom', fontsize=12)
        plt.tight_layout()

    def plot_and_save_class_performance(self,samples,set_loader="val"):
        samples=self.aggregate_by_year(samples)
        neg_years,neg_acc,neg_counts=self.prepare_plot_data(samples,label=0)
        pos_years,pos_acc,pos_counts=self.prepare_plot_data(samples,label=1)

        self.plot_class_performance(neg_years,neg_acc,neg_counts,"negative samples")
        plt.tight_layout()
        plt.savefig(self.run_path/f"negative_year_plot_{set_loader}.png")
        plt.close()
        self.plot_class_performance(pos_years,pos_acc,pos_counts,"positive samples")
        plt.tight_layout()
        plt.savefig(self.run_path/f"positive_year_plot_{set_loader}.png")
        plt.close()

    def plot_correctness_map(self, samples, save_dir, filename="correctness_map.png"):
        gdf=self.cube.get_gdf()
        # gdf = gdf.to_crs(epsg=4326)               # Convert to CRS
        # Load Sweden
        map_path = DATA / "map_countries.shp"
        sweden = gpd.read_file(map_path)
        sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']
        sweden = sweden.to_crs(epsg=4326)

        # Convert geometries to centroids for plotting
        centroids = gdf.copy()
        centroids["geometry"] = gdf.geometry.centroid
        centroids = centroids.to_crs(epsg=4326)               # Convert to CRS
        # Introduce correctness column 
        centroids["correctness"] = np.nan     # -1 both incorrect, 0: 1 correct, 1: both correct, nan: not in set

        # Loop through samples
        for sample in samples:
            p = sample["p"]
            pred = sample["predicted"]
            label = sample["label"]
            val = centroids.loc[p, "correctness"]   # Current correctness values
            if pred == label:   # If correct pred
                if pd.isna(val): # First occurence
                    centroids.loc[p, "correctness"] = 1
                elif val == -1:
                    centroids.loc[p, "correctness"] = 0
            else:   # If incorrect pred
                if pd.isna(val): # First occurence
                    centroids.loc[p, "correctness"] = -1
                elif val== 1:
                    centroids.loc[p, "correctness"] = 0

        # PNG
        fig, ax = plt.subplots(figsize=(6, 8))
        sweden.boundary.plot(ax=ax, color="black", linewidth=1)

        cmap = ListedColormap(["red", "orange","green"])
        gdf_filtered = centroids[centroids["correctness"].notna()].copy()
        gdf_filtered["correctness_str"] = gdf_filtered["correctness"].map({ # Map values to labels
            -1: "Incorrect",
            0: "One right",
            1: "Correct"
        })
        
        gdf_filtered["correctness_str"] = pd.Categorical(
            gdf_filtered["correctness_str"],
            categories=["Incorrect", "One right", "Correct"],
            ordered=True
        )

        gdf_filtered.plot(
            ax=ax,
            column="correctness_str",
            categorical = True,
            cmap=cmap,
            markersize=5,
            legend=True,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        plt.savefig(save_dir / filename)
        plt.close()

        # HTML
        m = folium.Map(
            location=[62, 15],   # center of Sweden
            zoom_start=5,
            tiles="CartoDB positron" 
        )

        for idx, row in gdf_filtered.iterrows():
            color_map = {
            "Incorrect": "red",
            "One right": "orange",
            "Correct": "green"
            }

            # Plot centroid
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color_map[row["correctness_str"]],
                fill=True,
                fill_opacity=0.8
            ).add_to(m)

        map_path=DATA/"map_countries.shp"
        sweden = gpd.read_file(map_path)
        sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']
        sweden = sweden.to_crs(epsg=4326)

        folium.GeoJson(
            sweden,
            name="Sweden boundary",
            style_function=lambda x: {"color": "black", "weight": 1, "fillOpacity": 0},
        ).add_to(m)

        m.save(save_dir/(filename+".html"))

    def set_cube(self,zarr_path=ZARR,gpkg_path=GPKGS/"final02-20.gpkg",cloud_path=CLOUD_MASKS/"cloud_mask_04_17_thresholds_20_15_10.npy"):
        cube=zarr_class.Zarr(zarr_path,gpkg_path,cloud_path=cloud_path)
        cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
        cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")
        return cube
    
    def set_cube_from_cube(self,cube):
        self.cube=cube

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Misc models etc. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_model
def build_resnet18(in_channels, num_classes: int = 1):
    """
    Builds a ResNet-18 that accepts arbitrary-channel input.
    """

    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # --- Modify input conv layer ---
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    old_conv = model.conv1
    
    # Create a new conv with same params but different input channels
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Initialize new weights (e.g. copy mean of original RGB weights) <----- DID something new here! Something about not removing pretrained weights in deeper layers
    with torch.no_grad():   
        # Sentinel-2 is [B, G, R]
        model.conv1.weight[:, 1] = old_conv.weight[:, 2]  # B
        model.conv1.weight[:, 2] = old_conv.weight[:, 1]  # G
        model.conv1.weight[:, 3] = old_conv.weight[:, 0]  # R

        # Initialize remaining channels
        for ch in range(in_channels):
            if ch not in [1, 2, 3]:
                nn.init.kaiming_normal_(model.conv1.weight[:, ch:ch+1])

    # --- Modify output layer ---
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)        # Move to device
    return model, OrderedDict(model.named_modules())  
  
@register_model
def build_custom_model(loader):
    """
    Builds our custom CNN model.
    """
    
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()
    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(3, 3))          
    layers["act1"] = nn.LeakyReLU()
    layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    # layers["drop2"] = nn.Dropout2d(0.1)
    layers["conv3"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()
    #layers["drop3"] = nn.Dropout2d(0.1)
    layers["mp1"] = nn.MaxPool2d((2, 2))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer
    layers["flat"] = nn.Flatten()
    layers["mlp1"] = nn.Linear(n_features, 128)
    layers["amlp1"] = nn.LeakyReLU()
    #layers["drop4"] = nn.Dropout(0.1)
    layers["out"] = nn.Linear(128,1)

    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_custom_model_dropout_tests(loader):
    """
    Using our custom model, but testing where dropout should be placed if any in order to reduce overfitting. 
    """
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()
    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(3, 3))          
    layers["act1"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((3, 3))         # <---------- Added

    layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    layers["drop2"] = nn.Dropout2d(0.1)
    layers["mp2"] = nn.MaxPool2d((3, 3))         # <---------- Added

    layers["conv3"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()
    layers["drop3"] = nn.Dropout2d(0.15)        # <---------- Increased dropout rate
    layers["mp3"] = nn.MaxPool2d((2, 2))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer
    layers["flat"] = nn.Flatten()
    layers["mlp1"] = nn.Linear(n_features, 1000)    # <---------- Changed to 1000      
    layers["amlp1"] = nn.LeakyReLU()
    layers["drop4"] = nn.Dropout(0.5)       # <---------- Increased dropout rate

    layers["mlp2"] = nn.Linear(1000,128)
    layers["amlp2"] = nn.LeakyReLU()
    layers["drop5"] = nn.Dropout(0.35)      # <---------- Increased dropout rate

    layers["mlp3"] = nn.Linear(128,64)
    layers["amlp3"] = nn.LeakyReLU()
    layers["drop6"] = nn.Dropout(0.25)      # <---------- Added

    # Output layer 
    layers["out"] = nn.Linear(64, 1)
 
    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_custom_model_experimental(loader):
    """
    Using our custom model, but testing where dropout should be placed if any in order to reduce overfitting. 
    """
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()
    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(5, 5))          
    layers["act1"] = nn.LeakyReLU()
    layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((2, 2))        
    layers["conv3"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()   
    layers["conv4"] = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 3))          
    layers["act4"] = nn.LeakyReLU()

    layers["mp2"] = nn.MaxPool2d((3, 3))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer
    layers["drop1"] = nn.Dropout2d(0.5) 
    layers["flat"] = nn.Flatten()
    layers["mlp1"] = nn.Linear(n_features, 128)  # <---------- Changed to 1000      
    layers["amlp1"] = nn.LeakyReLU()
    layers["drop2"] = nn.Dropout(0.4)      # <---------- Increased dropout rate

    # Output layer 
    layers["out"] = nn.Linear(128, 1)
 
    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_custom_model_experimental_improved(loader):
    """
    Using our custom model, but testing where dropout should be placed if any in order to reduce overfitting. 
    """
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()
    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(3, 3))          
    layers["act1"] = nn.LeakyReLU()
    layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((2, 2))        
    layers["conv3"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()   
    layers["conv4"] = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 3))          
    layers["act4"] = nn.LeakyReLU()
    layers["mp2"] = nn.MaxPool2d((3, 3))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer
    #<-----added
    #layers["drop1"] = nn.Dropout2d(0.35)  #<----removed
    layers["flat"] = nn.Flatten()
    #layers["bn"] = nn.BatchNorm1d(n_features)
    layers["mlp1"] = nn.Linear(n_features, 128)     
    layers["amlp1"] = nn.LeakyReLU()
    layers["drop2"] = nn.Dropout(0.2)      # <---------- Decreased dropout rate

    # Output layer 
    layers["out"] = nn.Linear(128, 1)
 
    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_custom_model_experimental_improved_months(loader):
    """
    Using our custom model, but testing where dropout should be placed if any in order to reduce overfitting. 
    """
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()

    layers["conv1"] = nn.Conv2d(in_channels=36, out_channels=64,kernel_size=(3, 3))          
    layers["act1"] = nn.LeakyReLU()
    layers["conv2"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((2, 2))        
    layers["conv3"] = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()   
    layers["conv4"] = nn.Conv2d(in_channels=256, out_channels=256,kernel_size=(3, 3))          
    layers["act4"] = nn.LeakyReLU()
    layers["mp2"] = nn.MaxPool2d((3, 3))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer

    #layers["drop1"] = nn.Dropout2d(0.3) # <----removed
    layers["flat"] = nn.Flatten()
    #layers["bn"] = nn.BatchNorm1d(n_features)
    layers["mlp1"] = nn.Linear(n_features, 128)     
    layers["amlp1"] = nn.LeakyReLU()
    layers["drop2"] = nn.Dropout(0.3)      # <---------- Decreased dropout rate

    # Output layer 
    layers["out"] = nn.Linear(128, 1)
 
    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_custom_model_experimental2(loader):
    """
    Using our custom model, but tweaking for stability
    """
    # Container for CNN layers that will be used to construct the Network
    layers = OrderedDict()

    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(3, 3))
    layers["act1"] = nn.LeakyReLU()
    layers["conv2"] = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3))          
    layers["act2"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((2, 2))        

    layers["conv3"] = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=(3, 3))          
    layers["act3"] = nn.LeakyReLU()   
    layers["conv4"] = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=(3, 3))          
    layers["act4"] = nn.LeakyReLU()
    layers["mp2"] = nn.MaxPool2d((3, 3))

    # Use one input image to compute the number of nodes after convolution layers.
    first_batch = next(iter(loader))        # Get one batch from dataloader
    x_trn0 = first_batch[0][0].unsqueeze(0)        # One input image
    n_features = nn.Sequential(layers)(Tensor(x_trn0)).numel()  # Passes one input through the network

    # Fully connected layer
    layers["drop1"] = nn.Dropout2d(0.35) 
    layers["flat"] = nn.Flatten()
    layers["mlp1"] = nn.Linear(n_features, 128)  
    layers["amlp1"] = nn.LeakyReLU()
    layers["drop2"] = nn.Dropout(0.4)

    # Output layer 
    layers["out"] = nn.Linear(128, 1)

    model = Network(layers, l2regularization=0).to(device)
    return model, layers

@register_model
def build_genesis_stack(in_channels, num_classes=1):
    """
    New model after previous failed
    """
    layers = OrderedDict()
    layers["stem"] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(64)
    layers["res2"] = ResidualBlock(64)

    layers["down1"] = ResidualDownsample(64, 128)
    layers["res3"] = ResidualBlock(128)

    layers["res4"] = ResidualBlock(128)

    layers["gap"]  = nn.AdaptiveAvgPool2d(1)
    layers["flat"] = nn.Flatten()

    layers["fc1"]  = nn.Linear(128, 256) #REMOVED
    layers["actf"] = nn.LeakyReLU()
    layers["out"]  = nn.Linear(256, num_classes)

    model = Network(layers, l2regularization=0).to(device)
    model.num_classes=num_classes
    return model, layers

@register_model 
def build_genesis_adv(in_channels, num_classes=1, num_years=8):

    layers = OrderedDict()

    layers["stem"] = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(32)
    layers["res2"] = ResidualBlock(32)

    layers["down1"] = ResidualDownsample(32, 64)
    layers["res3"] = ResidualBlock(64)
    layers["res4"] = ResidualBlock(64)

    layers["gap"]  = nn.AdaptiveAvgPool2d(1)
    layers["flat"] = nn.Flatten()

    backbone = nn.Sequential(layers)
    model = AdversarialGenesis(
        backbone,
        feature_dim=64,
        num_classes=num_classes,
        num_years=num_years
    ).to(device)

    return model, layers

@register_model 
def build_genesis__stack_adv(in_channels, num_classes=1, num_years=8):

    layers = OrderedDict()

    layers["stem"] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(64)
    layers["res2"] = ResidualBlock(64)

    layers["down1"] = ResidualDownsample(64, 128)
    layers["res3"] = ResidualBlock(128)
    layers["res4"] = ResidualBlock(128)

    layers["gap"]  = nn.AdaptiveAvgPool2d(1)
    layers["flat"] = nn.Flatten()

    backbone = nn.Sequential(layers)

    model = AdversarialGenesis(
        backbone,
        feature_dim=64,
        num_classes=num_classes,
        num_years=num_years
    ).to(device)

    return model, layers

@register_model
def build_genesis2(loader):
    """
    New model after previous failed
    """
    layers = OrderedDict()

    layers["stem"] = nn.Conv2d(12, 32, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(32)
    layers["res2"] = ResidualBlock(32)

    layers["down1"] = ResidualDownsample(32, 64)
    layers["res3"] = ResidualBlock(64)

    layers["res4"] = ResidualBlock(64)   
    layers["res5"] = ResidualBlock(64)

    layers["gap"]  = nn.AdaptiveAvgPool2d(1)
    layers["flat"] = nn.Flatten()

    layers["fc1"]  = nn.Linear(64, 128) #REMOVED
    layers["actf"] = nn.LeakyReLU()
    layers["drop1"]= nn.Dropout(0.2)
    layers["out"]  = nn.Linear(128, 1)

    model = Network(layers, l2regularization=0).to(device)
    return model, layers

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N,)
        # targets: (N,) in {0,1}
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
MAIN
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":  
    # Define data cube. The cube can be modified in any way before starting to train (e.g. changing cloud mask)
    cube=zarr_class.Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_15_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    if False:
        # The following line will start training a network 
        main("EXAMPLE RUN", opt_method = torch.optim.AdamW, learning_rate=0.0001, weight_decay=0.0 ,scheduler=None,cube=cube ,loss_fn=nn.BCEWithLogitsLoss(), number_epochs=100,
        minibatch_size=64,sample_method=ld.first_and_last_year_balanced, builder=ld.build_global_normalized_masked_composite, model_and_layers=build_genesis)
        # This does the exact same thing
        #run_variants(cube,models=["genesis"],variants=["baseline"])

    # In order to run tests on a previously trained model, we define a "tester" object as following
    print("\n<------------- Creating tester object --------------------------------------------------------->\n")
    tester = Test_runs("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/best_models/CNN/custom_cnn_monthly stack pasture norm_wd0.01/")

    if True:
        # For example, a test on the validation set is done by
        print("\n<------------- Evaluating on the standard validation set -------------------------------------->\n")
        tester.validation_set_test()  # change to test_set_test function if test set instead of val set

    if False:
        # Testing pasture norm bias
        print("\n<------------- Condućting pasture bias evaluation on the standard validation set -------------->\n")
        results = tester.pasture_bias_test_stats(og_builder="stack", set_loader="val")  # "val" or "test"
        tester.plot_pasture_bias(results, set_loader="val")

    if False:
        # Testing in which year a pasture is classified as restored
        print("\n<------------- Doing what-year-restored assessment on the standard validation set ------------->\n")
        results = tester.when_restored_test_stats(og_builder="stack", set_loader="val")  # "val" or "test"
        tester.plot_when_restored(results, set_loader="val")
        
    #print("\n<------------- Done with all evaluations ------------->\n")

        # NOTE: More tests are available (!). See Test_runs class for more info

        # Multi-year bias test (i.e. year detection)
        # Not sure if this is 100% right but should be something like this
        # cube.set_cluster_gdf(gdf_with_clusters)
        # metrics = {
        #    "accuracy": MulticlassAccuracyMetric(),
        #    "precision": MulticlassPrecisionMetric(8),
        #    "f1": MulticlassF1Metric(8),
        # }

        # gdf_with_clusters=gpd.read_file(GPKGS/"polygons_with_clusters_dist_1500.gpkg")
        # sample_method=ld.multi_year_bias_test(gdf_with_clusters,cube,balanced=False,og_builder="random_stack")
        # builder=ld.multi_year_bias_builder(masking="normal",normalization="pasture",builder_type="random_stack",random_img=True)
        
        # for p, year, _ in sample_method.samples:
        #    cube.cache_cluster_inverse_mask(p, year)

        # main("MULTI YEAR BIAS TEST EXAMPLE", opt_method = torch.optim.AdamW, learning_rate=0.0001, weight_decay=0.0005 ,scheduler=None,cube=cube , metrics=metrics ,loss_fn=nn.CrossEntropyLoss(), number_epochs=100,num_classes=8,
        #                 minibatch_size=64,sample_method=sample_method, builder=builder ,model_and_layers=build_genesis_stack,num_workers=8,flips=True)