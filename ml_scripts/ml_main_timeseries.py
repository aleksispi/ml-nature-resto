import torch
import time
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from ml_scripts.Network import Network ,ResidualDownsample, ResidualBlock
from ml_scripts.CombinedNetwork import CNNLSTM, CNNBlock
from ml_scripts.training import train_loop, plot_training, logits_to_classes
from ml_scripts.stats import stats_classification, make_cm_plot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from ml_scripts.metrics import BinaryAccuracyMetric, BinaryRecallMetric,BinaryPrecisionMetric,BinarySpecificityMetric,BinaryNPVMetric, MulticlassAccuracyMetric, MulticlassPrecisionMetric, MulticlassF1Metric
from collections import OrderedDict
from paths import GPKGS, ZARR, ROOT, DATA
from classes import zarr_class
import ml_scripts.load_data as ld
from ml_scripts.registry import MODEL_BUILDERS, BUILDERS, SAMPLE_METHODS
from ml_scripts.registry import register_model
# Plotting:
from functions.utils import make_savedir
from functions.plotting_functions import plot_rgb, plot_image_grid
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# OOga booga?
from ml_scripts.ml_main import Test_runs
import geopandas as gpd


"""
This module contains methods for training and evaluating a CNN-LSTM. Scroll to the bottom for usage examples. 

This module is large and therefore divided into sections:
- "main" function
- Most important model
- Methods for saving network info, results and visualization of samples.
- Tester class to test model performance
- Misc models etc
- Actual __main__ with some examples usage
"""

def main(run_info, opt_method, learning_rate, loss_fn, number_epochs, minibatch_size, sample_method, builder, model_and_layers, cube, num_classes, temp_mode, weight_decay = None, metrics={"accuracy": BinaryAccuracyMetric()},save_preset=True, n_plotted_samples=10, num_workers=6,ps=None, flips=True):
    t0=time.time()
   
    run_info = run_info 
    # Make directory for results, statistics, sample plots etc
    save_dir = make_savedir('ml_runs_timeseries', path = ROOT, additional_info=run_info)

    train_loader, val_loader, test_loader = ld.load_data(cube,sample_method,builder,minibatch_size,ps=ps,num_workers=num_workers,pad_in_T_dim = True,flips=flips)
    print("Training size: ", len(train_loader.dataset)) 

    # Peek at the very first batch to confirm batch dimension
    X0, seq_lens0, y0 = next(iter(train_loader))       # X0 dim: (B, T, C, H, W)
        
    print("First batch shapes: ", X0.shape, seq_lens0.shape, y0.shape)

    # Plotting some samples
    plot_samples(train_loader,save_dir,n_samples=n_plotted_samples,randomize=True)

    # Take a single image (used to calculate dimensions for the CNN)        
    single_im_sample = X0[0,0,:,:,:]       
    C, H, W = single_im_sample.shape

    # num_classes = 2

    # model = CNNLSTM(cnn_layers, hidden_lstm_dim, num_lstm_layers, bidir=bidir,temp_mode=temp_mode).to(device)
    model, cnn_layers = model_and_layers(C,num_classes,temp_mode)
    # Getting some attributes to save network info 
    hidden_lstm_dim = model.hidden_dim
    num_lstm_layers = model.num_layers
    bidir = model.bidir
    temp_mode = model.temp_mode
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)

    
    builder_name = getattr(builder, "__name__", builder.__class__.__name__)
    sample_name = getattr(sample_method, "__name__", sample_method.__class__.__name__)
    model_name = getattr(model_and_layers, "__name__", model_and_layers.__class__.__name__)
    
    # Save specified hyperparameters and ML structure
    save_network_info(save_dir, opt_method, learning_rate, loss_fn, number_epochs, minibatch_size, weight_decay, bidir, flips, temp_mode,
                      cnn_layers, hidden_lstm_dim, num_lstm_layers,train_loader, val_loader, sample_method, builder, total_params)
    
    # Save network preset 
    if save_preset:
        np.savez(save_dir/"model_preset.npz",layers=cnn_layers,opt_method=opt_method.__name__,learning_rate=learning_rate,loss_fn=loss_fn.__class__.__name__,
                 number_epochs=number_epochs,minibatch_size=minibatch_size,in_channels=C,model_and_layers=model_name,builder=builder_name,sample_method=sample_name)                                           
    
    
   # Set up the optimizer
    if weight_decay is None:
        optimizer = opt_method(model.parameters(), lr=learning_rate)
    else:    
        optimizer = opt_method(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
        train_on_timeseries = True)
    
    run_time=time.time()-t0
    print("RUNTIME: ", run_time/60, "minutes")
    # TODO: save stats and results, etc

    # Plot the training history
    plot_training(train_loss, val_loss, save_dir, metrics_res=metrics_res,title=run_info)

    # Calculate accuracy, sensitivity, etc
    stats_train = stats_classification(model, train_loader, loss_fn=loss_fn, label="Training", print_stats = False,train_on_timeseries=True)
    stats_val = stats_classification(model, val_loader, loss_fn=loss_fn, label="Validation", print_stats = False, plot_samples=False, save_dir=save_dir,train_on_timeseries=True)
    # TODO: make so that plot_samples works
    # Make a confusion matrix
    # make_cm_plot(model, val_loader, save_dir, file_name='Confusion_matrix_val', print_stats = True, label='Validation data')

    #Save model
    torch.save(model.state_dict(),save_dir/"model_weights.pth")

    # Make a confusion matrix
    # make_cm_plot(model, val_loader, save_dir, file_name='Confusion_matrix_val', print_stats = True, label='Validation data')
    # final_metrics = {k: float(v[-1]) for k, v in metrics_res.items()}

    # Save results to network info file
    save_results(save_dir,metrics_res['accuracy-t'][-1],stats_train['Loss'],stats_train['Sensitivity'],stats_train['Specificity'],
                metrics_res['accuracy-v'][-1],stats_val['Loss'],stats_val['Sensitivity'],stats_val['Specificity'])
     
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Most important model. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_genesis_layers(input_dim=12):
    """
    New model after previous failed
    """
    layers = OrderedDict()
    layers["stem"] = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(32)
    layers["res2"] = ResidualBlock(32)

    layers["down1"] = ResidualDownsample(32, 64)
    layers["res3"] = ResidualBlock(64) 

    layers["res4"] = ResidualBlock(64)

    return layers

@register_model
def build_genesis_cnn_lstm(in_channels,num_classes,temp_mode):
    """
    Builds a CNN-LSTM model with genesis (without classification head) as the CNN block.
    """
    genesis_layers = build_genesis_layers()
    model = CNNLSTM(genesis_layers, hidden_dim=64, num_layers=1, num_classes = num_classes, bidir=False,temp_mode=temp_mode).to(device)
    return model, genesis_layers

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
        im_series, label = dataloader.dataset[idx]
        fig = plot_image_grid(im_series,fig_title = '')
        fig.text(0.01, 0.99, f"Sample number {idx}", ha="left", va="top",fontsize=36)
        fig.text(0.99, 0.99, f"Label {label}", ha="right", va="top",fontsize=36)

        plt.savefig(sample_dir/f"sample_{idx}_label_{label}.png") 
        plt.close()


def save_network_info(
    save_dir,
    opt_method,
    learning_rate,
    loss_fn,
    number_epochs,
    minibatch_size,
    weight_decay,
    bidir,
    flips,
    temp_mode,
    cnn_layers,
    hidden_lstm_dim,
    num_lstm_layers,
    train_loader,
    val_loader,
    sample_method,
    builder,
    tot_params,
):
    file_path = save_dir / "network_info.txt"

    with open(file_path, "w") as f:
        f.write("NETWORK INFORMATION\n")

        # Data
        f.write("Data:\n")
        f.write(f"Number of train samples: {len(train_loader.dataset)}\n")
        f.write(f"Number of validation samples: {len(val_loader.dataset)}\n")

        method_name = getattr(sample_method, "__name__", sample_method.__class__.__name__)
        builder_name = getattr(builder, "__name__", builder.__class__.__name__)

        f.write(f"Sample method: {method_name}\n")
        f.write(f"Builder: {builder_name}\n")
        f.write(f"Bidirectional: {bidir}\n")
        f.write(f"Flips: {flips}\n")
        f.write(f"Temp mode: {temp_mode}\n")

        # Hyperparameters
        f.write("\nHyperparameters:\n")
        opt_name = getattr(opt_method, "__name__", opt_method.__class__.__name__)

        f.write(f"Optimizer method: {opt_name}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Loss function: {loss_fn}\n")
        f.write(f"Number of epochs: {number_epochs}\n")
        f.write(f"Minibatch size: {minibatch_size}\n")
        f.write(f"Weight decay: {weight_decay}\n")

        # CNN layers
        f.write("\nCNN layers:\n")
        for name, layer in cnn_layers.items():
            f.write(f"  {name}: {layer}\n")

        # LSTM
        f.write("\nLSTM:\n")
        f.write(f"Hidden dim: {hidden_lstm_dim}\n")
        f.write(f"Number of layers: {num_lstm_layers}\n")

        # Total params
        f.write(f"\nTOTAL number of parameters: {tot_params}\n")

def save_results(save_dir,train_acc,train_loss,train_sens,train_spec,val_acc,val_loss,val_sens,val_spec):       # OBS exakt samma som ml_main
    """
    Updates the textfile with network info to also contain the results after training.
    """
    file_path = save_dir/"network_info.txt"

    with open(file_path, "a") as f:
        f.write("\n")
        f.write("FINAL RESULTS \n")
        f.write(f"Training accuracy: {train_acc} \n")
        f.write(f"Training sensitivity: {train_sens} \n")
        f.write(f"Training specificity: {train_spec} \n")
        f.write(f"Training loss: {train_loss} \n")
        f.write("\n")
        f.write(f"Validation accuracy: {val_acc} \n")
        f.write(f"Validation sensitivity: {val_sens} \n")
        f.write(f"Validation specificity: {val_spec} \n")
        f.write(f"Validation loss: {val_loss} \n")

    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Class to test model performance. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Test_runs_ts(Test_runs):
    """
    The same class as deefined in ml_main, but with some methods overshadowed.
    """
    def __init__(self, run_path, temp_mode, num_classes=1):
        self.temp_mode = temp_mode  # Additional attribute in cnn-lstm
        super().__init__(run_path, num_classes) # Inherit from super


    def _get_model(self,model_data):
        model_name = model_data["model_and_layers"].item()
        in_channels = int(model_data["in_channels"])
        model_and_layers = MODEL_BUILDERS[model_name]

        model,layers=model_and_layers(in_channels,num_classes=self.num_classes,temp_mode=self.temp_mode)
        return model
    
    def get_train_val_test_loaders(self):
        """
        Adapted for CNN-LSTM by padding in T dim.
        """
        train_loader, val_loader, test_loader = ld.load_data(cube=self.cube,method=self.sample_method,pad_in_T_dim=True,
                                                             builder=self.builder,batch_size=self.minibatch_size)
        return train_loader, val_loader, test_loader
    

    def test(self,dataloader):
        """
        All preds etc expects shuffling to be false.
        Adapted for CNN-LSTM by including seq_lens.
        """
        device=self.device

        all_preds=[]
        all_targs=[]
        self.model.eval()

        # Reset metrics
        for m in self.metrics.values():
            m.reset()

        with torch.no_grad():
            for X, seq_lens, y in dataloader:
                X, y = X.to(device), y.to(device)
                
                y_labels = y.to(device)

                logits = self.model(X,seq_lens)

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
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Misc models etc. 
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_cnn_layers():
    layers = OrderedDict()
    layers["conv1"] = nn.Conv2d(in_channels=12, out_channels=32,kernel_size=(3, 3))          
    layers["act1"] = nn.LeakyReLU()
    layers["mp1"] = nn.MaxPool2d((2, 2))
    return layers

def build_custom_cnn_layers():
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

    return layers

@register_model
def build_modified_genesis_cnn_lstm(in_channels,num_classes):
    """
    Builds a CNN-LSTM model with a modified genesis (without classification head) as the CNN block.
    """
    layers = OrderedDict()
    layers["stem"] = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
    layers["act0"] = nn.LeakyReLU()

    layers["res1"] = ResidualBlock(32)
    layers["res2"] = ResidualBlock(32)

    layers["down1"] = ResidualDownsample(32, 64)
    layers["res3"] = ResidualBlock(64) 

    layers["res4"] = ResidualBlock(64)

    layers["down2"] = ResidualDownsample(64, 128)
    layers["res5"] = ResidualBlock(128)


    model = CNNLSTM(layers, hidden_dim=64, num_layers=1, num_classes=num_classes,bidir=False,temp_mode="attention").to(device)
    return model, layers

@register_model
def build_larger_hidden_dim(in_channels,num_classes):
    """
    Builds a CNN-LSTM model with genesis (without classification head) as the CNN block.
    """
    genesis_layers = build_genesis_layers()
    model = CNNLSTM(genesis_layers, hidden_dim=128, num_layers=1, num_classes = num_classes, bidir=False,temp_mode="attention").to(device)
    return model, genesis_layers

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
MAIN
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":  

    # Define data cube. The cube can be modified in any way before starting to train (e.g. changing cloud mask)
    cube=zarr_class.Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    # The following line will start training a network 
    main("EXAMPLE RUN", opt_method = torch.optim.AdamW, learning_rate=0.0001, num_classes=1, weight_decay=0.0,loss_fn=nn.BCEWithLogitsLoss(), number_epochs=100, minibatch_size=12, sample_method=ld.first_and_last_year_balanced,
          builder=ld.build_raw_masked_pasture, model_and_layers=build_genesis_cnn_lstm, cube=cube, metrics={"accuracy": BinaryAccuracyMetric()},temp_mode="final")  # temp_mode is final hidden state or attention
    
    # The tests are run similarly to ml_main
    Test_runs("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/best_models/CNN/cnn_lstm_pature_norm/",temp_mode="final")  # <-- NOTE Test_runs_ts instead of Test_runs
    tester.test_set_test()

    # YEAR BIAS TEST
    # cube.set_cluster_gdf(gdf_with_clusters)
    
    # metrics = {
    #     "accuracy": MulticlassAccuracyMetric(),
    #     "precision": MulticlassPrecisionMetric(8),
    #     "f1": MulticlassF1Metric(8),
    # }

    # sample_method=ld.multi_year_bias_test(gdf_with_clusters,cube,balanced=False,og_builder="raw")
    # builder=ld.multi_year_bias_builder(masking="normal",normalization="pasture",builder_type="raw",random_img=False)
    
    # for p, year, _ in sample_method.samples:
    #     cube.cache_cluster_inverse_mask(p, year)


    # main("MULTI TEST PASTURE LOW THRESH", opt_method = torch.optim.AdamW, learning_rate=0.0001, weight_decay=0.0005 ,cube=cube , metrics=metrics ,loss_fn=nn.CrossEntropyLoss(), number_epochs=5,num_classes=8,
    #                  minibatch_size=12,sample_method=sample_method, builder=builder ,model_and_layers=build_genesis_cnn_lstm,num_workers=8,flips=True)
    #

 
