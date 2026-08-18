import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from paths import GPKGS, ZARR, DATA
import numpy as np
from sklearn.model_selection import train_test_split
import geopandas as gpd
from classes import zarr_class, zarr_dataset_class
import time
import matplotlib.pyplot as plt
import functions.plotting_functions as pf
import functions.stat_analysis_methods as sam
import functions.utils as ut
import folium       # Seeing if we can make zoomable html maps
from matplotlib.colors import ListedColormap    # Also for map plotting
import random as rn
from ml_scripts.registry import register_builder,register_sample_method
from collections import defaultdict
import random

"""
This module contains much of the ML functionality when using our dataset. The method "load_data" is the main method of this script.

This module is large and therefore divided into sections:
- load_data
- Most important builders
- Most importamt sample methods
- Misc Methods
- Classes for creating the sample methods and builders needed for specific tasks such as some of the bias tests 
- More niche sample Methods and builders
- Methods with varied or no importance
"""
def load_data(cube, method, builder, batch_size, ps=None, num_workers=6, pad_in_T_dim = False,flips=False):
    """
    Loads data from specified zarr file, and creates train, val and test dataloader objects. Samples in the different
    classes are specified by the method argument. 
    """
    
    poly_idxs = ps if ps is not None else range(cube.p_dim)
    gdf = cube.gdf
    seed = 16     # For reproducibility
    if "cluster" in gdf.columns:        # If cluster column exists
        # Split clusters into train, val and test
        # gdf_with_clusters = gpd.read_file(GPKGS/"polygons_with_clusters_dist_1500.gpkg")     # Old solution  
        gdf_with_clusters = cube.gdf   
        gdf_with_clusters = gdf_with_clusters.loc[gdf_with_clusters.index.isin(poly_idxs)]         # Only take the specified polygons (note that row numbers are preserved)
        unique_clusters = list(set(gdf_with_clusters["cluster"]))       # OBS! This can be undeterministic for larger sets. Modify for future use
        
        # Split clusters into train, val and test
        train_val_clusters, test_clusters = train_test_split(unique_clusters, test_size=0.10, random_state=seed, shuffle=True)
        val_fraction_of_trainval = 0.10 / 0.90
        train_clusters, val_clusters = train_test_split(train_val_clusters, test_size=val_fraction_of_trainval, random_state=seed, shuffle=True)

        # Extract poly idxs for the clusters in each set
        train = gdf_with_clusters.index[gdf_with_clusters["cluster"].isin(train_clusters)].tolist()
        val = gdf_with_clusters.index[gdf_with_clusters["cluster"].isin(val_clusters)].tolist()
        test = gdf_with_clusters.index[gdf_with_clusters["cluster"].isin(test_clusters)].tolist()

        #print("VAL SET HASH:", hash(tuple(sorted(val))))
        #print("TEST SET HASH:", hash(tuple(sorted(test))))
    else:
        # Split polygons into train, val and test
        train_val, test = train_test_split(poly_idxs, test_size=0.10, random_state=seed, shuffle=True)
        val_fraction_of_trainval = 0.10 / 0.90
        train, val = train_test_split(train_val, test_size=val_fraction_of_trainval, random_state=seed, shuffle=True)

    # Plot split map (train/test/val)
    #savedir = ut.make_savedir("linnea_testar_grejer_runs", additional_info =f'Split map')
    #filetypes = ["png","html"]
    #for filetype in filetypes:
    #   plot_split_map(gdf_with_clusters,train,val,test,savedir,filetype = filetype)

    # print_split_stats(train,val,test)   # Prints fractions of polygons in each set
    # overlap_test(train,val,test)        # Prints potential overlap in the sets (should not be any)

    # Construct samples from poly idxs
    train_samples = idx_to_samples(train,cube,method,builder)
    val_samples = idx_to_samples(val,cube,method,builder)
    test_samples = idx_to_samples(test,cube,method,builder)

    print_split_stats(train_samples,val_samples,test_samples,info="samples")   # Prints fractions of samples in each set

    # train_samples,val_samples = overfitting_test(train_samples, val_samples)   # See if the ML can overfit on a small dataset

    # Create ZarrDatasets for each set
    if flips:
        train_ds = zarr_dataset_class.ZarrDatasetFlip(cube, train_samples)
    else:
        train_ds = zarr_dataset_class.ZarrDataset(cube, train_samples)

    val_ds   = zarr_dataset_class.ZarrDataset(cube, val_samples)
    test_ds  = zarr_dataset_class.ZarrDataset(cube, test_samples)

    # dataset_tests([train_ds, val_ds, test_ds], ["Train dataset", "Validation dataset", "Train dataset"])

    if pad_in_T_dim:    # Used for LSTM type networks
        return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False,collate_fn = collate_fn),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False,collate_fn = collate_fn),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False,collate_fn = collate_fn),
        )
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,persistent_workers=False,prefetch_factor=2,pin_memory=False),
        )

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Most important sample methods
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_sample_method
def first_and_last_year(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    OBS! Note that first_and_last_year_balanced is the sample method that is used in the project. 
    """

    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_sample_method
def first_and_last_year_balanced(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    Makes sure that the negative and positive sample have the same number of images. If they are
    unbalanced, images are randomly selected so that they are the same length. 
    """
    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    rng = np.random.default_rng(seed=poly_idx)
    if len(last_ids) <= len(first_ids):
        first_ids=rng.choice(first_ids,len(last_ids),replace=False)
    else:
        last_ids=rng.choice(last_ids,len(first_ids),replace=False)

    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }
    return [sample_neg,sample_pos]

@register_sample_method
def first_and_last_year_monthly(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    Used to make monthly composites in the project. 
    """
    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
    dates=cube.timestamps[first_ids]
    first_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
    first_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
    first_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

    dates=cube.timestamps[last_ids]
    last_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
    last_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
    last_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

    if (len(first_jun_ts)==0) or (len(first_jul_ts)==0) or (len(first_aug_ts)==0):
        return []
    
    if (len(last_jun_ts)==0) or (len(last_jul_ts)==0) or (len(last_aug_ts)==0):
        return []
    
    sample_neg = {
    "p": poly_idx,
    "times": [first_jun_ts,first_jul_ts,first_aug_ts],
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": [last_jun_ts,last_jul_ts,last_aug_ts],
    "label": 1,
    "builder": builder
    }
    return [sample_neg,sample_pos]

@register_sample_method
def fifty_fifty_timeseries_balanced(poly_idx,cube,builder):
    """
    Takes the first half of the time series as a negative sample, and the second half as a positive.
    For odd durations, middle year has a 50/50 probability of belonging to etiher class. Balances
    number of images in positive and negative sample. 
    Used as "Multiple year" input in the project
    """
    year_after = cube.get_last_year(poly_idx)+1
    first_year = cube.get_first_year(poly_idx)
    nbr_years_tot = year_after-first_year+1

    half = nbr_years_tot // 2

    if nbr_years_tot % 2 == 0:
        split = first_year + half
    else:
        # 50/50: middle year goes left or right
        split = first_year + half + (np.random.rand() < 0.5)

    neg_years = range(first_year, split)
    pos_years = range(split, year_after + 1)

    time_ids=cube.cloud_free_time(poly_idx)
    dates=cube.timestamps[time_ids]

    neg_ts = cube.id_from_date([d for d in dates if d.year in neg_years])
    pos_ts  = cube.id_from_date([d for d in dates if d.year in pos_years])

    if (len(neg_ts)==0) or (len(pos_ts)==0):
        return []
    
    rng = np.random.default_rng(seed=poly_idx)

    if len(pos_ts) <= len(neg_ts):
        neg_ts=rng.choice(neg_ts,len(pos_ts),replace=False)
    else:
        pos_ts=rng.choice(pos_ts,len(neg_ts),replace=False)
    
    sample_neg = {
    "p": poly_idx,
    "times": neg_ts,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": pos_ts,
    "label": 1,
    "builder": builder
    }
    return [sample_neg,sample_pos]
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Most important builders
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_builder 
def build_masked_composite(cube,sample):
    """Sample instructions for creating poly masked median composites (/10000 normalized)
    """
    comp = np.median(cube.get_normalized_masked_images(sample["p"],sample["times"],fill_val=0),axis=0)
    return comp

@register_builder
def build_global_normalized_masked_composite(cube,sample):
    """Sample instructions for creating poly masked median composites (global normalized)
    """
    comp = np.median(cube.get_global_normalized_masked_images(sample["p"],sample["times"],fill_val=0),axis=0).astype(np.float32)
    return comp

@register_builder
def build_masked_composite_pasture_norm(cube,sample):
    """Sample instructions for creating poly masked median composites (per pasture norm)
    """
    comp = np.median(cube.get_pasture_normalized_masked_images(sample["p"],sample["times"],fill_val=0),axis=0).astype(np.float32)
    return comp

@register_builder
def build_masked_composite_monthly(cube,sample):
    """Sample instructions for creating poly masked median composites of each month and stacking them.
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    p=sample["p"]
    jun_comp= np.median(cube.get_normalized_masked_images(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_normalized_masked_images(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_normalized_masked_images(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_masked_composite_monthly_global_normalization(cube,sample):
    """Sample instructions for creating poly masked median composites of each month and stacking them. Global normalization.
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    p=sample["p"]
    jun_comp= np.median(cube.get_global_normalized_masked_images(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_global_normalized_masked_images(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_global_normalized_masked_images(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_masked_composite_monthly_pasture_normalization(cube,sample):
    """
    Sample instructions for creating poly masked median composites of each month and stacking them
    Per pasture normalization.
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    p=sample["p"]
    jun_comp= np.median(cube.get_pasture_normalized_masked_images(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_pasture_normalized_masked_images(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_pasture_normalized_masked_images(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_raw_masked_uniform_norm(cube, sample):
    """Build raw masked image based on the /10000 normalization
    """
    return cube.get_normalized_masked_images(sample["p"], sample["times"],fill_val=0)

@register_builder
def build_raw_masked_global_norm(cube, sample):
    """Build raw masked image based on the global normalization
    """
    return cube.get_global_normalized_masked_images(sample["p"], sample["times"],fill_val=0).astype(np.float32)

@register_builder
def build_raw_masked_pasture(cube,sample):
    """
    Sample instructions for creating poly masked, pasture normalized, raw images.
    """
    p=sample["p"]
    ts=sample["times"]
    imgs = cube.get_pasture_normalized_masked_images(p,ts,fill_val=0.0).astype(np.float32)
    return imgs

@register_builder
def build_raw_image(cube, sample):
    """Sample instructions for simple extraction of raw images
    """
    return cube.get_normalized_images(sample["p"],sample["times"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Misc Methods
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def idx_to_samples(poly_idxs,cube,method,builder):
    """ 
    Makes samples from polygon ids, using a specified zarr cube and method.
    """
    samples = []
    for poly_idx in poly_idxs:                                       
        samples.extend(method(poly_idx,cube,builder))
    return samples

# TESTS AND VISUALIZATION --------------------------------------------------------------------------------
def overlap_test(train,val,test):
    """
    Checking overlap between the sets (there should be none)
    """
    print("Overlap test/train: ", [e for e in test if e in train])
    print("Overlap test/val: ", [e for e in test if e in val])
    print("Overlap val/train: ", [e for e in val if e in train])

def dataset_tests(datasets, labels):
    """
    Loops through all the datasets, and extracts samples. Throws an exception if something is wrong.
    """
    for i, dataset in enumerate(datasets):
        for idx in range(len(dataset)):
            try:
                img, label = dataset[idx]
            except:
                print(f"Index {idx} in {labels[i]} yielded an exception")  
    
    print("Dataset tests all done")   

def dataloader_tests(dataloaders,labels):             
    """
    Ensures we can loop through dataloader batches without problem.
    """
    error_counter = 0
    for k, dataloader in enumerate(dataloaders):
        try:
            for i, batch in enumerate(dataloader):
                X, y = batch
                #print(f"Got batch {i}") 
        except Exception as e:
            error_counter +=1 
            print(f"Unexpected error {error_counter}: {e}")
            print(f"Index {i} in {labels[k]} yielded an exception") 

    print("Dataloader tests all done")    
    
def print_split_stats(train,val,test,info="polygons"):
    """
    Print some statistics about the splits
    """
    tot = len(train)+len(val)+len(test)
    print()
    print("----------------------------------------------------------------------------------------------")
    print("Total number of " + info + f" {tot}")
    print(f"Length of train: {len(train)}, or about {np.round(len(train)/tot*100, decimals=1)}% of the " + info)
    print(f"Length of val: {len(val)}, or about {np.round(len(val)/tot*100, decimals=1)}% of the " + info)
    print(f"Length of test: {len(test)}, or about {np.round(len(test)/tot*100, decimals=1)}% of the " + info)
    print("----------------------------------------------------------------------------------------------")

def plot_split_map(gdf,train,val,test,savedir,filename="Map_of_train_val_test_split",filetype = "png"):
    """
    Plots a map where you can see which polygons are in train, val and test, respectively. 
    """
    # Load map of Sweden
    map_path=DATA/"map_countries.shp"
    sweden = gpd.read_file(map_path)
    sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']

    centroids = gdf.copy()
    centroids["geometry"] = gdf.geometry.centroid

    # Make a new column where it is specified if the polygon is in train/val/test     
    centroids.loc[centroids.index.isin(train), "split"] = "train"
    centroids.loc[centroids.index.isin(val), "split"] = "val"
    centroids.loc[centroids.index.isin(test), "split"] = "test"

    if filetype == "png":
        sweden = sweden.to_crs(epsg=4326)   # Switch to correct coord system
        centroids = centroids.to_crs(epsg=4326)

        fig,ax = plt.subplots(figsize=(6, 8))
        sweden.boundary.plot(ax=ax, color="black", linewidth=1)
        cmap = ListedColormap(["red", "blue", "green"])
        centroids.plot(
        ax=ax,
        column="split",
        categorical=True,
        cmap = cmap,
        markersize=5,
        legend=True,
        )

        # Crop to shape + no ticks or axis
        # minx, miny, maxx, maxy = sweden.total_bounds
        # ax.set_xlim(minx, maxx)
        # ax.set_ylim(miny, maxy)
        # ax.margins(0)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.axis("off")

        # Fix ticks and lon lat coordinates
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.set_xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 1, 2))
        ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 1, 2))

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.grid(True)

        plt.savefig(savedir/filename)
        plt.close() 

    if filetype == "html":    
        centroids = centroids.to_crs(epsg=4326) # Ensure coord system?
        
        m = folium.Map(
            location=[62, 15],   # center of Sweden
            zoom_start=5,
            tiles="CartoDB positron" 
        )
        
        for idx, row in centroids.iterrows():
            color_map = {
            "test": "red",
            "train": "blue",
            "val": "green"
            }

            popup = folium.Popup(
                f"Poly index: {idx}<br>Group: {row['split']}",
                max_width=250
            )

            # Plot centroid
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color_map[row["split"]],
                fill=True,
                fill_opacity=0.8,
                popup=popup
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
        m.save(savedir/(filename+".html"))
 
def overfitting_test(train_samples,val_samples,n_samples=10):
    """
    Take smaller datasets to ensure the ML model can overfit on the training.
    """
    print("Overfitting test coming up ------------------------------------------------------------------------------")
    train_samples = train_samples[:10]
    val_samples = val_samples[:10]
    print("Shape of train samples: ", np.shape(train_samples))     
    return train_samples,val_samples
# ---------------------------------------------------------------------------------------------------------------------------------  

def collate_fn(batch):
    """
    Zero pads all sequences in the batch to the longest length. 
    batch: list of (X, y)
        X: Tensor of shape (T_i, C, H, W)
        y: scalar label
    """
    # Unzip batch
    sequences, labels = zip(*batch)
    sequences = [
        torch.from_numpy(seq).float() if isinstance(seq, np.ndarray) else seq.float()
        for seq in sequences
    ]

    # Sequence lengths (before padding)
    seq_lens = torch.tensor([seq.shape[0] for seq in sequences])
    # Pad sequences along time dimension (T)
    # Output shape: (B, T_max, C, H, W)
    
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,      # (B, T, ...)
        padding_value=0.0      # zero padding
    )
    labels = torch.tensor(labels)
    return padded_sequences, seq_lens, labels

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Classes for creating the sample methods and builders needed for specific tasks such as some of the bias tests etc. As you have seen in this
module, there are too many sample methods and builder. Moving forward it is recommended to create a single umbrella class for them.
To do this, please be inspired by these classes bellow
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class multi_year_bias_test:
    def  __init__(self,cluster_gdf,cube,og_builder="stack",seed=1,number_imgs=3,balanced=False,exclude_years=[]):
        self.balanced=balanced
        self.exclude_years=exclude_years
        self.cluster_gdf=cluster_gdf
        self.cube=cube
        self.number_imgs=number_imgs
        self.og_builder=og_builder
        self.all_years=np.arange(2018,2026)
        self.rng_np = np.random.default_rng(seed)
        self.rng_py = random.Random(seed)
        self.samples=self.sample_balanced_dataset()
        self.samples_by_poly=self.group_by_polygon(self.samples)
        self.year_to_label = {year: i for i, year in enumerate(self.all_years)}
        self.name = "multi_year_bias_test"

    def __call__(self, poly_idx, cube, builder):
        final_samples=[]
        samples=self.samples_by_poly.get(poly_idx, [])

        for _, year, _ in samples:
            
            if year in self.exclude_years:
                continue
            ts = self.cube.ids_from_year(poly_idx, year)
            label=self.year_to_label[year]
            if len(ts) == 0:
                continue
            if (len(ts)<self.number_imgs) and self.og_builder=="random_stack":
                continue
            
            if self.og_builder=="composite" or self.og_builder=="raw" or self.og_builder=="random_stack":
                sample = {
                    "p": poly_idx,
                    "times": ts,
                    "label": label,
                    "builder": builder
                }
            
            elif self.og_builder=="stack":

                dates=cube.timestamps[ts]
                jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
                jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
                aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

                if (len(jun_ts)==0) or (len(jul_ts)==0) or (len(aug_ts)==0):
                    continue
                
                sample = {
                "p": poly_idx,
                "times": [jun_ts,jul_ts,aug_ts],
                "label": label,
                "builder": builder
                }
        
            final_samples.append(sample)
        return final_samples

    def group_by_polygon(self,balanced_samples):
        by_poly = defaultdict(list)

        for p, year, cluster in balanced_samples:
            by_poly[p].append((p, year, cluster))

        return by_poly
    
    def build_year_cluster_dict(self):
        self.cube.build_cluster_index(self.cluster_gdf)
        
        by_year_cluster = defaultdict(lambda: defaultdict(list))
    
        for cluster, polys in self.cube.cluster_to_polys.items():
        
            min_ts=10000
            p=polys[0]
            
            #Extract minimum ts
            for year in self.all_years:
                
                ts = self.cube.ids_from_year(p, year)
                if len(ts) == 0:
                    continue
                
                if len(ts)<min_ts:
                    min_ts=len(ts)

            for year in self.all_years:
                ts = self.cube.ids_from_year(p, year)
                if len(ts) == 0:
                    continue
                
                if self.balanced:
                    if len(ts)>min_ts:
                        rng=np.random.default_rng(seed=p)
                        ts=rng.choice(ts,min_ts,replace=False)

                if self.og_builder == "stack":
                    dates = self.cube.timestamps[ts]
                    jun_ts = self.cube.id_from_date(
                        [d for d in dates if d.month == 6]
                    )
                    jul_ts = self.cube.id_from_date(
                        [d for d in dates if d.month == 7]
                    )
                    aug_ts = self.cube.id_from_date(
                        [d for d in dates if d.month == 8]
                    )
                    if (len(jun_ts) == 0 or
                        len(jul_ts) == 0 or
                        len(aug_ts) == 0):
                        continue
                elif self.og_builder == "composite" or self.og_builder=="raw":
                    pass
                sample = (p, year, cluster)
                by_year_cluster[year][cluster].append(sample)

        return by_year_cluster
    
    def sample_balanced_dataset(self, K=None):
        by_year_cluster = self.build_year_cluster_dict()

        # How many clusters can we sample per year?
        cluster_counts = {
            year: len(cluster_dict)
            for year, cluster_dict in by_year_cluster.items()
        }
        min_clusters = min(cluster_counts.values())

        # If K is specified, cap it
        if K is not None:
            min_clusters = min(min_clusters, K)

        balanced_samples = []
        for year, cluster_dict in by_year_cluster.items():

            # Choose clusters for this year
            chosen_clusters = self.rng_py.sample(
                list(cluster_dict.keys()),
                min_clusters
            )

            for cluster in chosen_clusters:
                samples = cluster_dict[cluster]
                # pick one sample from that cluster
                s = self.rng_py.choice(samples)
                balanced_samples.append(s)

        return balanced_samples

@register_builder
class multi_year_bias_builder:

    def __init__(self,builder_type="stack", normalization="global",masking="anti",random_img=False,number_imgs_stack=3,seed=38):  
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.number_imgs_stack=number_imgs_stack
        self.random_img=random_img
        self.builder_type=builder_type
        self.masking=masking
        self.normalization=normalization
        self.normalized_img_method=self.get_normalized_img_method()
        self.name = "multi_year_bias_builder"

    def __call__(self, cube, sample):
        method = getattr(cube, self.normalized_img_method)
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000))

        if self.builder_type=="mask":
            p=sample["p"]
            ts=sample["times"]
            dates=cube.timestamps[ts]
            year=dates[0].year
            mask=cube.get_cluster_inverse_mask(p,year).astype(np.float32)
            mask = mask[None, :, :]    
            return mask
        
        if self.builder_type=="stack":
            jun_ts=sample["times"][0]
            jul_ts=sample["times"][1]
            aug_ts=sample["times"][2]
            p=sample["p"]

            if self.random_img:
                jun_ts=rng.choice(jun_ts,1,replace=False)
                jul_ts=rng.choice(jul_ts,1,replace=False)
                aug_ts=rng.choice(aug_ts,1,replace=False)

            if self.normalization=="whitened":
                ts = np.concatenate([jun_ts, jul_ts, aug_ts])
                p=sample["p"]
                mean,std=cube.single_pasture_mean_std(p,ts)
                jun_comp= np.median(method(p,jun_ts,mean,std, fill_val=0),axis=0).astype(np.float32)
                jul_comp= np.median(method(p,jul_ts,mean,std, fill_val=0),axis=0).astype(np.float32)
                aug_comp= np.median(method(p,aug_ts,mean,std, fill_val=0),axis=0).astype(np.float32)

            else:
                jun_comp= np.median(method(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
                jul_comp= np.median(method(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
                aug_comp= np.median(method(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
            
            comp = np.concatenate(
                [jun_comp, jul_comp, aug_comp],
                axis=0
            )
            return comp
        
        elif self.builder_type=="composite":
            ts=sample["times"]
            if self.random_img:
                ts=rng.choice(ts,1,replace=False)

            if self.normalization=="whitened":
                p=sample["p"]
                mean,std=cube.single_pasture_mean_std(p,ts)
                comp = np.median(method(sample["p"],ts,mean,std,fill_val=0),axis=0).astype(np.float32)
            else:
                comp = np.median(method(sample["p"],ts,fill_val=0),axis=0).astype(np.float32)
            return comp
        
        elif self.builder_type=="raw":
            p=sample["p"]
            ts=sample["times"]
            imgs = method(p,ts,fill_val=0.0).astype(np.float32)

            return imgs
        
        elif self.builder_type=="random_stack":
            
            p = sample["p"]
            ts = np.array(sample["times"])
            chosen_ts = rng.choice(ts, self.number_imgs_stack, replace=False)
            comps = []
        
            if self.normalization == "whitened":
                mean, std = cube.single_pasture_mean_std(p, chosen_ts)

                for t in chosen_ts:
                    img = method(p, [t], mean, std, fill_val=0)
                    img = img.astype(np.float32)[0]  # remove temporal dim
                    comps.append(img)
        
            else:
                for t in chosen_ts:
                    img = method(p, [t], fill_val=0)
                    img = img.astype(np.float32)[0]
                    comps.append(img)
        
            # Stack along channel axis
            comp = np.concatenate(comps, axis=0)

            return comp

    def get_normalized_img_method(self):
        if self.normalization=="global":
            if self.masking != "anti":
                return "get_global_normalized_masked_images"
            return "get_global_normalized_anti_masked_images"
        if self.normalization=="pasture":
            if self.masking != "anti":
                return "get_pasture_normalized_masked_images"
            return "get_pasture_normalized_anti_masked_images"
        if self.normalization=="whitened":
            if self.masking != "anti":
                return "get_custom_normalized_masked_images"
            return "get_custom_normalized_anti_masked_images" #NOT IMPLEMENTED

class pasture_bias_sample_method:
    """Class for handling sampler methods for the pasture normalized bias test

    parameters:
    year: number of years after restoration start to name as last year
    """
    def __init__(self,year=1,og_builder="stack"):
        self.year=year
        self.og_builder=og_builder

    def __call__(self,poly_idx, cube, _):
        first_year=cube.get_first_year(poly_idx)
        last_year=cube.get_last_year(poly_idx)+1
        num_years=last_year-first_year+1

        if num_years!=6:
            return []
        
        last_year=first_year+self.year
        first_ids=cube.ids_from_year(poly_idx,first_year)
        last_ids=cube.ids_from_year(poly_idx,last_year)
        if (len(first_ids)==0) or (len(last_ids)==0):
            return []
        
        range_ts=cube.cloud_free_range_ts(poly_idx,first_year,last_year)
        mean, std=cube.single_pasture_mean_std(poly_idx,range_ts)
        builder=pasture_bias_builder(mean,std,og_builder=self.og_builder)

        if self.og_builder=="composite":
    
            sample_neg = {
            "p": poly_idx,
            "times": first_ids,
            "label": 0,
            "builder": builder
            }

            sample_pos = {
            "p": poly_idx,
            "times": last_ids,
            "label": 1,
            "builder": builder
            }

            if self.year==0:
                return [sample_neg]

            return [sample_neg,sample_pos]
        
        elif self.og_builder=="stack":

            dates=cube.timestamps[first_ids]
            first_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
            first_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
            first_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

            dates=cube.timestamps[last_ids]
            last_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
            last_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
            last_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

            if (len(first_jun_ts)==0) or (len(first_jul_ts)==0) or (len(first_aug_ts)==0):
                return []

            if (len(last_jun_ts)==0) or (len(last_jul_ts)==0) or (len(last_aug_ts)==0):
                return []

            sample_neg = {
            "p": poly_idx,
            "times": [first_jun_ts,first_jul_ts,first_aug_ts],
            "label": 0,
            "builder": builder
            }

            sample_pos = {
            "p": poly_idx,
            "times": [last_jun_ts,last_jul_ts,last_aug_ts],
            "label": 1,
            "builder": builder
            }

            if self.year==0:
                return [sample_neg]
            
            return [sample_neg,sample_pos]

class pasture_bias_builder:

    def __init__(self,mean,std,og_builder="stack"):
        self.mean=mean
        self.std=std
        self.og_builder=og_builder

    def __call__(self,cube, sample):
        if self.og_builder=="stack":
            jun_ts=sample["times"][0]
            jul_ts=sample["times"][1]
            aug_ts=sample["times"][2]
            p=sample["p"]
            jun_comp= np.median(cube.get_custom_normalized_masked_images(p,jun_ts,self.mean,self.std,fill_val=0),axis=0).astype(np.float32)
            jul_comp= np.median(cube.get_custom_normalized_masked_images(p,jul_ts,self.mean,self.std,fill_val=0),axis=0).astype(np.float32)
            aug_comp= np.median(cube.get_custom_normalized_masked_images(p,aug_ts,self.mean,self.std,fill_val=0),axis=0).astype(np.float32)

            comp = np.concatenate(
                [jun_comp, jul_comp, aug_comp],
                axis=0
            )
            return comp
        
        elif self.og_builder=="composite":
            comp = np.median(cube.get_custom_normalized_masked_images(sample["p"],sample["times"],self.mean,self.std,fill_val=0),axis=0).astype(np.float32)
            return comp

class year_bias_test:
    """Class for handling sampler methods with year bias tests
    """
    def __init__(self,year,balanced=False):
        self.bias_year=year
        self.balanced=balanced

    def __call__(self, poly_idx, cube, builder):
        """
        Takes the given year as positive and a random other year as negative
        """
        pos_ids=cube.ids_from_year(poly_idx,self.bias_year)
        
        if len(pos_ids)==0:
            return []

        first_year=cube.get_first_year(poly_idx)
        last_year=cube.get_last_year(poly_idx)+1

        #Randomly select a year from remaining years
        years=np.arange(first_year,last_year+1)
        neg_years=years[years != self.bias_year]
        #Randomly choose one that actually has images
        neg_year= np.random.choice(neg_years)
        neg_ids = cube.ids_from_year(poly_idx,neg_year)
        while (len(neg_ids)==0) and (len(neg_years)>0):
            #Get rid of empty negative year
            neg_years=neg_years[neg_years!=neg_year]
            
            if len(neg_years) == 0:
                break

            #Try again
            neg_year= np.random.choice(neg_years)
            neg_ids = cube.ids_from_year(poly_idx,neg_year)
        
        if len(neg_ids)==0:
            return []
        
        if self.balanced:
            if len(pos_ids) <= len(neg_ids):
                neg_ids=np.random.choice(neg_ids,len(pos_ids),replace=False)
            else:
                pos_ids=np.random.choice(pos_ids,len(neg_ids),replace=False)

        sample_neg = {
        "p": poly_idx,
        "times": neg_ids,
        "label": 0,
        "builder": builder
        }
            
        sample_pos = {
        "p": poly_idx,
        "times": pos_ids,
        "label": 1,
        "builder": builder
        }
    
        return [sample_neg, sample_pos]

class single_year_sample_method:
    
    def __init__(self, rest_len=2,year=0,og_builder="stack"):
        self.rest_len=rest_len
        self.year=year
        self.og_builder=og_builder

    def __call__(self,poly_idx, cube, builder):

        first_year=cube.get_first_year(poly_idx)
        last_year=cube.get_last_year(poly_idx)+1
        rest_len=last_year-first_year+1
        if rest_len!=self.rest_len:
            return []
        
        year=first_year+self.year
        ts=cube.ids_from_year(poly_idx,year)

        if len(ts) == 0:
            return []

        if self.og_builder=="composite":
            sample = {
            "p": poly_idx,
            "times": ts,
            "label": 0,
            "builder": builder
            }

            return [sample]
    
        elif self.og_builder=="stack":

            dates=cube.timestamps[ts]
            jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
            jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
            aug_ts = cube.id_from_date([d for d in dates if d.month == 8])
            if (len(jun_ts)==0) or (len(jul_ts)==0) or (len(aug_ts)==0):
                return []
            sample = {
            "p": poly_idx,
            "times": [jun_ts,jul_ts,aug_ts],
            "label": 0,
            "builder": builder
            }
            return [sample]
        
class res_duration_sample_method:
    
    def __init__(self, rest_len=2,og_builder="stack"):
        self.rest_len=rest_len
        self.og_builder=og_builder

    def __call__(self,poly_idx, cube, builder):

        first_year=cube.get_first_year(poly_idx)
        last_year=cube.get_last_year(poly_idx)+1
        rest_len=last_year-first_year+1
        if rest_len!=self.rest_len:
            return []
        
        first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
        if self.og_builder=="composite":

            if (len(first_ids)==0) or (len(last_ids)==0):
                return []

            rng = np.random.default_rng(seed=poly_idx)

            if len(last_ids) <= len(first_ids):
                first_ids=rng.choice(first_ids,len(last_ids),replace=False)
            else:
                last_ids=rng.choice(last_ids,len(first_ids),replace=False)
        
            sample_neg = {
            "p": poly_idx,
            "times": first_ids,
            "label": 0,
            "builder": builder
            }
                
            sample_pos = {
            "p": poly_idx,
            "times": last_ids,
            "label": 1,
            "builder": builder
            }
    
        elif self.og_builder=="stack":

            dates=cube.timestamps[first_ids]
            first_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
            first_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
            first_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

            dates=cube.timestamps[last_ids]
            last_jun_ts = cube.id_from_date([d for d in dates if d.month == 6])
            last_jul_ts = cube.id_from_date([d for d in dates if d.month == 7])
            last_aug_ts = cube.id_from_date([d for d in dates if d.month == 8])

            if (len(first_jun_ts)==0) or (len(first_jul_ts)==0) or (len(first_aug_ts)==0):
                return []

            if (len(last_jun_ts)==0) or (len(last_jul_ts)==0) or (len(last_aug_ts)==0):
                return []

            sample_neg = {
            "p": poly_idx,
            "times": [first_jun_ts,first_jul_ts,first_aug_ts],
            "label": 0,
            "builder": builder
            }

            sample_pos = {
            "p": poly_idx,
            "times": [last_jun_ts,last_jul_ts,last_aug_ts],
            "label": 1,
            "builder": builder
            }

            return [sample_neg,sample_pos]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
More niche sample Methods and builders
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_sample_method
def first_and_last_year_req_imgs(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    Makes sure that there are at least three images in each sample. Used in three random raw image stack tests.
    """
    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)

    if (len(first_ids)<3) or (len(last_ids)<3):
        return []
    
    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }
    return [sample_neg,sample_pos]

@register_sample_method
def first_and_2ndlast_year(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the 2nd last year (i.e. approved year) and assigning them to samples
    """
    time_ids=cube.cloud_free_time(poly_idx)
    dates=cube.timestamps[time_ids]
    last_year=cube.get_last_year(poly_idx)
    first_year=cube.get_first_year(poly_idx)
    first_ids = cube.id_from_date([d for d in dates if d.year == first_year])
    last_ids  = cube.id_from_date([d for d in dates if d.year == last_year])
    
    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    rng = np.random.default_rng(seed=poly_idx)

    if len(last_ids) <= len(first_ids):
        first_ids=rng.choice(first_ids,len(last_ids),replace=False)
    else:
        last_ids=rng.choice(last_ids,len(first_ids),replace=False)

    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }
    return [sample_neg,sample_pos]

@register_builder
def build_masked_composite_monthly_global_normalization_random(cube,sample):
    """
    Sample instructions for creating poly masked median composites of each month and stacking them
    Global normalization. Three random images. 
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    p=sample["p"]
    rng = np.random.default_rng()
    jun_ts=rng.choice(jun_ts,1,replace=False)
    jul_ts=rng.choice(jul_ts,1,replace=False)
    aug_ts=rng.choice(aug_ts,1,replace=False)
    jun_comp= np.median(cube.get_global_normalized_masked_images(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_global_normalized_masked_images(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_global_normalized_masked_images(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_masked_monthly_pasture_normalization_random(cube,sample):
    """
    Sample instructions for creating poly masked median composites of each month and stacking them
    Per pasture normalization. Three random images from June, July, August.
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    rng = np.random.default_rng()
    jun_ts=rng.choice(jun_ts,1,replace=False)
    jul_ts=rng.choice(jul_ts,1,replace=False)
    aug_ts=rng.choice(aug_ts,1,replace=False)
    p=sample["p"]
    jun_comp= np.median(cube.get_pasture_normalized_masked_images(p,jun_ts,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_pasture_normalized_masked_images(p,jul_ts,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_pasture_normalized_masked_images(p,aug_ts,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_veg_inds(cube,sample):
    """
    Builds a sample with NDVI, NDMI, EVI and MSAVI as image channels (then makes masked composites).
    """
    raw_imgs = cube.get_global_normalized_masked_images(sample["p"],sample["times"])       # Changed to global normalized 14/5

    # Calculate veg inds for each of the images in raw imgs
    ndvi_series = sam.calculate_ndvi_series(raw_imgs)
    ndmi_series = sam.calculate_ndmi_series(raw_imgs)
    evi_series = sam.calculate_evi_series(raw_imgs)
    msavi_series = sam.calculate_msavi_series(raw_imgs)
    assert ndvi_series.shape == ndmi_series.shape == evi_series.shape == msavi_series.shape          # Assert same shape
    veg_inds_series = np.stack([ndvi_series, ndmi_series, evi_series, msavi_series], axis=1)    # Stack --> (T,4,H,W)

    # Make a composite
    comp = np.nanmedian(veg_inds_series,axis=0)
    comp = np.nan_to_num(comp, nan=0.0)     # Exchange NaNs to zeros
    return comp

@register_builder
def build_masked_composite_monthly_whitened_normalization_random(cube,sample):
    """
    Sample instructions for creating poly masked median composites of each month and stacking them.
    Normalized by whitening (i.e. z-score per image). 
    """
    jun_ts=sample["times"][0]
    jul_ts=sample["times"][1]
    aug_ts=sample["times"][2]
    rng = np.random.default_rng()
    jun_ts=rng.choice(jun_ts,1,replace=False)
    jul_ts=rng.choice(jul_ts,1,replace=False)
    aug_ts=rng.choice(aug_ts,1,replace=False)
    ts = np.concatenate([jun_ts, jul_ts, aug_ts])
    p=sample["p"]
    mean,std=cube.single_pasture_mean_std(p,ts)
    jun_comp= np.median(cube.get_custom_normalized_masked_images(p,jun_ts,mean,std,fill_val=0),axis=0).astype(np.float32)
    jul_comp= np.median(cube.get_custom_normalized_masked_images(p,jul_ts,mean,std,fill_val=0),axis=0).astype(np.float32)
    aug_comp= np.median(cube.get_custom_normalized_masked_images(p,aug_ts,mean,std,fill_val=0),axis=0).astype(np.float32)
    
    comp = np.concatenate(
        [jun_comp, jul_comp, aug_comp],
        axis=0
    )
    return comp

@register_builder
def build_bands_and_veg_inds_comp(cube,sample):
    """
    Builds samples that have NDVI, NDMI and MSAVI as additional channels. 
    """
    veg_ind_comp = build_veg_inds(cube,sample)
    bands_img_comp = build_global_normalized_masked_composite(cube,sample)
    comp_stack = np.concatenate([bands_img_comp,veg_ind_comp], axis=0).astype(np.float32)
    return comp_stack

@register_builder
def build_random_masked_img(cube,sample):
    """Extracts a random t from the times sent and builds raw masked image
    """
    t = np.random.choice(sample["times"])
    return cube.get_normalized_masked_images(sample["p"], t,fill_val=0)

@register_builder
def build_random_global_normalized_masked_img(cube,sample):
    """Extracts a random t from the times sent and builds raw masked image
    """
    t = np.random.choice(sample["times"])
    return cube.get_global_normalized_masked_images(sample["p"], t,fill_val=0).astype(np.float32)


@register_builder
def build_composite(cube,sample):   
    """Sample instructions for creating median composites
    """
    comp = np.median(cube.get_normalized_images(sample["p"],sample["times"]),axis=0)
       
    return comp

@register_builder
def build_noise_sample(cube,sample):
    """
    Generates a random noise image, that has a unique seed for each sample.
    """
    # Settings
    img_size = (12,100,100)                           # Decide image size for sample
    # img_size =(len(sample["times"]),12,100,100)         # Padding test
    padding_test = True

    # So that each sample has a unique seed
    if (sample["label"] == 0):
        rng = np.random.default_rng(sample["p"])
    elif (sample["label"] == 1):
        rng = np.random.default_rng(sample["p"] + cube.p_dim)
    else: 
        raise Exception("Unknown label: ", sample["label"])
    
    # Generate random image 
    im_noise = rng.random(img_size).astype(np.float32)

    # Testing if padding makes any difference
    if padding_test:
        im_series = cube.reflectance.oindex[sample["p"],sample["times"]]
        # Add the padding to noisy image
        padding_mask = (im_series[0,:,:,:] == -9999)        # Assuming border is the same (which is true for one year im_series)
        im_noise[padding_mask]= -5
 
    return im_noise 

@register_builder
def build_raw_masked_pasture_mchannel(cube,sample):
    """
     Sample instructions for creating poly masked, pasture normalized, raw images with a separate mask channel.
    """
    p=sample["p"]
    ts=sample["times"]
    imgs = cube.get_pasture_normalized_masked_images(p,ts,fill_val=0.0).astype(np.float32)
    
    mask_ids=cube.mask_idx_map[p][ts]
    masks=cube.precomputed_masks[p][mask_ids].astype(np.float32)
    masks = masks[:, None, :, :]              # (T, 1, H, W)
    
    imgs_with_mask = np.concatenate([imgs, masks], axis=1)  # (T, C+1, H, W)
    return imgs_with_mask

@register_builder
def build_mask_only(cube,sample):
    """
    Builds only the mask channel.
    """
    p=sample["p"]
    ts=sample["times"]

    mask_ids=cube.mask_idx_map[p][ts]
    masks=cube.precomputed_masks[p][mask_ids]
    mask = masks[0].astype(np.float32)              # (H, W), values {0.0, 1.0}
    mask = mask[None, :, :]    

    return mask

@register_builder
def build_masked_composite_mchannel(cube,sample):
    """
    Sample instructions for creating poly masked, pasture normalized, composite images with a separate mask channel.
    Fill value 0.
    """
    p=sample["p"]
    ts=sample["times"]
    comp = np.median(cube.get_pasture_normalized_masked_images(p,ts,fill_val=0.0),axis=0).astype(np.float32)
    mask_ids=cube.mask_idx_map[p][ts]
    masks=cube.precomputed_masks[p][mask_ids]

    mask = masks[0].astype(np.float32)              # (H, W), values {0.0, 1.0}
    mask = mask[None, :, :]                     # (1, H, W)
    comp_with_mask = np.concatenate([comp, mask], axis=0)

    return comp_with_mask

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Are these relevant? You tell me :)
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_sample_method
def first_and_last_image(poly_idx,cube,builder):
    """ 
    First and last time step for each polygon.
    """
    time_idxs=cube.valid_time(poly_idx)

    sample_neg = {
    "p": poly_idx,
    "times": time_idxs[0],
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": time_idxs[-1],
    "label": 1,
    "builder": builder
    }

    return [sample_neg, sample_pos]

@register_sample_method
def first_and_last_year_balanced_soft(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    Makes sure that the negative and positive sample have the same number of images. If they are
    unbalanced, images are randomly selected so that they are the same length. 
    Uses labels 0.05 and 0.95 instead of 0 and 1. 
    """
    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    if len(last_ids) <= len(first_ids):
        first_ids=np.random.choice(first_ids,len(last_ids),replace=False)
    else:
        last_ids=np.random.choice(last_ids,len(first_ids),replace=False)

    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0.05,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 0.95,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_sample_method
def first_and_last_year_individual_samples(poly_idx,cube,builder):
    """Extracting image ids of ALL first year and last year as individual samples. Makes it balanced by checking which has more data.
    """
    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)
    samples=[]

    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    rng = np.random.default_rng(seed=poly_idx)

    if len(first_ids) <= len(last_ids):
        last_ids=rng.choice(last_ids,len(first_ids),replace=False)
    else:
        first_ids=rng.choice(first_ids,len(last_ids),replace=False)

    for t in first_ids:
        sample_neg = {
        "p": poly_idx,
        "times": t,
        "label": 0,
        "builder": builder
            }
        samples.append(sample_neg)
    
    for t in last_ids:
        sample_pos = {
        "p": poly_idx,
        "times": t,
        "label": 1,
        "builder": builder
            }
        samples.append(sample_pos)
        
    return samples
    
@register_sample_method
def split_for_2022(poly_idx,cube,builder):
    """
    Dividing images into pre and post 2022, post being labled 1, and pre being labled 0.
    Used for checking if 2022 split can be detected.
    """
    ts=cube.cloud_free_time(poly_idx)
    idx_2022=cube.get_2022_split_idx()
    pre_ts=ts[ts<idx_2022]
    post_ts=ts[ts>idx_2022]
    samples=[]

    if len(pre_ts) <= len(post_ts):
        post_ts=np.random.choice(post_ts,len(pre_ts),replace=False)
    else:
        pre_ts=np.random.choice(pre_ts,len(post_ts),replace=False)


    for t in pre_ts:
        sample_neg = {
        "p": poly_idx,
        "times": t,
        "label": 0,
        "builder": builder
            }
        samples.append(sample_neg)
    
    for t in post_ts:
        sample_pos = {
        "p": poly_idx,
        "times": t,
        "label": 1,
        "builder": builder
            }
        samples.append(sample_pos)
        
    return samples

@register_sample_method
def first_and_last_year_post_2022(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year (after 2022) and the last year and assigning them to samples.
    Used for checking if 2022 split can be detected.
    """
    time_ids = cube.cloud_free_time(poly_idx)
    dates = cube.timestamps[time_ids]
    last_year = cube.get_last_year(poly_idx)+1 # Due to year after

    if last_year < 2024:        
        return []
    first_year = 2022
    
    first_ids = cube.id_from_date([d for d in dates if d.year == first_year])
    last_ids  = cube.id_from_date([d for d in dates if d.year == last_year])

    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_sample_method
def first_and_last_year_size_thresh(poly_idx,cube,builder):
    """
    Extracting the image ids of the first year and the last year and assigning them to samples.
    Thresholds samples on polygon area size. 
    """

    gdf=cube.get_gdf()
    if gdf.loc[poly_idx, "calc_area"] <= 1000:
        return []

    first_ids, last_ids=cube.first_and_year_after_ids(poly_idx,cloud_free=True)

    if (len(first_ids)==0) or (len(last_ids)==0):
        return []
    
    sample_neg = {
    "p": poly_idx,
    "times": first_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": last_ids,
    "label": 1,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_sample_method
def random_length_timeseries(poly_idx,cube,builder):
    """
    Takes a random number of years in the end of the time series as a positive sample,
    and random number of years in the beginning as a negative sample. Ensures no temporal overlap.
    between the two samples. 
    OBS! Not ensured to be balanced.
    """
   
    year_after = cube.get_last_year(poly_idx)+1
    first_year = cube.get_first_year(poly_idx)
    nbr_years_tot = year_after-first_year+1
    
    if nbr_years_tot ==2:
        neg_years = [first_year]
        pos_years = [year_after]
    else:    
        # Randomizes number of negative samples
        # rng_neg = np.random.default_rng(seed=poly_idx)  # With seed
        rng_neg = np.random.default_rng()    
        nbr_neg = rng_neg.integers(1,nbr_years_tot-1)   # Year after not included
        neg_years = range(first_year,first_year+nbr_neg)

        # Randomizes number of positive samples
        # rng_pos = np.random.default_rng(seed=poly_idx+cube.p_dim) # With seed
        rng_pos = np.random.default_rng() 
        nbr_pos = rng_pos.integers(1, nbr_years_tot-nbr_neg) # Guarantees no overlap
        pos_years = range(year_after-nbr_pos+1, year_after+1)

    time_ids=cube.cloud_free_time(poly_idx)
    dates=cube.timestamps[time_ids]

    neg_ids = cube.id_from_date([d for d in dates if d.year in neg_years])
    pos_ids  = cube.id_from_date([d for d in dates if d.year in pos_years])

    if (len(neg_ids)==0) or (len(pos_ids)==0):
        return []
     
    sample_neg = {
    "p": poly_idx,
    "times": neg_ids,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": pos_ids,
    "label": 1,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_sample_method
def random_length_timeseries_per_season(poly_idx,cube,builder):
    """
    Takes a random number of years in the end of the time series as a positive sample,
    and random number of years in the beginning as a negative sample. Ensures no temporal overlap.
    between the two samples. Saves June, July and August separately to make composite stack possible.
    OBS! Not ensured to be balanced.
    """
   
    year_after = cube.get_last_year(poly_idx)+1
    first_year = cube.get_first_year(poly_idx)
    nbr_years_tot = year_after-first_year+1
    
    if nbr_years_tot ==2:
        neg_years = [first_year]
        pos_years = [year_after]
    else:    
        # Randomizes number of negative samples
        # rng_neg = np.random.default_rng(seed=poly_idx)  # With seed
        rng_neg = np.random.default_rng()    
        nbr_neg = rng_neg.integers(1,nbr_years_tot-1)   # Year after not included
        neg_years = range(first_year,first_year+nbr_neg)

        # Randomizes number of positive samples
        # rng_pos = np.random.default_rng(seed=poly_idx+cube.p_dim) # With seed
        rng_pos = np.random.default_rng() 
        nbr_pos = rng_pos.integers(1, nbr_years_tot-nbr_neg) # Guarantees no overlap
        pos_years = range(year_after-nbr_pos+1, year_after+1)

    time_ids=cube.cloud_free_time(poly_idx)
    dates=cube.timestamps[time_ids]

    # Adding the negative times per season
    neg_ts = []
    for year in neg_years:
        jun_ts = cube.id_from_date([d for d in dates if (d.month == 6 and d.year == year)])
        jul_ts = cube.id_from_date([d for d in dates if (d.month == 7 and d.year == year)])
        aug_ts = cube.id_from_date([d for d in dates if (d.month == 8 and d.year == year)])

        if (len(jun_ts)==0) or (len(jul_ts)==0) or (len(aug_ts)==0):
            return []

        neg_ts.extend([jun_ts,jul_ts,aug_ts])

    # Adding the positive times per season
    pos_ts = []
    for year in pos_years:
        jun_ts = cube.id_from_date([d for d in dates if (d.month == 6 and d.year == year)])
        jul_ts = cube.id_from_date([d for d in dates if (d.month == 7 and d.year == year)])
        aug_ts = cube.id_from_date([d for d in dates if (d.month == 8 and d.year == year)])

        if (len(jun_ts)==0) or (len(jul_ts)==0) or (len(aug_ts)==0):
            return []

        pos_ts.extend([jun_ts,jul_ts,aug_ts])
    
    
    sample_neg = {
    "p": poly_idx,
    "times": neg_ts,
    "label": 0,
    "builder": builder
    }
        
    sample_pos = {
    "p": poly_idx,
    "times": pos_ts,
    "label": 1,
    "builder": builder
    }

    return [sample_neg,sample_pos]

@register_builder
def build_raw_masked_image_different_mean_std(cube, sample):
    """Build raw masked image that is z score normalized pre and post 2022 separately
    """
    return cube.get_mean_std_normalized_masked_images(sample["p"],sample["times"],fill_val=-5)

@register_builder
def build_masked_composite_mchannel_fillval(cube,sample):
    """
    Sample instructions for creating poly masked, pasture normalized, composite images with a separate mask channel.
    Fill value -2.
    """
    p=sample["p"]
    ts=sample["times"]
    comp = np.median(cube.get_pasture_normalized_masked_images(p,ts,fill_val=-2),axis=0).astype(np.float32)
    
    mask_ids=cube.mask_idx_map[p][ts]
    masks=cube.precomputed_masks[p][mask_ids]

    mask = masks[0].astype(np.float32)              # (H, W), values {0.0, 1.0}
    mask = mask[None, :, :]                     # (1, H, W)

    comp_with_mask = np.concatenate([comp, mask], axis=0)

    return comp_with_mask

@register_builder
def build_masked_composite_mchannel_old_norm(cube,sample):
    """
    Sample instructions for creating poly masked, uniformly normalized (/10000), composite images with a separate mask channel.
    Fill value 0.5.
    """
    p=sample["p"]
    ts=sample["times"]
    comp = np.median(cube.get_normalized_masked_images(p,ts,fill_val=0.5),axis=0).astype(np.float32)
    
    mask_ids=cube.mask_idx_map[p][ts]
    masks=cube.precomputed_masks[p][mask_ids]

    mask = masks[0].astype(np.float32)              # (H, W), values {0.0, 1.0}
    mask = mask[None, :, :]                     # (1, H, W)

    comp_with_mask = np.concatenate([comp, mask], axis=0)

    return comp_with_mask

@register_builder
def build_masked_composite_monthly_series_global_norm(cube,sample):
    """Sample instructions for creating poly masked median composites of each month and stacking them in T axis.
    """
    p=sample["p"]
    comps = []
    for month_ts in sample["times"]:
        comps.append(np.median(cube.get_global_normalized_masked_images(p,month_ts,fill_val=0),axis=0).astype(np.float32))
    
    comp_series = np.stack(comps,axis=0)

    return comp_series

@register_builder
def build_2022_mean_std_normalized(cube,sample):
    """
    Sample instructions for creating masked composites that are mean std normalized differently
    depending on if they are from before or after 2022.
    """
    comp = np.median(cube.get_mean_std_normalized_masked_images(sample["p"],sample["times"],fill_val=-5), axis=0)
    return comp

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
MAIN
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    # cube=zarr_class.Zarr(ZARR,GPKGS/'final02-20.gpkg',cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    # cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    # cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    cube=zarr_class.Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    # The following will plot split map (if uncommented in load_data())
    load_data(cube, first_and_last_year, build_masked_composite, 1)

    # multi_test=multi_year_bias_test(gdf_with_clusters,cube)
    # multi_test.get_dataset()