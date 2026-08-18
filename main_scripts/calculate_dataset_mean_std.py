import numpy as np
from paths import GPKGS, ZARR, ROOT
from classes import zarr_class
import geopandas as gpd
from sklearn.model_selection import train_test_split

# Numerically stable way to calculate mean and std incrementally 
class WelfordVariance:      
    def __init__(self,C):  
        self.mean = np.zeros(C, dtype=np.float64)  
        self.count = 0
        self.M2 = np.zeros(C, dtype=np.float64)    

    def add_variable(self, img):
        mask = ~np.isnan(img[0])        # Same mask across channels
        pixels = img[:, mask]   

        for k in range(np.shape(pixels)[1]):
            self.count += 1
            x = pixels[:,k]
            old_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.count
            self.M2 += (x - old_mean) * (x - self.mean)
        
    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / self.count
    
    def get_sample_variance(self):
        return self.M2 / (self.count - 1)
    
def print_mean_and_variance(cube,C,poly_idxs):
    """
    Prints and saves the mean, variance and std for pre 2022, post 2022 and the entire dataset
    """
    pre_2022_stats = WelfordVariance(C)
    post_2022_stats = WelfordVariance(C)
    all_images_stats = WelfordVariance(C)

    for i,poly_idx in enumerate(poly_idxs):
        print("Polygon ", poly_idx)

        ts=cube.cloud_free_time(poly_idx)
        idx_2022=cube.get_2022_split_idx()
        pre_ts=ts[ts<idx_2022]
        post_ts=ts[ts>idx_2022]
 
        # Pre 2022
        for t in pre_ts:
            img = cube.get_unpadded_img(poly_idx,t).astype(np.float64)
            pre_2022_stats.add_variable(img)
            all_images_stats.add_variable(img)

        # Post 2022
        for t in post_ts:
            img = cube.get_unpadded_img(poly_idx,t).astype(np.float64)
            post_2022_stats.add_variable(img)
            all_images_stats.add_variable(img)  
 
    # PRINTS
    print("PRE 2022")
    print("Mean: ", pre_2022_stats.get_mean())    
    print("Variance: ",  pre_2022_stats.get_variance())   
    print("Standard deviation: ", np.sqrt(pre_2022_stats.get_variance()))  
    print("-----------------------------------------------------------------")
    print()
    print("POST 2022")
    print("Mean: ", post_2022_stats.get_mean())    
    print("Variance: ",  post_2022_stats.get_variance())   
    print("Standard deviation: ", np.sqrt(post_2022_stats.get_variance())) 
    print("-----------------------------------------------------------------------")
    print()
    print("THE ENTIRE DATASET")
    print("Mean: ", all_images_stats.get_mean())    
    print("Variance: ",  all_images_stats.get_variance())   
    print("Standard deviation: ", np.sqrt(all_images_stats.get_variance()))  
    

def get_train_idxs(cube):
    """
    Does the same split as in load data to get the training idxs.
    """
    poly_idxs = range(cube.p_dim)
 
    gdf_with_clusters = gpd.read_file(GPKGS/"polygons_with_clusters_dist_1500.gpkg")       
    gdf_with_clusters = gdf_with_clusters.loc[gdf_with_clusters.index.isin(poly_idxs)]         # Only take the specified polygons (note that row numbers are preserved)
    unique_clusters = list(set(gdf_with_clusters["cluster"]))
    seed = 16     # For reproducibility
    
    # Split clusters into train, val and test
    train_val_clusters, test_clusters = train_test_split(unique_clusters, test_size=0.10, random_state=seed, shuffle=True)
    val_fraction_of_trainval = 0.10 / 0.90
    train_clusters, val_clusters = train_test_split(train_val_clusters, test_size=val_fraction_of_trainval, random_state=seed, shuffle=True)

    # Extract poly idxs for train cluster
    train = gdf_with_clusters.index[gdf_with_clusters["cluster"].isin(train_clusters)].tolist()

    return train

def save_training_masked_mean_std(cube,C,save_dir):
    """
    Saves global mean std for the training set to disk (within polygon pixels).  
    """
    poly_idxs = get_train_idxs(cube)
    training_set_stats = WelfordVariance(C)
    training_set_means_stds = np.zeros((C,2))   
    
    for i,poly_idx in enumerate(poly_idxs):    
        print("Processing polygon ", poly_idx)
        # ts = cube.cloud_free_time(poly_idx)
        ts = cube.cloud_free_range_ts(poly_idx)
        for t in ts:
            img = cube.get_masked_images(poly_idx,t).astype(np.float64)
            training_set_stats.add_variable(img)

    training_set_means_stds[:,0]= training_set_stats.get_mean()     
    training_set_means_stds[:,1]= np.sqrt(training_set_stats.get_variance())
    print("All done!")
    print("Mean: ", training_set_means_stds[:,0])
    print("Std: ", training_set_means_stds[:,1])
    np.save(save_dir/"training_set_means_stds_cloud_151510_05_25.npy", training_set_means_stds) 
      




if __name__=="__main__": 
    path_to_gpkg = GPKGS/'final02-20.gpkg'
    cube = zarr_class.Zarr(ZARR,path_to_gpkg,"/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_15_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    C = 12      # Number of channels
    save_dir= ROOT/"data/mean_std/"
    save_training_masked_mean_std(cube,C,save_dir)
   