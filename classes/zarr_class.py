import sys
import numpy as np
from datetime import datetime, date
import zarr
import dask.array as da
from paths import ZARR, GPKGS, NC,POLY_MASKS,CLOUD_MASKS,IMGS, DATA
import geopandas as gpd
import functions.utils as ut
import functions.plotting_functions as pf
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import pickle
import json
import classes.gdf_class as gc

class Zarr:
    """
    Wrapper class for handling useful functions for using our zarr file.
    """
    def __init__(self, path_to_zarrfile,path_to_gpkg,cloud_path=None):
        self.cloud_path=cloud_path

        #Letting the huge chunk of images be opened using dask to allow for 
        #numpy-style indexing without having to load it all at once
        self.root=zarr.open(path_to_zarrfile,mode="r")
        self.reflectance=self.root["reflectance"]
        self.da_reflectance=da.from_zarr(self.root["reflectance"])
        
        #Turn smaller side arrays into np arrays for ease of use
        self.im_id=np.array(self.root["image_index"])
        self.x0=np.array(self.root["image_x0"])
        self.y0=np.array(self.root["image_y0"])
        self.year_area=np.array(self.root["image_year"])
        
        # Convert numpy.datetime64 to datetime.date
        self.timestamps = self._timestamps_from_zarr()
        
        #Creates a dictionary for going from date to index
        self.date_to_index = {d: i for i, d in enumerate(self.timestamps)}

        # Dimensions of P,T
        self.p_dim=self.im_id.shape[0]
        self.t_dim=self.im_id.shape[1]
        
        #H and W
        self.H=self.reflectance.shape[-2]
        self.W=self.reflectance.shape[-1]

        #GDF where the polygons live
        self.gdf=gpd.read_file(path_to_gpkg)
        self.gdf["Years_Areas"] = self.gdf["Years_Areas_json"].apply(json.loads)

        #Sets up dictionaries for masks and load mask from data
        self.precomputed_masks={}
        self.mask_idx_map= {}
        #If there is no mask cache from the hard_coded filepath, create and set it as the attrbutes
        try:
            self.load_mask_cache()
        except Exception:
            self.create_mask_cache()

        #Post_22 mask for normalization (this is from a time we thought the satellite data was not harmonized)
        self.post_2022=self._post_2022_mask()

        # Cloud filter mask 
        self.load_cloud_mask(self.cloud_path)

        #pasture means and stds, If the path isn't working, sets to mean=0 and std=1
        try:
            self.pasture_means=np.load(DATA/"mean_std"/"pasture_means.npy")
            self.pasture_stds=np.load(DATA/"mean_std"/"pasture_stds.npy")
        except Exception:     
            self.pasture_means = np.zeros((self.p_dim, 12))
            self.pasture_stds  = np.ones((self.p_dim, 12))

        #Global means and stds, if path isn't working, sets to mean=0 and std=1
        try:
            glob_mean_stds=np.load(DATA/"mean_std"/"training_set_means_stds_with_range_ts_05_14.npy")
        
            self.glob_means=glob_mean_stds[:,0]
            self.glob_stds=glob_mean_stds[:,1]
        except Exception:    
            self.glob_means = np.zeros(12)
            self.glob_stds  = np.ones(12)

    def get_im_id(self):
        """
        Returns image id matrix.
        """
        return self.im_id
    
    def get_cloud_mask(self):
        """
        Returns cloud mask.
        """
        return self.cloud_mask
    
    def _timestamps_from_zarr(self):
        """
        Helper method used to format from datetime64 to date objects

        Returns: np.array of correctly formatted date objects
        """
        return np.array([np.datetime64(t).astype("datetime64[D]").astype(date) for t in 
                np.array(self.root["time_stamps"])])
    
    def _normalize_date(self, d):
        """
        Helper method for normalizing date input to date on form "YYYY-MM-DD"

        Parameters:
        d (str, datetime, date): date input

        Returns:
        d (date): correctly formated date

        Raises:
        TypeError: If d is an unsupported datatype
        """
        if isinstance(d, str):
            return datetime.fromisoformat(d).date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, date):
            return d
        else:
            raise TypeError(f"Unsupported date type: {type(d)}")
    
    def _post_2022_mask(self):
        """
        Creates a mask of the timestamps for which are 2022 or after
        """
        cutoff = date(2022, 1, 1)
        return self.timestamps>=cutoff
    
    def id_from_date(self,date_input):
        """
        Takes a date or collection of dates and returns corresponding time indices

        Paramters:
        date_input (list, tuple, string, datetime, date): date or collection of dates on form "YYYY-MM-DD"

        Returns:
        idices (list): time indices of the dates in the zarr cube. Dates that don't exist images fore are returned as None
        """
        #Normalize to list
        if not isinstance(date_input,(list,tuple)):
            date_input=[date_input]

        formated=[self._normalize_date(d) for d in date_input]

        #Sets None if the date doesn't exist
        indices = [self.date_to_index.get(d, None) for d in formated]

        return indices
    
    def range_to_indices(self, start, end):
        """
        Takes a start and end date and returns all corresponding indices

        Parameters: 
        start (date, datetime, str): start date
        end (date, datetime, str): end date

        Returns:
        np.array of indices corresponding to the date range
        """
        start, end = map(self._normalize_date, [start, end])
        mask = (self.timestamps >= start) & (self.timestamps <= end)
        return np.where(mask)[0]
    
    def info(self,printing=True):
        """
        Method for extracting and printing basic information about the cube

        Returns:
        data (dict): Dictionary containing basic data about the cube
        """
        data= {
            "shape": self.da_reflectance.shape,
            "num_dates": len(self.timestamps),
            "min_date": min(self.timestamps),
            "max_date": max(self.timestamps),
            "chunks": self.da_reflectance.chunksize,
        }
        
        if printing:
            print("Zarr Dataset Information")
            print("-" * 30)
            for key, value in data.items():
                print(f"{key:12}: {value}")

        return data

    def valid_time(self,p,t=None):
        """
        Takes a polygons and times and returns only the time ids that there exists images for

        Parameters:
        p (int): polygon index
        t (int, list, np.array, array-like): time or times to check images for

        Returns:
        t (np.array): All time indices that contain an image for given polygon and time indices
        """
        if t is None:
            t = np.arange(self.t_dim)
        t=np.array(t,ndmin=1)
        return t[np.where(self.im_id[p,t]!=-1)[0]]

    def cloud_free_time(self,p,t=None):
        """
        Takes a polygons and times and returns only the time ids that there exists images for that are also
        Cloud free

        Parameters:
        p (int): polygon index
        t (int, list, np.array, array-like): time or times to check images for

        Returns:
        t (np.array): All time indices that contain an image for given polygon and time indices
        """
        if t is None:
            t = np.arange(self.t_dim)
        t=np.array(t,ndmin=1)
        return t[np.where(self.cloud_mask[p,t])[0]]

    def no_image(self,p):
        """
        Checks if a polygon doesn't contain any images
        Paramters:
        p (int): poly idx

        Returns:
        Bool: True if there is no image for the polygon. Otherwise False
        """
        return np.all(self.im_id[p]==-1)
    
    def all_no_image(self):
        """ 
        Returns all polygon ids for which no images exist
        """
        mask = np.all(self.im_id == -1, axis=1)
        return np.where(mask)[0]
    
    def number_polys_for_t(self,t):
        """
        Returns the amount of polygons that have an image for a given date. If input is multiple dates, returns a dictionary of (t: no_polys) pairs.
        """
        #If more than one time id is entered
        if len(np.array(t,ndmin=1))>1:
            mask = self.im_id[:, t] != -1
            results = {}
            for ti in t:
                valid_polys = np.where(self.im_id[:, ti] != -1)[0]
                results[ti] = len(valid_polys)
            return results

        return len(np.where(self.im_id[:,t]!=-1)[0])
    
    def get_gdf(self):
        return self.gdf
    
    def get_first_year(self,p):
        """
        Takes a poly idx and checks for the first listed year for that polygon
        """
        return int(self.gdf.iloc[p]["firstYear"])

    def get_last_year(self,p):
        """
        Takes a poly idx and checks for the first listed year for that polygon
        """
        return int(self.gdf.iloc[p]["lastYear"])

    def im_stats(self, printing=True):
        """
        Method for getting and printing basic information about the images. How many slots are filled with actual images,
        how sparse it is etc.
        """
        im = np.array(self.im_id)
        total = im.size
        missing = (im == -1).sum()
        present = total - missing
        stats = {
            "num_polygons": im.shape[0],
            "num_timestamps": im.shape[1],
            "total_entries": total,
            "num_missing_total": int(missing),
            "num_present_total": int(present),
            "missing_ratio_total": float(missing / total),
        }
        if printing:
            print("im_id Coverage Summary")
            print("-" * 35)
            for key, value in stats.items():
                print(f"{key:24}: {value}")
            
        return stats

    def first_and_year_after_ids(self,p,cloud_free=True):
        """
        Returns valid time ids for a given polygon of the first year of restoration and the year after
        """
        if cloud_free:
            time_ids=self.cloud_free_time(p)
        else:
            time_ids=self.valid_time(p)
        dates=self.timestamps[time_ids]
        last_year=self.get_last_year(p)+1 #Due to year after
        first_year=self.get_first_year(p)
        first_year_ids = self.id_from_date([d for d in dates if d.year == first_year])
        last_year_ids  = self.id_from_date([d for d in dates if d.year == last_year])

        return first_year_ids, last_year_ids
    
    def ids_from_year(self,p,year,cloud_free=True):
        """
        Given a year, returns valid ts for that polygon and year
        """
        if cloud_free:
            time_ids=self.cloud_free_time(p)
        else:
            time_ids=self.valid_time(p)

        dates=self.timestamps[time_ids]
        year_ids=self.id_from_date([d for d in dates if d.year == year])
        
        return year_ids

    def get_2022_split_idx(self):
        """
        Returns the time-index of where 2022 begins
        """
        return np.where(self.post_2022)[0][0]

    def get_available_dates(self):
        """
        Returns all available dates there exists at least 1 image for
        """
        return self.timestamps
    
    def check_load_size(self, arr, max_gb=50):
        """
        Method for ensuring loadsize doesn't extend RAM capacity
        """
        est = arr.nbytes / 1e9
        if est > max_gb:
            raise MemoryError(f"Request would load {est:.2f} GB > {max_gb} GB limit.")

    def nearest_date(self, d):
        """
        Return closest available timestamp to the one entered
        """
        return min(self.timestamps, key=lambda x: abs(x - d))
    
    def valid_date(self, d, raise_error=False):
        """
        Checks if the date (d) exists in the dataset. Can be rigid and raise error if wished
        """
        d = self._normalize_date(d)
        if raise_error and d not in self.date_to_index:
            raise ValueError(f"Date {d} not in dataset.")
        return d in self.date_to_index
    
    def cloud_free_range_ts(self,p,first_year=None,last_year=None):
        """
        Returns the range of ts between 2 given years. If None are given, instead returns all cloud free ts
        for the restoration time
        """
        if first_year is None:
            first_year=self.get_first_year(p)
        if last_year is None:
            last_year=self.get_last_year(p) + 1 #Due to year after
        start=datetime(first_year,6,1)
        end=datetime(last_year,8,31)
        ts_range=self.range_to_indices(start,end)
        ts=self.cloud_free_time(p,ts_range)
        return ts
    
    def per_pasture_mean_std(self):
        """
        Creates 2 arrays containing the means and stds for each individual pastures for each of their channels
        """
        ps=np.arange(self.p_dim)
        means=np.zeros((len(ps),12))
        stds=np.zeros((len(ps),12))
        for p in ps:
            ts=self.cloud_free_range_ts(p)
            print("Polygon: ", p)

            imgs=self.reflectance.oindex[p,ts].astype(np.float32) #(T,C,H,W)
            mask_ids=self.mask_idx_map[p][ts]
        
            masks=self.precomputed_masks[p][mask_ids]
            #Making mask the correct broadcasting shape (T,1,H,W)
            masks_formated=masks[:,None,:,:]

            #Apply mask and set Nan outside
            masked_imgs=np.where(masks_formated,imgs,np.nan)

            p_means = np.nanmean(masked_imgs, axis=(0, 2, 3))  # shape (12,)
            p_stds  = np.nanstd(masked_imgs,  axis=(0, 2, 3))  # shape (12,)

            means[p]=p_means
            stds[p]=p_stds

        np.save(DATA/"mean_std"/"pasture_means.npy", means)
        np.save(DATA/"mean_std"/"pasture_stds.npy", stds)

        return means, stds
    
    def set_other_pasture_mean_std(self,mean_path,std_path):
        """
        Method for changing the pasture means and stds from the ones hardcoded in initialization
        """
        self.pasture_means=np.load(mean_path)
        self.pasture_stds=np.load(std_path)
    
    def set_other_glob_mean_std(self, mean_std_path):
        """
        Method for changing the gloBAL means and stds from the ones hardcoded in initialization
        """
        glob_mean_stds=np.load(mean_std_path)
        self.glob_means=glob_mean_stds[:,0]
        self.glob_stds=glob_mean_stds[:,1]

    def single_pasture_mean_std(self,p,ts):
        """
        Normalizes pasture wise from only given ts and p of an image
        """
        imgs=self.reflectance.oindex[p,ts].astype(np.float32) #(T,C,H,W)
        mask_ids=self.mask_idx_map[p][ts]
    
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]
        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,np.nan)
        
        means = np.nanmean(masked_imgs, axis=(0, 2, 3))  # shape (12,)
        stds  = np.nanstd(masked_imgs,  axis=(0, 2, 3))  # shape (12,)

        return means, stds
    
    def get_custom_normalized_images(self,p,t,mean,std,remove_border=False):
        """
        Expects mean, stds to have shape (12,)
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32)
        border_mask = (imgs == -9999)

        #Reformats to handle the same way as multiple images
        single_im=(imgs.ndim==3)
        if single_im:
            imgs=imgs[None,...]
            t=np.atleast_1d(t)

        mean = np.nan_to_num(mean, nan=0.0)
        std  = np.nan_to_num(std,  nan=1.0)
        # Reshape for broadcasting: (1,C,1,1)
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    
        # Normalize
        imgs = (imgs - mean) / std
    
        if remove_border:
            imgs[border_mask] = 0
    
        if single_im:
            return imgs[0]
        
        return imgs
    
    def get_custom_normalized_masked_images(self,p,t,mean,std,fill_val=np.nan):
        """
        Takes p and t (or ts) and returns nan masked images for a custom normalization
        """
        imgs=self.get_custom_normalized_images(p,t,mean,std) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs

    def get_normalized_images(self,p,t, remove_border=False):
        """
        Takes a polygon and a time (or multiple timesteps) and returns unifrom normalized images
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32)
        border_mask = (imgs == -9999)

        #Reformats to handle the same way as multiple images
        single_im=(imgs.ndim==3)
        if single_im:
            imgs=imgs[None,...]
            t=np.atleast_1d(t)
        
        #-----------------SAME FOR POST AND PRE 2022----------------------"
        imgs=imgs/10000

        ##Mask of which are post 2022
        #mask=self.post_2022[t]
       #
        ##Does normalization for pre 2022 if there are any
        #if (~mask).any():
        #    imgs[~mask]=imgs[~mask]/10000
        ##Does normalization for post 2022 if there are any
        #if (mask).any():
        #    imgs[mask]=(imgs[mask]-1000)/10000

        #Reformats single im to remove t-dimension
        
        if remove_border:
            imgs[border_mask] = 0

        if single_im:
            return imgs[0]
        
        return imgs
    
    def get_pasture_normalized_images(self,p,t,remove_border=False):
        """
        Takes a polygon and a time (or multiple timesteps) and returns normalized images per pasture
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32)
        border_mask = (imgs == -9999)

        #Reformats to handle the same way as multiple images
        single_im=(imgs.ndim==3)
        if single_im:
            imgs=imgs[None,...]
            t=np.atleast_1d(t)

        # Load per pasture stats
        mean = self.pasture_means[p] # (12,)
        std  = self.pasture_stds[p]  # (12,)
        mean = np.nan_to_num(mean, nan=0.0)
        std  = np.nan_to_num(std,  nan=1.0)

        # Reshape for broadcasting: (1,C,1,1)
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    
        # Normalize
        imgs = (imgs - mean) / std
    
        if remove_border:
            imgs[border_mask] = 0

        if single_im:
            return imgs[0]
        
        return imgs
    
    def get_global_normalized_images(self,p,t,remove_border=False):
        """
        Takes a polygon and a time (or multiple timesteps) and returns globally normalized images per pasture
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32)
        
        border_mask = (imgs == -9999)
        #Reformats to handle the same way as multiple images
        single_im=(imgs.ndim==3)
        if single_im:
            imgs=imgs[None,...]
            t=np.atleast_1d(t)
        
        # Load per pasture stats
        mean = self.glob_means # (12,)
        std  = self.glob_means # (12,)
        mean = np.nan_to_num(mean, nan=0.0)
        std  = np.nan_to_num(std,  nan=1.0)

        # Reshape for broadcasting: (1,C,1,1)
        mean = mean[None, :, None, None]
        std  = std[None, :, None, None]
    
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
    
        # Normalize
        imgs = (imgs - mean) / std
    
        if remove_border:
            imgs[border_mask] = 0

        if single_im:
            return imgs[0]
        
        return imgs
    
    def get_masked_images(self,p,t,fill_val=np.nan):
        """
        Takes p and t (or ts) and returns nan masked images
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)
        
        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs

    def get_normalized_masked_images(self,p,t,fill_val=np.nan):
        """
        Takes p and t (or ts) and returns normalized nan masked images
        """
        imgs=self.get_normalized_images(p,t) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs
    
    def get_pasture_normalized_masked_images(self,p,t,fill_val=np.nan):
        """
        Takes p and t (or ts) and returns pasture normalized masked images
        """
        imgs=self.get_pasture_normalized_images(p,t) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs
    
    def get_global_normalized_masked_images(self,p,t,fill_val=np.nan):
        """
        Takes p and t (or ts) and returns nan globally normalized masked images
        """
        imgs=self.get_global_normalized_images(p,t) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs

    def get_global_normalized_anti_masked_images(self,p,t,fill_val=np.nan):
        """
        Method similar to the masked_images methods but where all polygons that would exist in an image
        are masked out so only the surroundnings remain.
        """
        imgs=self.get_global_normalized_images(p,t,remove_border=True) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        dates=self.timestamps[t]
        year=dates[0].year
        masks=self.get_cluster_inverse_mask(p,year)
        
        T = imgs.shape[0]
        masks = np.broadcast_to(masks, (T, *masks.shape))

        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs

    def get_pasture_normalized_anti_masked_images(self,p,t,fill_val=np.nan):
        """
        Method similar to the masked_images methods but where all polygons that would exist in an image
        are masked out so only the surroundnings remain.
        """

        imgs=self.get_pasture_normalized_images(p,t,remove_border=True) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        dates=self.timestamps[t]
        year=dates[0].year
        masks=self.get_cluster_inverse_mask(p,year)
        
        T = imgs.shape[0]
        masks = np.broadcast_to(masks, (T, *masks.shape))

        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs
    
    def get_normalized_anti_masked_images(self,p,t,fill_val=np.nan):
        """
        Method similar to the masked_images methods but where all polygons that would exist in an image
        are masked out so only the surroundnings remain.
        """
        imgs=self.get_normalized_images(p,t,remove_border=True) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        dates=self.timestamps[t]
        year=dates[0].year
        masks=self.get_cluster_inverse_mask(p,year)
        
        T = imgs.shape[0]
        masks = np.broadcast_to(masks, (T, *masks.shape))

        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs
    
    def get_custom_normalized_anti_masked_images(self,p,t,mean,std,fill_val=np.nan):
        """
        Method similar to the masked_images methods but where all polygons that would exist in an image
        are masked out so only the surroundnings remain.
        """
        imgs=self.get_custom_normalized_images(p,t,mean,std,remove_border=True) #(T,C,H,W) or (C,H,W) for single image

        # Handle single-image case: shape (C,H,W)
        single_im = (imgs.ndim == 3)
        if single_im:
            imgs = imgs[None]    # sets to (1,C,H,W)
            t = np.atleast_1d(t)

        dates=self.timestamps[t]
        year=dates[0].year
        masks=self.get_cluster_inverse_mask(p,year)
        
        T = imgs.shape[0]
        masks = np.broadcast_to(masks, (T, *masks.shape))

        #Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        #Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)

        if single_im:
            return masked_imgs[0]   # remove T dim

        return masked_imgs

    def get_mean_std_normalized_masked_images(self,p,t,fill_val=np.nan):
        """
        Takes a polygon and a time (or multiple timesteps) and returns normalized images. THIS IS AN OLD METHOD. DONT USE
        """
        imgs=self.reflectance.oindex[p,t].astype(np.float32)
        mean_pre = np.array([1338.64506742, 1376.68814735, 1546.33043015, 1430.64841003, 1854.27603069,
                    3059.03081024, 3468.54776612, 3590.84442677, 3698.06519941, 3754.62180706,
                    2518.95928305, 1835.32583504])                                                                        
        std_pre = np.array([507.8831745,   504.17139126,  496.49872656,  518.92682152,  579.8882168,
                            974.87750551, 1164.17756379, 1266.05597324, 1254.34816975, 1260.1197776,
                            848.25198051,  615.43690804])
        mean_post = np.array([1476.04256481, 1496.95820624, 1666.32016797, 1532.13684569, 1992.47687193,
                            3275.32865644, 3706.71518792, 3844.12386983, 3944.81476631, 4062.55381004,
                            2613.50874538, 1910.11537083])
        std_post = np.array([694.31440625,  679.66519254,  648.55349506,  661.12880816,  709.60072485,
                            1032.42851096, 1204.87657449, 1305.60526224, 1284.93148467, 1365.93179377,
                            876.71152826,  679.48437188])

        # Reshaping 
        mean_pre  = mean_pre[None, :, None, None]
        std_pre   = std_pre[None, :, None, None]
        mean_post = mean_post[None, :, None, None]
        std_post  = std_post[None, :, None, None]

        # Reformats to handle the same way as multiple images
        single_im=(imgs.ndim==3)
        if single_im:
            imgs=imgs[None,...]
            t=np.atleast_1d(t)

        # Mask of which are post 2022
        mask=self.post_2022[t]
       
        # Does normalization for pre 2022 if there are any
        if (~mask).any():
           imgs[~mask]=(imgs[~mask]-mean_pre)/std_pre
        # Does normalization for post 2022 if there are any
        if (mask).any():
           imgs[mask]=(imgs[mask]-mean_post)/std_post

        # Reformats single im to remove t-dimension
        if single_im:
            imgs = imgs[0]
        
        # Masks --------------------------------
        mask_ids=self.mask_idx_map[p][t]
        masks=self.precomputed_masks[p][mask_ids]
        # Making mask the correct broadcasting shape (T,1,H,W)
        masks_formated=masks[:,None,:,:]

        # Apply mask and set Nan outside
        masked_imgs=np.where(masks_formated,imgs,fill_val)
        if single_im:
            return masked_imgs[0]   # remove T dim
        
        return masked_imgs

    def get_unpadded_img(self,p,t):
        """
        Removes the -9999 padding of an image of shape (C,H,W). Returns the cropped image.
        """
        img = self.reflectance.oindex[p,t]
        valid_mask = np.any(img != -9999, axis=0)  # shape (100, 100)
        rows = np.any(valid_mask, axis=1)
        cols = np.any(valid_mask, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        return img[:, row_min:row_max+1, col_min:col_max+1]
    
    def update_cloud_mask(self, mask, poly_ids = None):
        """
        Updates cloud_mask for polygon(s) in poly_ids.
        poly_ids: int or iterable of ints
        mask: np.ndarray of shape (T,) or (len(poly_ids), T), dtype bool
        """
        if isinstance(poly_ids, int):
            poly_ids = [poly_ids]
            mask = mask[None, ...] if mask.ndim == 1 else mask
        if poly_ids == None:    # If none specified: use all
            n_polygons = np.shape(self.im_id)[0]
            poly_ids = np.arange(0,n_polygons)
        for i, p in enumerate(poly_ids):
            # self.cloud_mask[p][~mask[i,:]] = False
            self.cloud_mask[p,:] = mask[i,:]
    
    def load_cloud_mask(self,load_path):
        """Sets cloud mask from file, if load_path is None, it sets the cloudmask to a boolean which is false for the invalid
        images in im_id
        """
        if load_path is None:
            mask=self.im_id!=-1
        else:
            mask=np.load(load_path)
        self.cloud_mask=mask

    def save_cloud_mask_to_disk(self, save_path: str):
        """
        Saves the cloud mask of this object to specified save_path.
        """
        cloud_mask = self.cloud_mask
        np.save(save_path, cloud_mask)

    def reset_cloud_mask(self):
        """
        Resets the cloud mask to default, as specified by cloud_path
        """
        self.load_cloud_mask(self.cloud_path)

    def calculate_non_cloudy_fracs(self,poly_ids=None,print_stats=False):
        """
        Calculates the fraction of non-cloudy images of the time series for the specified polygons.
        """
        if poly_ids == None:
            poly_ids = np.arange(0,self.p_dim)

        non_cloudy_fracs = np.zeros(len(poly_ids))
        for i, poly_id in enumerate(poly_ids):
            non_cloudy_times = np.where((self.cloud_mask[poly_id,:]))[0]    
            cloudy_times = np.where(~self.cloud_mask[poly_id,:] & (self.im_id[poly_id,:] != -1))[0]     # Cloud mask is false, and image exist
            non_cloudy_frac = len(non_cloudy_times)/(len(non_cloudy_times) + len(cloudy_times))
            non_cloudy_fracs[i] = non_cloudy_frac

            if print_stats:
                print(f"For polygon {poly_id} {non_cloudy_frac*100}% of the images were deemed non-cloudy")
        
        if print_stats:
            print(f"On average, {np.mean(non_cloudy_fracs)*100}% of the images were deemed non-cloudy")

        return non_cloudy_fracs     

    def align_poly(self,p,t):
        im_id=self.im_id[p,t]
        x0=self.x0[im_id]
        y0=self.y0[im_id]
        H=self.H
        W=self.W
        poly=np.array(self.gdf.iloc[p]["geometry"].exterior.coords)
        x,y=pf.get_world_axis(x0,y0,H,W,meter_per_pxl=10)
        aligned_poly, poly_mask=pf.align_poly(poly, H, W, x, y, clip_if_poly_outside=True)
        return aligned_poly, poly_mask
    
    def align_poly_to_other_im(self,p,im_id):
        x0=self.x0[im_id]
        y0=self.y0[im_id]
        H=self.H
        W=self.W
        poly=np.array(self.gdf.iloc[p]["geometry"].exterior.coords)
        x,y=pf.get_world_axis(x0,y0,H,W,meter_per_pxl=10)
        aligned_poly, poly_mask=pf.align_poly(poly, H, W, x, y, clip_if_poly_outside=True)
        return aligned_poly, poly_mask
    
    def remove_outliers(self,filename):
        """
        Given a filename that points do a dictionary with polygons as keys and timesteps as , set's these to invalid+cloudy so that they are not used
        when calling valid/ cloud_free iamges
        """
        with open(filename, "rb") as f:
            outliers = pickle.load(f)
        ps=np.array([int(p) for p in outliers.keys()])
        for p in ps:
            for t in outliers[p]:
                self.im_id[p,t]=-1
                self.cloud_mask[p,t]=False

    def remove_negatives(self,filename):
        """
        Removes images with negative values.
        """
        im_neg=np.load(filename)
        self.im_id[im_neg]=-1
        self.cloud_mask[im_neg]=False

    def create_mask_cache(self):
        """
        Initialization method for setting up the polygons masks. As the polygons will not be in a 
        consistant place always, this is needed to properly mask the polygons. Running this method and then the
        save_mask_cache() will ensure the mask cache exists.

        Returns: precomputed_masks (dict), key: p, item: np.array (p_E,H,W) mask stack for that polygon 
                mask_idx_map (dict), key: p, item np.array (T_glob) which time_ids correspond to which mask for that polygon
        """
        precomputed_masks = {}
        mask_idx_map = {}

        for p in range(self.p_dim):
            eras=self.compute_eras(p)
            poly_masks, idx_map=self._build_mask_and_img_map_arrays(eras)

            precomputed_masks[p]=poly_masks
            mask_idx_map[p]=idx_map

        #Sets the attributes before returning
        self.precomputed_masks=precomputed_masks
        self.mask_idx_map=mask_idx_map

        return precomputed_masks, mask_idx_map
    
    def save_mask_cache(self,path=POLY_MASKS/"poly_mask.pk"):
        with open(path, "wb") as f:
                pickle.dump(
                    {
                        "precomputed_masks": self.precomputed_masks,
                        "mask_idx_map": self.mask_idx_map
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

    def load_mask_cache(self, path=POLY_MASKS/"poly_mask.pk"):
        """
        See above methods for what the file is supposed to contain
        """
        with open(path, "rb") as f:
            cache = pickle.load(f)
        self.precomputed_masks = cache["precomputed_masks"]
        self.mask_idx_map = cache["mask_idx_map"]

    def _build_mask_and_img_map_arrays(self,eras):
        """
        Takes eras from compute eras and creates the structure of the items
        that are going into the dictionaries in set_mask_cache
        """

        #Build era stack, stacking the masks along the number of eras
        poly_masks=np.stack([mask for (_,mask) in eras], axis=0)

        #Building glob_t_id to mask for given era. Initializing array
        idx_map=np.zeros(self.t_dim,dtype=np.uint16)

        for era_index, (era_ts,_) in enumerate(eras):
            idx_map[era_ts]=era_index

        return poly_masks, idx_map

    def compute_eras(self,p):
        """
        Computes "eras" which are the consequtive valid times for a polygon that the polygon is
        in the same place in the picture
        """
        ts=self.valid_time(p)
        change_times=self.poly_mask_position_change(p)
        eras=[]

        #If it doesn't change, there's only one era
        if len(change_times)==0:
            _, poly_mask=self.align_poly(p,ts[0])
            return [(ts,poly_mask)]

        #Loops through each change and creates an era with corresponding ts
        start_t=0
        for ct in change_times:
            end_t=np.where(ts==ct)[0][0]
            ts_for_era=ts[start_t:end_t]
            _, poly_mask=self.align_poly(p,ts_for_era[0])
            eras.append((ts_for_era,poly_mask))
            start_t=end_t

        #Last era
        _, poly_mask=self.align_poly(p,ts[start_t])
        eras.append((ts[start_t:],poly_mask))
        return eras
    
    def poly_mask_position_change(self,p):
        """
        Checks if poly-mask is consistent or moves and returns time ids for where it moves
        """
        ts=self.valid_time(p)

        # Creates list of aligned masks
        aligned_masks=[]
        for t in ts:
            poly_aligned,poly_mask=self.align_poly(p,t)
            aligned_masks.append(poly_mask)

        # Stacks them as np array
        stack=np.stack(aligned_masks)

        # Creates a t-1 x H x W boolean array
        diff=stack[1:]!=stack[:-1]

        # Gets indices of where they dont match with previous mask
        changed = np.any(diff, axis=(1,2))
        indices = np.where(changed)[0] + 1
        change_times=ts[indices]
        return change_times 

    def build_cluster_index(self,cluster_col="cluster"):
        """
        Creates an attribute "cluste_to_polys", which is a dict where the keys are clusters and values are all polygons that
        is part of that cluster. Expects the gdf to have a cluster collumn
        """
        clusters = self.gdf[cluster_col].values
        self.cluster_to_polys = {}

        for p, c in enumerate(clusters):
            if c not in self.cluster_to_polys:
                self.cluster_to_polys[c] = []
            self.cluster_to_polys[c].append(p)

    def cache_all_cluster_inverse_masks(self):
        """
        method for creating a cache of "anti-masks" for each polygon and year.
        """
        for p in np.arange(0,self.p_dim):
            for year in np.arange(2018,2026):
                self.cache_cluster_inverse_mask(p,year)

    def cache_cluster_subset(self, p_list):
        """
        method for creating a cache of "anti-masks" for a subset of polygons.
        """
        for p in p_list:
            for year in range(2018, 2026):
                self.cache_cluster_inverse_mask(p, year)

    def cache_cluster_inverse_mask(self, p_ref, year):
        """
        Mask out all polygons in the same cluster.
        Returns background-only mask.
        """
        if not hasattr(self, "_cluster_mask_cache"):
                self._cluster_mask_cache = {}

        key = (p_ref, year)
        mask = np.ones((self.H, self.W), dtype=bool)
        cluster=self.gdf.loc[p_ref]["cluster"]
        ts_ref=self.ids_from_year(p_ref,year)
        im_id=self.im_id[p_ref,ts_ref[0]]

        for p in self.cluster_to_polys[cluster]:
            _, poly_mask=self.align_poly_to_other_im(p,im_id)
            if poly_mask is None:
                continue

            mask &= ~poly_mask
        self._cluster_mask_cache[key] = mask
        return mask
    
    def get_cluster_inverse_mask(self, p_ref, year):
        """
        Returns the anti-mask for a polygon p_ref. P_ref is reffering to the polygon used to extract the image
        that get's anti-masked
        """
        if not hasattr(self, "_cluster_mask_cache"):
            self._cluster_mask_cache = {}

        key = (p_ref, year)

        if key not in self._cluster_mask_cache:
            return self.cache_cluster_inverse_mask(p_ref, year)

        return self._cluster_mask_cache[key]

    def plot_rgb(self, p, ts,title=None,normalize=True,mask=False,directory_name="zarr_ims",path=IMGS,additional_info="",plot_poly=True,fig=None,ax=None):
        """
        Plots and saves an RGB image for a single polygon and a sinlge time or multiple times. If only a single time is
        sent, will return fig and ax for that time. If normalize is set to True, will normalize the image, if mask= True will mask the image
        """

        # Normalize to list
        if not hasattr(ts, "__len__") or isinstance(ts, (str, int)):
            ts = [ts]
        
        if mask:
            imgs=self.get_normalized_masked_images(p,ts)
        elif normalize:
            imgs=self.get_normalized_images(p,ts)
        else:    
            imgs=self.reflectance.oindex[p,ts]

        savedir=ut.make_savedir(directory_name,path=path,additional_info=additional_info)

        # Single image case as ndim is 3 for a single image, Turn into shape 1,C,H,W to handle similarly to T,C,H,W
        if imgs.ndim==3:
            imgs=imgs[None,...]

        for img, t in zip(imgs,ts):
            year=self.timestamps[t].year
            use_title=title or f"Polygon_{p}_{self.timestamps[t]}.png"
            aligned_poly=None
            if plot_poly:
                aligned_poly,poly_mask=self.align_poly(p,t)
            fig, ax = self.plot_image_rgb(img,use_title,savedir,aligned_poly=aligned_poly,fig=fig,ax=ax)
            if len(ts)==1:
                return fig,ax
            fig=None
            ax=None
            plt.close()
        
    def plot_image_rgb(self, img, title, savedir, aligned_poly=None, fig=None,ax=None):
        """
        Function for plotting from an image
        """

        # Create fig ax if not passed
        if (fig is None) and (ax is None):
            fig, ax =pf.plot_rgb(img,title=title)

        else:
            fig, ax=pf.plot_rgb(img,title=title,fig=fig,ax=ax)

        # Plotting polygon unsless otherwise specified
        if aligned_poly is not None:
            ax.plot(aligned_poly[:,0], aligned_poly[:,1],color="red")

        plt.savefig(savedir/title)
        return fig,ax
    
    def plot_image_1channel(self, img, title, savedir, aligned_poly=None, fig=None,ax=None):
        """
        Function for plotting from an image of one channel
        """

        #Create fig ax if not passed
        if (fig is None) and (ax is None):
            fig, ax=plt.subplots()
            ax.imshow(img)

        else:
            ax.imshow(img)

        #Plotting polygon unsless otherwise specified
        if aligned_poly is not None:
            ax.plot(aligned_poly[:,0], aligned_poly[:,1],color="red")

        plt.savefig(savedir/title)
        return fig,ax
    
    def plot_image_grid(self,p,ts,savedir,filename = None, title = None,plot_poly=True,select_cloudy=False): 
        # Normalize to list
        if not hasattr(ts, "__len__") or isinstance(ts, (str, int)):
            ts = [ts] 

        im_series = self.get_normalized_images(p,ts)  
        num_images = np.shape(im_series)[0]
        ncols = np.ceil(np.sqrt(num_images)).astype(int)
        nrows = np.ceil(num_images / ncols).astype(int)
        fig = plt.figure(figsize=(32, 32))
      
        for i, t in enumerate(ts):
            ax = fig.add_subplot(nrows,ncols,i+1)
            fig, ax = pf.plot_rgb(im_series[i,:,:,:],fig=fig,ax=ax,title=self.timestamps[t])
            if plot_poly:
                aligned_poly,_ = self.align_poly(p,t)
                ax.plot(aligned_poly[:,0], aligned_poly[:,1],color="red")
                
            # Add red frame if cloudy
            if select_cloudy and ~self.cloud_mask[p,t]:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3)
            
            # Hide numbers but keep red border
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        use_title=title or f"Polygon {p} image grid"
        fig.suptitle(use_title, fontsize=36)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename = filename or f"Polygon_{p}_image_grid.png"
        plt.savefig(savedir/filename)
        plt.close()

class Zarr_tests:
    """Class for running tests on the Zarr cube
    """
    def __init__(self, cube: Zarr):
        self.cube=cube

    def corrupt_images_test(self):
        """Checks if there are images that contain negative values or other faulty values
        """
        cube=self.cube
        ps=np.arange(cube.p_dim)
        corrupt_ids=[]
        for p in ps:
            ts=cube.valid_time(p)
            imgs=cube.reflectance.oindex[p,ts]

            #Gives true for channels that have fully negative images
            #t,c,h,w -> t,c
            c_neg_filled = np.all((imgs < 0), axis=(2, 3))

            #Gives true for times where any channel has fully padded image
            #t,c -> t
            corrupt_neg_im=np.any(c_neg_filled,axis=1)



            for idx , t_id in enumerate(ts):
                if corrupt_neg_im[idx]:
                    corrupt_ids.append((p,t_id))

        return corrupt_ids

def remove_corrupt_images(zarr_path,back_up_path,cube):
    """
    Removes corrupt images from zarr im_id and creates a backup of im_ids
    """

    tests=Zarr_tests(cube)
    corrupt=tests.corrupt_images_test()

    if len(corrupt) == 0:
        print("No corrupt entries found.")
        return

    #* makes it into len(corrupt) passes that then are zipped
    corrupt_ps, corrupt_ts=zip(*corrupt)
    cube.im_id[corrupt_ps,corrupt_ts]=-1

    src = zarr.open(zarr_path, mode="r")
    backup = zarr.open(back_up_path, mode="w")
    
    zarr.copy(src["image_index"], backup, name="image_index_backup")
    
    # Re-open original zarr in read/write mode
    orig = zarr.open(zarr_path, mode="r+")
    orig["image_index"][:] = cube.im_id

if __name__=="__main__":
    import geopandas as gpd

    #------------------------------------------------------------------------------------------------------------------------------
    """
    Example of how the class is efficiently used
    """
    #------------------------------------------------------------------------------------------------------------------------------
    
    # Create cube object (note: this assumes existing .zarr in 'ZARR' path; see zarr_creation.py for .zarr creation)
    cloud_path = CLOUD_MASKS/"cloud_mask_04_17_thresholds_20_15_10.npy"
    cube = Zarr(ZARR, GPKGS/"final02-20.gpkg", cloud_path=cloud_path)

    # Choose a polygon
    polygon = 15

    # Get timesteps for all cloud free images from the full restoration process
    ts = cube.cloud_free_range_ts(polygon)

    # Load all images from these ts for the given polygon and mask them as well as normalize them using the global mean and std (which are preloaded / need to be calculated) (the mask cache also needs to have been created)
    masked_imgs = cube.get_global_normalized_masked_images(polygon,ts)

    #------------------------------------------------------------------------------------------------------------------------------
    """
    Ways of getting relevant time steps from the cube
    """
    #------------------------------------------------------------------------------------------------------------------------------

    # Choose polygon index
    polygon = 10

    # Get all ts where polygon 10 has valid images (where ts are the time indices) if you remember: (p,t) -> a single image
    all_ts=cube.valid_time(polygon)

    # Or all cloud free time steps that have valid images
    all_cloud_free_ts = cube.cloud_free_time(polygon)

    # Or for a collection of ts, return only those for which the polygon has existing images for (same thing can be done for cloud_free_time).
    collection_of_ts = [10, 11, 12, 13, 14, 15]
    valid_collection_of_ts = cube.valid_time(polygon,collection_of_ts)

    # Get all valid or cloud free ts from the first and last year of a polygons restoration process
    first_ts, last_ts = cube.first_and_year_after_ids(polygon)
    first_ts, last_ts = cube.first_and_year_after_ids(polygon, cloud_free=True)  # Instead only gives those that are not cloudy according to the cloud mask

    # Get valid (or cloud free) ts from a given year of a pasture
    year = 2022
    valid_ts_2022 = cube.ids_from_year(polygon, 2022)
    
    # Get all cloud free timesteps from the full restoration process of a pasture
    cloud_free_ts = cube.cloud_free_range_ts(polygon)

    # Get all cloud free timesteps between 2 specified years for a pasture
    cloud_free_ts = cube.cloud_free_range_ts(polygon, 2018, 2022)

    # Choose start date and end date
    start_date = "2018-06-22"
    end_date = "2020-07-01"

    # Get corresponding ts for all existing timesteps between these 
    range_of_ts = cube.range_to_indices(start_date,end_date)

    print("GOT HEREEEE")
    sys.exit()

    #------------------------------------------------------------------------------------------------------------------------------
    """
    Ways of initializing the cube
    """
    #------------------------------------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------------------------------
    """
    How to create important chaches, mean_std-files etc
    """
    #------------------------------------------------------------------------------------------------------------------------------
    
    sys.exit()
    
    cube=Zarr(ZARR,GPKGS/"final02-20.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    #cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    #cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")
    #p=697
    #ts=cube.cloud_free_time(p)
#
    #cube.plot_rgb(p,ts,mask=True,plot_poly=True)
    #cube.per_pasture_mean_std()
   
    print(cube.pasture_means.shape)
    #cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    #cube.im_stats()

    #map_path=DATA/"map_countries.shp"
    #sweden = gpd.read_file(map_path)
    #gdfer=gc.GDFFilter(GPKGS/"final02-20.gpkg")
    #gdf=gdfer.cluster_by_distance(1500)

    #num_clusters = gdf["cluster"].nunique()
    #print("Number of clusters:", num_clusters)

    #cluster_sizes = gdf["cluster"].value_counts().sort_index()
    #print(cluster_sizes)

    ## Filter to only Sweden
    #sweden = sweden[sweden['SOVEREIGNT'] == 'Sweden']
    #sweden = sweden.to_crs(gdf.crs)

    #fig, ax = plt.subplots(figsize=(8, 8))
    #sweden.boundary.plot(ax=ax, color="black", linewidth=1)

    #centroids = gdf.copy()
    #centroids["geometry"] = gdf.geometry.centroid

    #centroids.plot(
    #    ax=ax,
    #    column="cluster",
    #    categorical=True,
    #    markersize=5,
    #    legend=False
    #)

    #savedir=ut.make_savedir("zarr_ims")
    #plt.savefig(savedir/"Sweden.png")
    #plt.close()
    #
    #
    #max_size = cluster_sizes.max()
    #bins = np.arange(1, max_size + 2)   # integer bins

    #fig, ax = pf.make_hist(
    #    cluster_sizes,
    #    bins=bins,
    #    title="Cluster Size Histogram",
    #    xlabel="Number of Polygons in Cluster",
    #    ylabel="Count of Clusters",
    #    color="steelblue"
    #)

    #ax.locator_params(axis="x", integer=True,nbins=20)  # optional

    #plt.savefig(savedir/"Cluster_stats.png")