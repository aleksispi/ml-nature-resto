import classes.zarr_class as zc
import functions.plotting_functions as pf
import functions.utils as ut
import numpy as np
import functions.stat_analysis_methods as sam
from paths import ZARR,GPKGS,IMGS, DATA, REPORT_FIGS
import matplotlib.pyplot as plt
import datetime
import pickle
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import os
from scipy.stats import spearmanr
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

from matplotlib.ticker import MaxNLocator, FuncFormatter
class data_statistics:
    """Class for running statistics on our dataset
    """

    def __init__(self, zarr_cube,savedir=IMGS/"data_statistics",dataname="data.pkl"):
        self.cube=zarr_cube
        self.savedir=savedir
        self.colors={
            "ndvi":"tab:green",
            "evi" : "lightgreen",
            "ndmi": "cornflowerblue",
            "msavi": "coral"
        }
        self.data_collection=self.load_data(savedir/"data_files"/dataname)

    def get_ndvi_diff(self,p):
        """Takes a polygon and returns the difference of nvdi of the composite image of the first year and the year after
        """
        cube=self.cube
        first_ids,last_ids=cube.first_and_year_after_ids(p,cloud_free=True)
        first_ims=cube.get_normalized_masked_images(p,first_ids)
        last_ims=cube.get_normalized_masked_images(p,last_ids)
        first_ndvi=np.median(sam.ndvi_median_series(first_ims))
        last_ndvi=np.median(sam.ndvi_median_series(last_ims))
        diff_ndvi=last_ndvi-first_ndvi
        return diff_ndvi
    

    def seq_len_diff(self):
        """returns a list of differences in number of images from first vs last years
        """

        cube=self.cube
        ps=np.arange(cube.p_dim)
        seq_diffs=[]

        for p in ps:
            first, last=cube.first_and_year_after_ids(p)
            if (len(first)==0) or (len(last)==0):
                continue
            diff=len(last)-len(first)
            seq_diffs.append(diff)
        
        return seq_diffs
    

    def plot_seq_len_hist(self, seq_diffs, savedir=None, title="sequence_length_difference_histogram.png"):
        # Force integer dtype (this also validates the data)
        seq_diffs = np.asarray(seq_diffs, dtype=int)

        min_diff = seq_diffs.min()
        max_diff = seq_diffs.max()

        bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)

        mean_diff = seq_diffs.mean()
        median_diff = np.median(seq_diffs)

        plt.figure(figsize=(7, 4.5))

        plt.hist(
            seq_diffs,
            bins=bins,
            color="lightblue",
            edgecolor="black",
            alpha=0.85
        )

        plt.axvline(0, color="tab:red", linestyle="--", linewidth=2, label="No change")
        plt.axvline(mean_diff, color="tab:blue", linewidth=2, label=f"Mean = {mean_diff:.2f}")
        plt.axvline(median_diff, color="tab:green", linewidth=2, label=f"Median = {median_diff:.0f}")

        plt.xlabel("Difference in number of images (last − first year)")
        plt.ylabel("Number of pastures")
        plt.title("Change in observation count between first and last year")

        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        savepath = self.savedir / "histograms" / title
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=300)

    def hist_ndvi_diff(self,title="Ndvi_histogram.png"):
            cube=self.cube
            ps=np.arange(0,cube.p_dim)
            diffs=[]
            for p in ps:
                try:
                    diff=self.get_ndvi_diff(p)
                    diffs.append(diff)
                except Exception:
                    continue
            fig,ax=pf.make_hist(diffs,title=title)
            savedir=self.savedir/"ndvi"
            savedir.mkdir(parents=True,exist_ok=True)
            plt.savefig(savedir/title)

    def polygon_normalization_range(self,p):
        """Does normalization for full timeseries of a polygon and returns the range of values.
        Ignores the edge-padding
        """
        pass

    def area_statistics(self, title="Area distribution of polygons"):
        with plt.rc_context({"font.size": 14}):
            gdf = self.cube.get_gdf()
            areas = np.floor(gdf["calc_area"])

            min_a = areas.min()
            max_a = areas.max()
            bin_edges = np.linspace(min_a, max_a, 21)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                areas,
                bins=bin_edges,
                edgecolor="black",
                alpha=0.7
            )

            ax.set_xticks(bin_centers[::3])
            ax.set_xticklabels(
                [f"{int(x):,}" for x in bin_centers[::3]],
                rotation=30,
                ha="right"
            )

            ax.set_xlabel("Area (km²)")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        return fig, ax 
    
    def get_reflectance_stats(self,p,ts,remove_all_negatives=False):
        """Takes ts and p and calculates the mean and std of each channel for each image, returning two arrays of
        shape (T,C) containing these
        """
        cube=self.cube
        imgs=np.asarray(cube.reflectance.oindex[p,ts],dtype=float)
        if remove_all_negatives:
            imgs[imgs<0]=np.nan
        else:
            imgs[imgs==-9999]=np.nan

        means=np.nanmean(imgs,axis=(2,3))
        stds=np.nanstd(imgs,axis=(2,3))

        return means, stds

    def get_pre_post_2022_reflectance_stats(self,cloud_free=True):
        cube=self.cube
        ps=np.arange(cube.p_dim)
        all_pre_means=[]
        all_pre_stds=[]

        all_post_means=[]
        all_post_stds=[]
        for p in ps:
            if cloud_free:
                ts=cube.cloud_free_time(p)
            else:
                ts=cube.valid_time(p)


            post_mask=cube.post_2022[ts]
           
            post_ts = ts[post_mask]
            pre_ts = ts[~post_mask]
            
            pre_means, pre_stds=self.get_reflectance_stats(p,pre_ts)
            post_means, post_stds=self.get_reflectance_stats(p,post_ts)

            all_pre_means.append(pre_means)
            all_post_means.append(post_means)

            all_pre_stds.append(pre_stds)
            all_post_stds.append(post_stds)

        all_pre_means = np.concatenate(all_pre_means, axis=0)   
        all_pre_stds  = np.concatenate(all_pre_stds, axis=0)    

        all_post_means = np.concatenate(all_post_means, axis=0)
        all_post_stds  = np.concatenate(all_post_stds, axis=0)

        return all_pre_means, all_pre_stds, all_post_means, all_post_stds 

    def plot__pre_post_2022_reflectance_stats(self,cloud_free=True,title=None):
        """Compares statistics of reflectance between pre and post 2022
        """

        all_pre_means, all_pre_stds, all_post_means, all_post_stds = self.get_reflectance_stats()
        num_channels = all_pre_means.shape[1]
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes=axes.ravel()
        for c in range(num_channels):
            
            ax=axes[c]

            # Post-2022
            ax.scatter(
                all_post_means[:, c],
                all_post_stds[:, c],
                alpha=0.5,
                s=3,
                label="Post-2022",
                color="red"
            )

            # Pre-2022
            ax.scatter(
                all_pre_means[:, c],
                all_pre_stds[:, c],
                alpha=0.5,
                s=3,
                label="Pre-2022",
                color="blue"
            )

            ax.set_title(f"Channel {c}: Mean vs Std Reflectance")
            ax.set_xlabel("Mean Reflectance")
            ax.set_ylabel("Std Reflectance")
            ax.legend(loc="upper right", fontsize=8)

            pre_centroid_mean = np.nanmean(all_pre_means[:, c])
            pre_centroid_std  = np.nanmean(all_pre_stds[:, c])

            post_centroid_mean = np.nanmean(all_post_means[:, c])
            post_centroid_std  = np.nanmean(all_post_stds[:, c])

            ax.scatter(
                pre_centroid_mean,
                pre_centroid_std,
                color="blue",
                s=150,
                marker="X",
                edgecolors="white",
                linewidths=0.5,
                label="Pre-2022 Centroid"
            )

            ax.scatter(
                post_centroid_mean,
                post_centroid_std,
                color="red",
                s=150,
                marker="X",
                edgecolors="white",
                linewidths=0.5,
                label="Post-2022 Centroid"
            )

            text = (
                f"Pre:  ({pre_centroid_mean:.3f}, {pre_centroid_std:.3f})\n"
                f"Post: ({post_centroid_mean:.3f}, {post_centroid_std:.3f})"
            )

            ax.text(
                0.05, 0.95, text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
            )

        if title is None:
            date=datetime.datetime.now()
            date=date.strftime("%Y-%m-%d_%H_%M")
            title=f"mean_std_all_{date}.png"

        plt.tight_layout()
        plt.savefig(str(self.savedir/"reflectance_mean_std"/title))
        plt.close()

    """
    Comments relating to the get_outliers (etc) function below

    --------------------------------------------------------------------------------
    Purpose of this method
    --------------------------------------------------------------------------------
    - This function computes and stores "outliers" for later removal during ML training.
    - However, these "outliers" are NOT primarily removed to improve model performance.
    - Instead, they are used to mitigate TEMPORAL BIAS in the dataset.

    Notes:
    - Outliers and negative-value samples are overrepresented in pre-2022 data.
    - If not removed, models may learn to distinguish years (data acquisition artifacts)
    rather than actual ecological restoration signals.
    - Therefore:
        → Removing these samples reduces temporal leakage / bias.
        → Keeping them may increase apparent performance but for the wrong reasons.
    - Similar trade-off exists for cloud filtering:
        → stricter filtering (threshold=15): less bias
        → looser filtering (threshold=20): better performance

    --------------------------------------------------------------------------------
    What the current implementation does
    --------------------------------------------------------------------------------
    - Iterates over all polygons (p)
    - Uses only pre-2022 timesteps:
        pre_ts = ts[~cube.post_2022[ts]]
    - Computes per-image reflectance statistics:
        means, stds = get_reflectance_stats(...)
    - Applies a threshold filter:
        (mean < 1000) AND (std < 1000)
        → any channel satisfying this condition keeps the sample

    - The resulting dictionary:
        outliers[p] = list of timesteps
    is then used in `cube.remove_outliers()` to mark those samples as invalid
    (i.e. removed from training)

    --------------------------------------------------------------------------------
    Important interpretation note
    --------------------------------------------------------------------------------
    - The term "outliers" is somewhat misleading:
        → This is NOT a standard outlier detection procedure
        → It is more accurately a heuristic filter for removing potentially
        biased or temporally distinguishable samples

    - Empirically:
        → Removing these samples may NOT significantly change validation accuracy
        → But may reduce reliance on temporal artifacts

    --------------------------------------------------------------------------------
    Potential issues / caveats
    --------------------------------------------------------------------------------
    - The thresholds (mean < 1000, std < 1000) are hard-coded and not well motivated
    - The condition uses `.any(axis=1)`, meaning a sample is kept if ANY channel passes
    - This may result in:
        → weak filtering
        → or unintended inclusion/exclusion of samples

    --------------------------------------------------------------------------------
    Summary
    --------------------------------------------------------------------------------
    - This step is mainly about bias control, not data cleaning
    - It is not strictly required for achieving good performance
    - Its scientific value lies in reducing temporal shortcut learning
    """
    def get_outliers(self):
        cube=self.cube
        all_means=[]
        all_stds=[]
        which_p=[]
        which_ts=[]

        ps=np.arange(cube.p_dim)
        for p in ps:
            ts=cube.cloud_free_time(p)
            post_mask=cube.post_2022[ts]
            pre_ts = ts[~post_mask]

            which_p.append(p*np.ones(len(pre_ts)))
            which_ts.append(pre_ts)

            means, stds= self.get_reflectance_stats(p,pre_ts)
            all_means.append(means)
            all_stds.append(stds)

        which_p=np.array(np.concatenate(which_p, axis=0))
        which_ts=np.array(np.concatenate(which_ts, axis=0))
        all_means = np.array(np.concatenate(all_means, axis=0))     
        all_stds  = np.array(np.concatenate(all_stds, axis=0))  

        outliers=(all_means<1000) & (all_stds<1000)
        outliers=outliers.any(axis=1)

        all_means=all_means[outliers,:]
        all_stds=all_stds[outliers,:] 
        which_p=which_p[outliers]
        which_ts=which_ts[outliers]
        which_ps_unique=np.unique(ps)
        outliers={}

        for p in which_ps_unique:
            outliers[p]=which_ts[which_p==p]

        dir=self.savedir/"outliers"
        filename=dir/"outliers.pkl"
        with open(filename,"wb") as f:
            pickle.dump(outliers, f)

        return all_means, all_stds, which_p, which_ts

    def analyze_outliers(self,filename):
        """Loads a dictionary from filename containting p_ids and corresponding ts of outliers
        """
        
        #Loads dictionary of p ts outlier pairs
        with open(filename, "rb") as f:
            outliers = pickle.load(f)

        cube=self.cube
        gdf=cube.get_gdf()
        ps=np.array([int(p) for p in outliers.keys()])

        #Plot of each polygon as a dot to get a feel for where they are (if they are all clumped together)
        centroids=gdf.loc[ps].geometry.centroid

        fig,ax=plt.subplots()

        centroids.plot(ax=ax,markersize=5)
        save_name=self.savedir/"outliers"/"Centroid_plot.png"
        plt.savefig(save_name)
        plt.close()
        
        #Get total number of images
        no_images = sum(len(v) for v in outliers.values())
        ncols = np.ceil(np.sqrt(no_images)).astype(int)
        nrows = np.ceil(no_images / ncols).astype(int)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(32, 36))
        axes = axes.ravel()
        
        k=0
        for p in ps:
            ts=[int(t) for t in outliers[p]]
            imgs=cube.get_normalized_images(p,ts)
            for i,t in enumerate(ts):
                ax=axes[k]
                year=cube.timestamps[t].year
                title=f"Polygon {p}, Time: {cube.timestamps[t]}"
                _, ax = pf.plot_rgb(imgs[i,:,:,:],year,normalized=True,fig=fig,ax=ax,title=title)

                ax.set_xticks([])
                ax.set_yticks([])
                k += 1

        for ax in axes[k:]:
            ax.axis("off")

        plt.tight_layout()
        save_name=self.savedir/"outliers"/"plotted_outliers.png"
        plt.savefig(save_name)

    def outlier_ts(self,filename):
        with open(filename, "rb") as f:
            outliers = pickle.load(f)

        unique_ts = np.unique(list(outliers.keys()))
        print(unique_ts)
        print(len(unique_ts))
        exit()
        dates=self.cube.timestamps[unique_ts]
        print(dates)

    def plot_outliers(self):
        all_means, all_stds, ps, ts,= self.get_outliers()
        num_channels = all_means.shape[1]
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes=axes.ravel()
        for c in range(num_channels):
            ax=axes[c]
            ax.scatter(
                all_means[:, c],
                all_stds[:, c],
                alpha=0.5,
                s=10,
                label="Outliers",
                color="Blue"
            )
            ax.set_title(f"Channel {c}: Mean vs Std Reflectance")
            ax.set_xlabel("Mean Reflectance")
            ax.set_ylabel("Std Reflectance")
            ax.legend(loc="upper right", fontsize=8)

        date=datetime.datetime.now()
        date=date.strftime("%Y-%m-%d_%H_%M")
        title=f"mean_std_outliers_{date}.png"

        plt.tight_layout()
        plt.savefig(str(self.savedir/"outliers"/title))
        plt.close()

    def year_after_stats(self,line=True):
        plt.rcParams.update({'font.size': 14})
        gdf=self.cube.get_gdf()
        year_after=gdf["lastYear"]+1
        fig,ax=plt.subplots()

        min_year = year_after.min()
        max_year = year_after.max()
        bins = np.arange(min_year - 0.5, max_year + 1.5, 1)

        ax.hist(year_after,edgecolor="black",bins=bins,align="mid")
        ax.set_xlabel("the year after restoration")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of last year")
        if line:
            ax.axvline(2021.5, color="red", linestyle="--", linewidth=2)
        ax.set_xticks(np.arange(min_year, max_year + 1, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        return ax        
    
    def first_year_stats(self,line=True):
        plt.rcParams.update({'font.size': 14})
        gdf=self.cube.get_gdf()
        first_year=gdf["firstYear"]
        fig,ax=plt.subplots()

        min_year = first_year.min()
        max_year = first_year.max()
        bins = np.arange(min_year - 0.5, max_year + 1.5, 1)

        ax.hist(first_year,edgecolor="black",bins=bins,align="mid")
        ax.set_xlabel("First Year of Restoration")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of start year")
        if line:
            ax.axvline(2021.5, color="red", linestyle="--", linewidth=2)
        ax.set_xticks(np.arange(min_year, max_year + 1, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
       
        return ax 

    def restoration_duration_stats(self):
        """
        Makes a histogram of polygon restoration durations. 
        """ 
        plt.rcParams.update({'font.size': 14})
        gdf = self.cube.get_gdf()
        first_year = gdf["firstYear"]
        last_year = gdf["lastYear"]  
        diff = last_year - first_year
        
        fig,ax=plt.subplots()
        min_diff = diff.min()
        max_diff = diff.max()
        bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)

        ax.hist(diff,edgecolor="black",bins=bins,align="mid")
        ax.set_xlabel("Years")
        ax.set_ylabel("Number of polygons")
        ax.set_title("Histogram of polygon restoration durations")
        ax.set_xticks(np.arange(min_diff, max_diff + 1, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        
        return ax
    
    def restoration_status_years(self):
        plt.rcParams.update({'font.size': 14})
        gdf = self.cube.get_gdf()

        # Assumes you have firstYear and lastYear
        first_years = gdf["firstYear"].values
        last_years = gdf["lastYear"].values+1  # <-- must exist

        min_year = int(np.min(first_years))
        max_year = int(np.max(last_years))
        years = np.arange(min_year, max_year + 1)

        # Containers for counts per year
        green_counts = np.zeros_like(years, dtype=int)
        yellow_counts = np.zeros_like(years, dtype=int)
        red_counts = np.zeros_like(years, dtype=int)

        # Loop over each pasture
        for fy, ly in zip(first_years, last_years):
            pasture_years = np.arange(fy, ly + 1)

            n = len(pasture_years)
            if n == 0:
                continue

            # Split into thirds
            split1 = 1
            split2 = n-1

            for i, year in enumerate(pasture_years):
                idx = int(year - min_year)

                if i < split1:
                    red_counts[idx] += 1
                elif i < split2:
                    yellow_counts[idx] += 1
                else:
                    green_counts[idx] += 1
          
        # Plot stacked histogram (bar chart)
        fig, ax = plt.subplots()
        ax.bar(years, red_counts, color="tab:red", label="Unrestored stage")
        ax.bar(years, yellow_counts, bottom=red_counts, color="yellow", label="Middle stage")
        ax.bar(years, green_counts,bottom= red_counts+yellow_counts ,color="tab:green", label="Restored Stage")

        ax.set_xlabel("Year")
        ax.set_ylabel("Active pastures")
        ax.set_title("Restoration status over time")
        ax.set_ylim(0, max(red_counts + yellow_counts + green_counts) * 1.3)

        ax.set_xticks(years)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax.legend(fontsize=12)
        plt.tight_layout()

        return ax
    
    def number_images_stats(self):
        plt.rcParams.update({'font.size': 14})
        cube=self.cube
        fig,ax=plt.subplots()
        ps=np.arange(cube.p_dim)
        num_imgs=[]
        for p in ps:
            ts=cube.valid_time(p)
            num_imgs.append(len(ts))

        num_imgs=np.array(num_imgs)
        min_val = num_imgs.min()
        max_val = num_imgs.max()
        bins = np.arange(min_val-9, max_val+1, 40)

        ax.hist(num_imgs, bins=bins, edgecolor="black")
        ax.set_xticks(np.arange(min_val-9, max_val+1, 40))
        ax.set_xlabel("Number of images")
        ax.set_ylabel("Count")
        ax.set_title("Number of existing images per polygon")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        return ax

    def images_for_a_season(self):
        plt.rcParams.update({'font.size': 14})
        cube=self.cube
        fig,ax=plt.subplots()
        ps=np.arange(cube.p_dim)
        years=np.arange(2018,2026)
        avgs=[]
        for year in years:
            year_avg=[]
            
            for p in ps:
                ts=cube.cloud_free_time(p)
                dates=cube.timestamps[ts]
                dates=[d for d in dates if d.year == year]
                ts=cube.id_from_date(dates)
                if len(ts)==0:
                    continue
                else:
                    year_avg.append(len(ts))
            
            avg=sum(year_avg)/len(year_avg)
            avgs.append(avg)

        ax.set_title("Average Number of Images per Season", pad=15)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Image Count")
        ax.plot(years, 
                avgs,
                marker='o',
                linewidth=2,
                markersize=6
                )   

        # Grid for readability
        ax.grid(True, linestyle='--', alpha=0.6)

        # Improve ticks
        ax.set_xticks(years)

        # Tight layout prevents clipping
        plt.tight_layout()

        return ax           
    
    def pasture_stats_year(self,p,year):
        """Takes a polygon and a year and returns a dictionary with all the relevant statistics for that year for that pasture
        """
        cube=self.cube
        ts=cube.cloud_free_time(p)

        #Only extracts the year we want
        dates=cube.timestamps[ts]
        dates=[d for d in dates if d.year == year]
        ts=cube.id_from_date(dates)

        imgs=cube.get_normalized_masked_images(p,ts)

        #Get ndvi series
        mean_series=sam.ndvi_mean_series(imgs)
        std_series=sam.ndvi_std_series(imgs)
        

        #Put all data into a dict
        data={}
        data["mean_series"]=mean_series
        data["std_series"]=std_series
        data["ts"]=ts
        data["dates"]=dates
        data["polygon"]=p
        data["year"]=year

        return data
    
    def pasture_feature_vector(self,p,feature_type=1,glob_norm=False):
        """Creates a feature vector for each year for a pasture

        featuretype 1: 
            mean reflectance per band

        featuretype 2:
            mean reflectance but only first year
        """
        cube=self.cube

        if feature_type==1:
            #Create the range of dates and valid cloud free times
            first_year=cube.get_first_year(p)
            last_year=cube.get_last_year(p)+1
            start=datetime.datetime(first_year,6,1)
            end=datetime.datetime(last_year,8,31)

            ts_range=cube.range_to_indices(start,end)
            ts=cube.cloud_free_time(p,ts_range)
            dates=cube.timestamps[ts]

            if glob_norm:
                imgs=cube.get_global_normalized_masked_images(p,ts)
            else:
                imgs=cube.get_normalized_masked_images(p,ts)

            mean_series=sam.channel_means(imgs) #(T,C)
            yearly_means=[]
            years=np.arange(first_year,last_year+1)
            for year in years:
                mask = np.array([d.year == year for d in dates])
                
                # Skip entire pasture if a year has no valid data
                if not np.any(mask):
                    return None, None

                year_mean=np.mean(mean_series[mask],axis=0) #(C)
                yearly_means.append(year_mean)

            yearly_means=np.array(yearly_means)

            meta_data={}
            meta_data["polygon"]=p
            meta_data["years"]=years #(y)

            features=yearly_means #(y,C)

        if feature_type==2:

            first_ids, _ =cube.first_and_year_after_ids(p)
            
            if len(first_ids)==0:
                return None, None
            
            if glob_norm:
                imgs=cube.get_global_normalized_masked_images(p,ts)
            else:
                imgs=cube.get_normalized_masked_images(p,ts)

            mean_series=sam.channel_means(imgs)

            means=np.mean(mean_series,axis=0)
            stds=np.std(mean_series,axis=0)
            
            meta_data={}
            meta_data["polygon"]=p
            
            features = np.concatenate([means, stds], axis=0)

        if feature_type==3:
            
            #Create the range of dates and valid cloud free times
            first_year=cube.get_first_year(p)
            last_year=cube.get_last_year(p)+1
            start=datetime.datetime(first_year,6,1)
            end=datetime.datetime(last_year,8,31)

            ts_range=cube.range_to_indices(start,end)
            ts=cube.cloud_free_time(p,ts_range)
            dates=cube.timestamps[ts]

            if glob_norm:
                imgs=cube.get_global_normalized_masked_images(p,ts)
            else:
                imgs=cube.get_normalized_masked_images(p,ts)

            mean_series=sam.channel_means(imgs) #(T,C)
            yearly_means=[]
            yearly_stds=[]
            years=np.arange(first_year,last_year+1)
            for year in years:
                mask = np.array([d.year == year for d in dates])
                
                # Skip entire pasture if a year has no valid data
                if not np.any(mask):
                    return None, None

                year_mean=np.mean(mean_series[mask],axis=0) #(C)
                year_std=np.std(mean_series[mask],axis=0)

                yearly_means.append(year_mean)
                yearly_stds.append(year_std)

            yearly_means=np.array(yearly_means)
            yearly_stds=np.array(yearly_stds)

            meta_data={}
            meta_data["polygon"]=p
            meta_data["years"]=years #(y)

            features=np.concatenate([yearly_means, yearly_stds], axis=1) #(y,C)

        if feature_type == 4:

            # Create the range of dates and valid cloud free times
            first_year = cube.get_first_year(p)
            last_year = cube.get_last_year(p) + 1
            start = datetime.datetime(first_year, 6, 1)
            end = datetime.datetime(last_year, 8, 31)

            ts_range = cube.range_to_indices(start, end)
            ts = cube.cloud_free_time(p, ts_range)
            dates = cube.timestamps[ts]

            if glob_norm:
                imgs=cube.get_global_normalized_masked_images(p,ts)
            else:
                imgs=cube.get_normalized_masked_images(p,ts)

            mean_series = sam.channel_means(imgs)  # (T, C)

            yearly_features = []
            years = np.arange(first_year, last_year + 1)

            for year in years:
                year_mask = np.array([d.year == year for d in dates])

                # Skip if no data for this year
                if not np.any(year_mask):
                    return None, None

                year_dates = dates[year_mask]
                year_data = mean_series[year_mask]

                monthly_means = []
                monthly_stds = []

                for month in [6, 7, 8]:
                    month_mask = np.array([d.month == month for d in year_dates])

                    # Skip pasture if any month is missing
                    if not np.any(month_mask):
                        return None, None

                    month_data = year_data[month_mask]

                    m_mean = np.mean(month_data, axis=0)  # (C)
                    m_std = np.std(month_data, axis=0)    # (C)

                    monthly_means.append(m_mean)
                    monthly_stds.append(m_std)

                # Concatenate: means first, then stds
                year_feature = np.concatenate(monthly_means + monthly_stds, axis=0)

                yearly_features.append(year_feature)

            features = np.array(yearly_features)  # (Y, 6*C)

            meta_data = {}
            meta_data["polygon"] = p
            meta_data["years"] = years
        
        return features, meta_data

    def full_dataset_feature_vector(self,feature_type=1,glob_norm=False):
        """Returns full vector of feature vector with corresponding metadata in a pandas dataframe
        """
        cube=self.cube
        ps=np.arange(cube.p_dim)
        full_X=[]
        full_meta=[]
        for p in ps:
            X, meta= self.pasture_feature_vector(p,feature_type=feature_type,glob_norm=glob_norm)
            
            #Skip invalid pastures
            if X is None:
                continue
            
            if feature_type==1 or feature_type==3 or feature_type==4:
                years=meta["years"]
                rest_year=years[-1]
                for x, y in zip(X,years):
                    full_X.append(x)
                    full_meta.append({
                        "polygon":p,
                        "year":y,
                        "time_to_rest":y-rest_year
                    })

            if feature_type==2:
                full_X.append(X)
                full_meta.append({
                    "polygon":p
                })

        #Our full vector of feature vectors
        X=np.stack(full_X)

        #Corresponding dataset
        metadata=pd.DataFrame(full_meta)

        return X, metadata

    def PCA(self, X,metadata,filename="pca.csv"):
        scaler= StandardScaler()
        X_norm=scaler.fit_transform(X)
        assert X_norm.shape == X.shape
        
        pca = PCA(n_components=3)
        Z = pca.fit_transform(X_norm)

        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))
        
        meta_df = metadata.copy()
        meta_df["PC1"] = Z[:, 0]
        meta_df["PC2"] = Z[:, 1]
        meta_df["PC3"] = Z[:, 2]

        assert len(meta_df) == Z.shape[0]

        pca_pastures=meta_df
        pca_pastures.to_csv(GPKGS/filename)

        return pca_pastures
    
    def plot_PCA_start2d(self,pca_pastures,ax1="PC1",ax2="PC2"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            pca_pastures[ax1],
            pca_pastures[ax2],
            color="tab:green",
            alpha=0.8,
        )
        plt.tight_layout()

    def plot_PCA_start3d(self,pca_pastures):
        fig = px.scatter_3d(
            pca_pastures,
            x="PC1",
            y="PC2",
            z="PC3",
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            title="Starting year PCA (PC1–PC2–PC3)",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.write_html("pca_start_3d.html")
    
    def plot_PCA_2d_restored_vs_non_restored(self, pca_pastures,ax1="PC1",ax2="PC2"):
        plt.rcParams.update({'font.size': 14})
        last_pastures=pca_pastures[pca_pastures["time_to_rest"] == 0].copy()
        idx = pca_pastures.groupby("polygon")["year"].idxmin()
        first_year_pastures = pca_pastures.loc[idx].copy()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            first_year_pastures[ax1],
            first_year_pastures[ax2],
            s=20,
            color="tab:orange",
            alpha=0.5,
            label="Non-restored"
        )

        ax.scatter(
            last_pastures[last_pastures["PC1"]<20][ax1],
            last_pastures[last_pastures["PC1"]<20][ax2],
            s=20,
            color="tab:green",
            alpha=0.5,
            label="Restored"
        )

        ax.set_xlabel(ax1)
        ax.set_ylabel(ax2)
        ax.set_title(f"PCA: {ax1} vs {ax2}")

        ax.legend(frameon=False)

        plt.tight_layout()

    def plot_PCA_3d_restored_vs_non_restored(
        self,
        pca_pastures,
        filename="pca_restored_vs_non_restored_3d.html"
    ):

        # Last year (restored)
        restored = pca_pastures[pca_pastures["time_to_rest"] == 0].copy()

        # First year per pasture
        idx = pca_pastures.groupby("polygon")["year"].idxmin()
        non_restored = pca_pastures.loc[idx].copy()

        fig = px.scatter_3d()

        fig.add_scatter3d(
            x=non_restored["PC1"],
            y=non_restored["PC2"],
            z=non_restored["PC3"],
            mode="markers",
            marker=dict(size=4, color="darkorange", opacity=0.5),
            name="Non-restored"
        )

        fig.add_scatter3d(
            x=restored[restored["PC1"]<20]["PC1"],
            y=restored[restored["PC1"]<20]["PC2"],
            z=restored[restored["PC1"]<20]["PC3"],
            mode="markers",
            marker=dict(size=4, color="mediumseagreen", opacity=0.5),
            name="Restored"
        )

        fig.update_layout(
            title="PCA: Restored vs Non-restored (3D)",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        savedir=ut.make_savedir("PCA",path=self.savedir,additional_info="first_PCA")
        fig.write_html(savedir/filename)

    def plot_PCA_2d_centroids_by_time_to_rest(self, pca_pastures,ax1="PC1", ax2="PC2"):
        plt.rcParams.update({'font.size': 14})
        
        # Compute centroids per time_to_rest
        centroids = (
            pca_pastures[pca_pastures["time_to_rest"] > -6]
            .groupby("time_to_rest")[[ax1, ax2]]
            .mean()
            .reset_index()
            .sort_values("time_to_rest", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            centroids[ax1],
            centroids[ax2],
            color="gray",
            linewidth=2,
            alpha=0.7,
            zorder=1
        )

        # Centroid scatter
        sc = ax.scatter(
            centroids[ax1],
            centroids[ax2],
            c=centroids["time_to_rest"],
            cmap="YlGn",
            s=50,
            edgecolor="black",
            linewidth=0.8,
            zorder=2
        )
    
        # ------------------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------------------
        ax.set_xlabel(f"{ax1}")
        ax.set_ylabel(f"{ax2}")
        ax.set_title("PCA Centroid Trajectory Toward Restoration")
    
        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
        # Colorbar (kept, but simplified visually)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Years to restoration")
    
        plt.tight_layout()

    def plot_PCA_3d_centroids_by_time_to_rest(
        self,
        pca_pastures,
        filename="pca_pc1_pc2_pc3_centroid_trajectory.html"
        ):

        # Compute centroids per time_to_rest
        centroids = (
            pca_pastures[pca_pastures["time_to_rest"] > -7]
            .groupby("time_to_rest")[["PC1", "PC2", "PC3"]]
            .mean()
            .reset_index()
            .sort_values("time_to_rest", ascending=False)
        )

        fig = px.scatter_3d(
            centroids,
            x="PC1",
            y="PC2",
            z="PC3",
            color="time_to_rest",
            color_continuous_scale="YlGn",
        )

        # Connect centroids with a line (trajectory)
        fig.add_scatter3d(
            x=centroids["PC1"],
            y=centroids["PC2"],
            z=centroids["PC3"],
            mode="lines",
            line=dict(color="gray", width=4),
            showlegend=False
        )

        fig.update_traces(marker=dict(size=6, line=dict(color="black", width=0.5)))
        fig.update_layout(
            title="PCA Centroid Trajectory Toward Restoration (3D)",
            margin=dict(l=0, r=0, b=0, t=40)
        )

        savedir = ut.make_savedir(
            "PCA",
            path=self.savedir,
            additional_info="centroids_time_to_rest_trajectory")

        fig.write_html(savedir/filename)

    def pasture_stats_full_process(self,p):
        """Extracts all relevant statistics for a pasture
        """

        cube=self.cube

        #Create the range of dates and valid cloud free times
        first_year=cube.get_first_year(p)
        last_year=cube.get_last_year(p)+1
        start=datetime.datetime(first_year,6,1)
        end=datetime.datetime(last_year,8,31)

        ts_range=cube.range_to_indices(start,end)
        ts=cube.cloud_free_time(p,ts_range)
        dates=cube.timestamps[ts]

        imgs=cube.get_normalized_masked_images(p,ts)

        #Get polygon size in pixels
        mask=cube.precomputed_masks[p][0]
        size = np.count_nonzero(mask)

        #Get ndvi series
        mean_ndvi_series=sam.ndvi_mean_series(imgs)
        std_ndvi_series=sam.ndvi_std_series(imgs)

        #Get evi series
        mean_evi_series=sam.evi_mean_series(imgs)
        std_evi_series=sam.evi_std_series(imgs)

        #get ndmi series
        mean_ndmi_series=sam.ndmi_mean_series(imgs)
        std_ndmi_series=sam.ndmi_std_series(imgs)

        #Get msavi series
        mean_msavi_series=sam.msavi_mean_series(imgs)
        std_msavi_series=sam.msavi_std_series(imgs)

        #Get mean and std of each year
        year_ndvi_means=[]
        year_evi_means=[]
        year_ndmi_means=[]
        year_msavi_means=[]

        year_ndvi_stds=[]
        year_evi_stds=[]
        year_ndmi_stds=[]
        year_msavi_stds=[]

        years=np.arange(first_year,last_year+1)
        for year in years:
            mask = np.array([d.year == year for d in dates])

            year_ndvi_mean=np.mean(mean_ndvi_series[mask])
            year_evi_mean=np.mean(mean_evi_series[mask])
            year_ndmi_mean=np.mean(mean_ndmi_series[mask])
            year_msavi_mean=np.mean(mean_msavi_series[mask])

            year_ndvi_std=np.mean(std_ndvi_series[mask])
            year_evi_std=np.mean(std_evi_series[mask])
            year_ndmi_std=np.mean(std_ndmi_series[mask])
            year_msavi_std=np.mean(std_msavi_series[mask])

            year_ndvi_means.append(year_ndvi_mean)
            year_evi_means.append(year_evi_mean)
            year_ndmi_means.append(year_ndmi_mean)
            year_msavi_means.append(year_msavi_mean)

            year_ndvi_stds.append(year_ndvi_std)
            year_evi_stds.append(year_evi_std)
            year_ndmi_stds.append(year_ndmi_std)
            year_msavi_stds.append(year_msavi_std)

        year_ndvi_means=np.array(year_ndvi_means)
        year_evi_means=np.array(year_evi_means)
        year_ndmi_means=np.array(year_ndmi_means)
        year_msavi_means=np.array(year_msavi_means)

        year_ndvi_stds=np.array(year_ndvi_stds)
        year_evi_stds=np.array(year_evi_stds)
        year_ndmi_stds=np.array(year_ndmi_stds)
        year_msavi_stds=np.array(year_msavi_stds)

        #Create corresponding dict for each index
        ndvi_dict={}
        evi_dict={}
        ndmi_dict={}
        msavi_dict={}

        ndvi_dict["mean_series"]=mean_ndvi_series
        ndvi_dict["std_series"] = std_ndvi_series
        ndvi_dict["year_means"] =year_ndvi_means
        ndvi_dict["year_stds"]  =year_ndvi_stds
        ndvi_dict["start_mean"] =year_ndvi_means[0]
        ndvi_dict["end_mean"]   =year_ndvi_means[-1]
        ndvi_dict["mean_diff"]  =year_ndvi_means[-1]-year_ndvi_means[0]

        evi_dict["mean_series"]=mean_evi_series
        evi_dict["std_series"] = std_evi_series
        evi_dict["year_means"] =year_evi_means
        evi_dict["year_stds"]  =year_evi_stds
        evi_dict["start_mean"] =year_evi_means[0]
        evi_dict["end_mean"]   =year_evi_means[-1]
        evi_dict["mean_diff"]  =year_evi_means[-1]-year_evi_means[0]
        
        ndmi_dict["mean_series"]=mean_ndmi_series
        ndmi_dict["std_series"] = std_ndmi_series
        ndmi_dict["year_means"] =year_ndmi_means
        ndmi_dict["year_stds"]  =year_ndmi_stds
        ndmi_dict["start_mean"] =year_ndmi_means[0]
        ndmi_dict["end_mean"]   =year_ndmi_means[-1]
        ndmi_dict["mean_diff"]  =year_ndmi_means[-1]-year_ndmi_means[0]

        msavi_dict["mean_series"]=mean_msavi_series
        msavi_dict["std_series"] = std_msavi_series
        msavi_dict["year_means"] =year_msavi_means
        msavi_dict["year_stds"]  =year_msavi_stds
        msavi_dict["start_mean"] =year_msavi_means[0]
        msavi_dict["end_mean"]   =year_msavi_means[-1]
        msavi_dict["mean_diff"]  =year_msavi_means[-1]-year_msavi_means[0]

        #Put all data into a dict
        data={}

        data["ndvi"]=ndvi_dict
        data["evi"]=evi_dict
        data["ndmi"]=ndmi_dict
        data["msavi"]=msavi_dict

        #Full timesteps of timeseries
        data["ts"]=ts

        #Full dates of timeseries
        data["dates"]=dates

        #The polygon id
        data["polygon"]=p

        #The years of restoration
        data["years"]=years

        #First year of restoration
        data["first_year"]=first_year

        #Year where restoration was finished
        data["last_year"]=last_year

        #Lenght of restoration time
        data["restoration_time"]=len(years)

        #Pixelsize of pasture
        data["size"]=size

        return data
    
    def save_data(self,data,filename="data.pkl"):
        """Saves data on disc
        """
        dir=self.savedir/"data_files"
        savename=dir/filename
        with open(savename,"wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        """Loads data from file
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        return data

    def create_and_save_full_data(self,filename="data.pkl"):
        ps=np.arange(cube.p_dim)
        full_data=[]
        for p in ps:
            data=data_stat.pasture_stats_full_process(p)
            full_data.append(data)

        data_stat.save_data(full_data,filename=filename)

    def index_diff_vs_diff_plot(
    self,
    data_collection,
    x_index="ndvi",
    y_index="ndmi",
    show_regression=True,
    ):
        """
        Plot diff (end-start) of one index vs diff of another index
        to examine co-variation of change.
        """
        x_diffs = []
        y_diffs = []

        for data in data_collection:
            x_diffs.append(data[x_index]["mean_diff"])
            y_diffs.append(data[y_index]["mean_diff"])

        x_diffs = np.array(x_diffs)
        y_diffs = np.array(y_diffs)

        # Keep only valid points
        mask = np.isfinite(x_diffs) & np.isfinite(y_diffs)
        x_diffs = x_diffs[mask]
        y_diffs = y_diffs[mask]

        fig, ax = plt.subplots(figsize=(6.5, 6))

        # Scatter
        ax.scatter(
            x_diffs,
            y_diffs,
            color=self.colors[y_index],
            s=40,
            edgecolor="black",
            linewidth=0.4,
            label="Pastures"
        )

        # Zero reference lines
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        # Optional regression fit
        if show_regression and len(x_diffs) > 2:
            coef = np.polyfit(x_diffs, y_diffs, 1)
            x_fit = np.linspace(x_diffs.min(), x_diffs.max(), 100)
            y_fit = np.polyval(coef, x_fit)

            ax.plot(
                x_fit,
                y_fit,
                color="red",
                linewidth=2,
                label=f"Linear fit (slope = {coef[0]:.2f})"
            )

        ax.set_xlabel(f"{x_index.upper()} difference")
        ax.set_ylabel(f"{y_index.upper()} difference")
        ax.set_title(f"{y_index.upper()} Change vs {x_index.upper()} Change")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        savepath = (
            ut.make_savedir("cross_index_diffs", path=self.savedir)
            / f"{y_index}_diff_vs_{x_index}_diff.png"
        )
        plt.savefig(savepath, dpi=300)
        plt.close()

    def start_vs_diff_plot(self,data_collection,index="ndvi",color_by_rest_time=False,cmap="viridis"):
        """Plots each polygon as a data point of mean ndvi of start year vs ndvi difference
        """
        with plt.rc_context({"font.size": 14}):
            cube=self.cube
            color=self.colors[index]
            ps = np.arange(cube.p_dim)

            starts=[]
            diffs=[]
            rest_times=[]
            for p in ps:
                data=data_collection[p][index]
                start=data["start_mean"]
                diff=data["mean_diff"]
                rest_time=data_collection[p]["restoration_time"]

                rest_times.append(rest_time)
                starts.append(start)
                diffs.append(diff)

            rest_times=np.array(rest_times)
            starts=np.array(starts)
            diffs=np.array(diffs)

            # Keep only valid points. Some become inf or nan etc.
            mask = (
                np.isfinite(starts) &
                np.isfinite(diffs) &
                np.isfinite(rest_times)
            )
            starts = starts[mask]
            diffs = diffs[mask]
            rest_times = rest_times[mask]

            # Keep only valid points
            mask = np.isfinite(starts) & np.isfinite(diffs)
            starts = starts[mask]
            diffs = diffs[mask]

            # Fit a line (least squares)
            coef = np.polyfit(starts, diffs, 1)
            x_fit = np.linspace(starts.min(), starts.max(), 100)
            y_fit = np.polyval(coef, x_fit)

            plt.figure(figsize=(6, 5))
            if color_by_rest_time:
                # Scatter points
                sc=plt.scatter(
                    starts,
                    diffs,
                    c=rest_times,
                    cmap=cmap,
                    s=40,
                    edgecolor="black",
                    linewidth=0.5,
                    label="Pastures"
                )
                cbar = plt.colorbar(sc)
                cbar.set_label("Restoration time (years)")
            else:
                # Scatter points
                plt.scatter(
                    starts,
                    diffs,
                    color=color,
                    s=40,
                    edgecolor="black",
                    linewidth=0.5,
                    label="Pastures"
                )

            # Regression line
            plt.plot(
                x_fit,
                y_fit,
                color="red",
                linewidth=2,
                label=f"Linear fit (slope={coef[0]:.3f})"
            )

            plt.xlabel(f"Initial {index}")
            plt.ylabel(f"{index} change after restoration")
            plt.title(f"Initial {index} vs {index} difference")

            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc="upper right")
            plt.tight_layout()
    
    def starts_vs_ends_plot(self, data_collection, index="ndvi"):
        """ Plots start vs end with identity line
        """

        cube=self.cube
        color=self.colors[index]
        ps = np.arange(cube.p_dim)
        
        starts=[]
        ends=[]

        for p in ps:
            data=data_collection[p][index]
            start=data["start_mean"]
            end=data["end_mean"]
            starts.append(start)
            ends.append(end)

        starts=np.array(starts)
        ends=np.array(ends)

        plt.figure(figsize=(6, 5))
        # Scatter points
        plt.scatter(
                starts,
                ends,
                color=color,
                s=40,
                edgecolor="black",
                linewidth=0.5,
                label="Pastures"
            )
        
        # Identity line (y = x)
        min_val = np.nanmin([starts.min(), ends.min()])
        max_val = np.nanmax([starts.max(), ends.max()])
        
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="tab:red",
            linewidth=2,
            label="y = x"
        )

        plt.xlabel(f"Initial {index}")
        plt.ylabel(f"{index} after restoration ")
        plt.title(f"Initial {index} vs {index} After Restoration")

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        savepath = ut.make_savedir(index, path=self.savedir) / f"{index}_start_vs_ends.png"
        plt.savefig(savepath, dpi=300)
        plt.close()

    def index_diff_histograms(self, data_collection,bins=30):
        """
        Plot 4 subplots showing histograms of start to end differences
        for NDVI, EVI, NDMI, and MSAVI.
        """
        indices = ["ndvi", "evi", "ndmi", "msavi"]

        
        all_start_values = []
        for data in data_collection:
            for index in indices:
                val = data[index]["mean_diff"]
                if np.isfinite(val):
                    all_start_values.append(val)
        all_start_values = np.array(all_start_values)

        global_min = np.nanmin(all_start_values)
        global_max = np.nanmax(all_start_values)
        bins = np.linspace(global_min, global_max, bins + 1)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=False,
            sharey=False
        )
        axes = axes.ravel()

        for ax, index in zip(axes, indices):

            diffs = []

            bins = np.arange(-0.30, 0.30 + 0.02, 0.02)
            for data in data_collection:
                index_data = data[index]
                diffs.append(index_data["mean_diff"])

            diffs = np.array(diffs)
            diffs = diffs[np.isfinite(diffs)]

            color = self.colors[index]

            ax.hist(
                diffs,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="black"
            )

            # Zero reference line
            ax.axvline(
                0,
                color="black",
                linestyle="--",
                linewidth=1
            )
            
            median_val=np.nanmedian(diffs)
            ax.axvline(
                median_val,
                color="tab:red",
                linewidth=1,
                alpha=0.7,
                label=f"Median = {median_val:.3f}"
            )

            ax.set_xlim(global_min, global_max)
            ax.set_title(index.upper())
            ax.set_xlabel(f"Difference in {index} before and after restoration")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)

            ax.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("Distribution of Vegetation Index Changes", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        savepath = ut.make_savedir("histograms", path=self.savedir) / "index_diff_histograms.png"
        plt.savefig(savepath, dpi=300)
        plt.close()

    def index_start_histograms(self, data_collection,bins=30):
        """
        Plot 4 subplots showing histograms of start to end differences
        for NDVI, EVI, NDMI, and MSAVI.
        """
        indices = ["ndvi", "evi", "ndmi", "msavi"]
        all_start_values = []
        for data in data_collection:
            for index in indices:
                val = data[index]["start_mean"]
                if np.isfinite(val):
                    all_start_values.append(val)

        all_start_values = np.array(all_start_values)
        global_min = np.nanmin(all_start_values)
        global_max = np.nanmax(all_start_values)
        bins = np.linspace(global_min, global_max, bins + 1)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=False,
            sharey=False
        )
        axes = axes.ravel()

        for ax, index in zip(axes, indices):
            start = []

            #bins = np.arange(-0.30, 0.30 + 0.02, 0.02)
            for data in data_collection:
                index_data = data[index]
                start.append(index_data["start_mean"])

            start = np.array(start)
            start = start[np.isfinite(start)]

            color = self.colors[index]

            ax.hist(
                start,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="black"
            )
            
            median_val=np.nanmedian(start)
            ax.axvline(
                median_val,
                color="tab:red",
                linewidth=1,
                alpha=0.7,
                label=f"Median = {median_val:.3f}"
            )

            ax.set_xlim(global_min, global_max)
            ax.set_title(index.upper())
            ax.set_xlabel(f"{index} at start of restoration")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)

            ax.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("Distribution of Vegetation Index starts", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        savepath = ut.make_savedir("histograms", path=self.savedir) / "index_start_histograms.png"
        plt.savefig(savepath, dpi=300)
        plt.close()

    def index_end_histograms(self, data_collection,bins=30):
        """
        Plot 4 subplots showing histograms end value
        for NDVI, EVI, NDMI, and MSAVI.
        """
        indices = ["ndvi", "evi", "ndmi", "msavi"]
        all_end_values = []
        for data in data_collection:
            for index in indices:
                val = data[index]["end_mean"]
                if np.isfinite(val):
                    all_end_values.append(val)
        all_end_values = np.array(all_end_values)

        global_min = np.nanmin(all_end_values)
        global_max = np.nanmax(all_end_values)
        bins = np.linspace(global_min, global_max, bins + 1)

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 8),
            sharex=False,
            sharey=False
        )
        axes = axes.ravel()

        for ax, index in zip(axes, indices):
            start = []

            #bins = np.arange(-0.30, 0.30 + 0.02, 0.02)
            for data in data_collection:
                index_data = data[index]
                start.append(index_data["end_mean"])

            start = np.array(start)
            start = start[np.isfinite(start)]
            color = self.colors[index]
            ax.hist(
                start,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="black"
            )
            
            median_val=np.nanmedian(start)
            ax.axvline(
                median_val,
                color="tab:red",
                linewidth=1,
                alpha=0.7,
                label=f"Median = {median_val:.3f}"
            )
            ax.set_xlim(global_min, global_max)
            ax.set_title(index.upper())
            ax.set_xlabel(f"{index} at end of restoration")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.suptitle("Distribution of Vegetation Index ends", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        savepath = ut.make_savedir("histograms", path=self.savedir) / "index_end_histograms.png"
        plt.savefig(savepath, dpi=300)
        plt.close()

    def index_start_vs_end_histogram(self, data_collection, index, bins=30):
        """
        Plot start vs end histograms side-by-side for a given index
        with identical bins and axis ranges.

        Parameters
        ----------
        data_collection : list
        index : str                # e.g. "ndvi", "evi", "ndmi", "msavi"
        bins : int
        """

        # ---- Collect values ----
        start_vals = []
        end_vals = []
        with plt.rc_context({"font.size": 14}):
            for data in data_collection:
                s = data[index]["start_mean"]
                e = data[index]["end_mean"]

                if np.isfinite(s):
                    start_vals.append(s)
                if np.isfinite(e):
                    end_vals.append(e)

            start_vals = np.array(start_vals)
            end_vals = np.array(end_vals)

            # ---- Shared binning ----
            combined = np.concatenate([start_vals, end_vals])
            global_min = np.nanmin(combined)
            global_max = np.nanmax(combined)
            bins = np.linspace(global_min, global_max, bins + 1)

            # ---- Plot ----
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
            color = self.colors[index]

            # Start
            axes[0].hist(
                start_vals,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="black"
            )
            median_start = np.nanmedian(start_vals)
            axes[0].axvline(
                median_start,
                color="tab:red",
                linewidth=1,
                label=f"Median = {median_start:.3f}"
            )
            axes[0].set_title(f"{index.upper()} (Start)")
            axes[0].set_xlabel(index)
            axes[0].set_ylabel("Count")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.4)

            # End
            axes[1].hist(
                end_vals,
                bins=bins,
                color=color,
                alpha=0.8,
                edgecolor="black"
            )
            median_end = np.nanmedian(end_vals)
            axes[1].axvline(
                median_end,
                color="tab:red",
                linewidth=1,
                label=f"Median = {median_end:.3f}"
            )
            axes[1].set_title(f"{index.upper()} (End)")
            axes[1].set_xlabel(index)
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.4)

            # Explicitly enforce identical limits (extra safety)
            axes[0].set_xlim(global_min, global_max)
            axes[1].set_xlim(global_min, global_max)

            plt.suptitle(f"{index.upper()} Start vs End Distribution", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

    def start_end_mean_std(
    self,
    data_collection,
    index="ndvi",
    ):
        """
        Plot diff (end-start) of one index vs diff of another index
        to examine co-variation of change.
        """
        mean_starts = []
        mean_ends = []
        std_starts = []
        std_ends = []
        for data in data_collection:
            mean_starts.append(data[index]["year_means"][0])
            mean_ends.append(data[index]["year_means"][-1])
            std_starts.append(data[index]["year_stds"][0])
            std_ends.append(data[index]["year_stds"][-1])

        mean_starts = np.array(mean_starts)
        mean_ends = np.array(mean_ends)
        std_starts = np.array(std_starts)
        std_ends = np.array(std_ends)

        # Keep only valid points
        mask = np.isfinite(mean_starts) & np.isfinite(mean_ends) & np.isfinite(std_ends) & np.isfinite(std_ends)
        mean_starts = mean_starts[mask]
        mean_ends = mean_ends[mask]
        std_starts = std_starts[mask]
        std_ends = std_ends[mask]

        fig, ax = plt.subplots(figsize=(6.5, 6))

        # Scatter
        ax.scatter(
            mean_ends,
            std_ends,
            color="tab:green",
            s=10,
            alpha=0.6,
            linewidth=0.4,
            label="Pastures"
        )

        # Scatter
        ax.scatter(
            mean_starts,
            std_starts,
            color="tab:orange",
            s=10,
            alpha=0.6,
            linewidth=0.4,
            label="Pastures"
        )

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()    

    def plot_pasture_stats_year(self, data):
        """Used originally to plo the timeseries of ndvi for a year
        """
        ndvi_data=data["ndvi"]
        mean_series=ndvi_data["mean_series"]
        std_series=ndvi_data["std_series"]
        ts=data["ts"]
        dates=data["dates"]
        year=data["year"]
        p=data["polygon"]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 6),
            sharex=True
        )

        axes[0].plot(
            dates,
            mean_series,
            color="tab:green",
            marker="o",
            linewidth=2,
            markersize=5
        )

        axes[0].set_title(
            f"Mean NDVI for polygon {p} at year {year}",
            fontsize=12
        )

        axes[0].set_ylabel("Mean NDVI")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(
            dates,
            std_series,
            color="tab:green",
            marker="o",
            linewidth=2,
            markersize=5
        )

        axes[1].set_title(
            f"STD NDVI for polygon {p} at year {year}",
            fontsize=12
        )

        axes[1].set_ylabel("STD NDVI")
        axes[1].grid(True, alpha=0.3)

        year_int = int(year)
        month_ticks = [
            datetime.datetime(year_int, 6, 1),
            datetime.datetime(year_int, 7, 1),
            datetime.datetime(year_int, 8, 1),
        ]

        for ax in axes:
            ax.set_xticks(month_ticks)
            ax.set_xticklabels(["June", "July", "August"])

        axes[0].tick_params(axis="x", labelbottom=True)
        plt.tight_layout()
        savepath=ut.make_savedir("ndvi",path=self.savedir)/f"ndvi_polygon_{p}_year_{year}.png"
        plt.savefig(savepath)
        plt.close()

    def plot_pasture_statistics(self,data):
        """Takes data from a full restoration process of a pasture and plots the data
        """
        savedir=ut.make_savedir("full_stats",path=self.savedir)

        #NDVI
        ndvi_data=data["ndvi"]        
        mean_series=ndvi_data["mean_series"]
        std_series=ndvi_data["std_series"]
        ts=data["ts"]
        dates=data["dates"]
        p=data["polygon"]

        SEASON_LENGTH = 92  # June 1 Aug 31

        season_x = []
        season_years = []
        for d in dates:
            season_start = datetime.date(d.year, 6, 1)
            days_into_season = (d - season_start).days
            season_x.append(days_into_season)
            season_years.append(d.year)
        season_x = np.array(season_x)

        unique_years = sorted(set(season_years))
        year_offset = {y: i * SEASON_LENGTH for i, y in enumerate(unique_years)}
        x = np.array([
            season_x[i] + year_offset[season_years[i]]
            for i in range(len(dates))
        ])

        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 6),
            sharex=True
        )

        axes[0].plot(
            x,
            mean_series,
            color="tab:green",
            marker="o",
            linewidth=2,
            markersize=5
        )

        axes[0].set_title(
            f"Mean NDVI for polygon {p}",
            fontsize=12
        )

        axes[0].set_ylabel("Mean NDVI")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(
            x,
            std_series,
            color="tab:green",
            marker="o",
            linewidth=2,
            markersize=5
        )

        axes[1].set_title(
            f"STD NDVI for polygon {p}",
            fontsize=12
        )

        axes[1].set_ylabel("STD NDVI")
        axes[1].grid(True, alpha=0.3)
    
        for ax in axes:
            for i, y in enumerate(unique_years):
                ax.axvline(
                    i * SEASON_LENGTH,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.3
                )

        major_ticks = []
        major_labels = []
        minor_ticks = []
        minor_labels = []
        for i, y in enumerate(unique_years):
            base = i * SEASON_LENGTH
        
            # Major tick: June
            major_ticks.append(base)
            major_labels.append(f"Jun {y}")
        
            # Minor ticks: July & August
            minor_ticks.append(base + 30)
            minor_labels.append("Jul")
        
            minor_ticks.append(base + 61)
            minor_labels.append("Aug")

        for ax in axes:
            ax.set_xticks(major_ticks)
            ax.set_xticklabels(major_labels)

            ax.set_xticks(minor_ticks, minor=True)
            ax.tick_params(axis="x", which="minor", length=4, color="0.5")

        axes[0].tick_params(axis="x", labelbottom=True)    

        plt.tight_layout()
        ndvi_dir = savedir / "ndvi"
        ndvi_dir.mkdir(parents=True, exist_ok=True)

        savepath=ndvi_dir/f"ndvi_polygon_{p}.png"
        plt.savefig(savepath)
        plt.close()

    def build_seasonal_trendline(self, dates, values, season_start_month=6):
        """
        For each year, fit NDVI ~ day_in_season and return piecewise trendlines.

        Returns
        -------
        x_all : np.ndarray
            Concatenated x values (season-offset time axis)
        y_all : np.ndarray
            Corresponding trendline NDVI values
        years : list
            Years included (ordered)
        """
        SEASON_LENGTH = 92

        # group indices by year
        years = sorted(set(d.year for d in dates))

        x_all = []
        y_all = []

        for i, y in enumerate(years):
            idx = [j for j, d in enumerate(dates) if d.year == y]
            if len(idx) < 3:
                continue  # skip unstable fits

            dts = [dates[j] for j in idx]
            vals = np.array([values[j] for j in idx])

            season_start = datetime.date(y, season_start_month, 1)
            x_season = np.array([(d - season_start).days for d in dts])

            # linear fit
            a, b = np.polyfit(x_season, vals, deg=1)

            # evaluate on full season grid
            xs = np.arange(SEASON_LENGTH)
            ys = a * xs + b

            x_global = xs + i * SEASON_LENGTH

            x_all.append(x_global)
            y_all.append(ys)

        if not x_all:
            return None, None, None

        return np.concatenate(x_all), np.concatenate(y_all), years

    def plot_polygon_trend_only(self, data):
        ndvi = data["ndvi"]["mean_series"]
        dates = data["dates"]
        p = data["polygon"]

        x, y, years = self.build_seasonal_trendline(dates, ndvi)
        if x is None:
            return

        plt.figure(figsize=(10, 3))
        plt.plot(x, y, linewidth=2)
        plt.title(f"Seasonal NDVI trendlines – polygon {p}")
        plt.xlabel("Seasonal time (Jun–Aug stacked by year)")
        plt.ylabel("NDVI")
        plt.grid(alpha=0.3)
        out = self.savedir / "ndvi_single_trial.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    def plot_all_polygons_trendlines(self, all_data, savedir):
        """
        all_data : iterable of polygon data dicts
        """

        SEASON_LENGTH = 92
        plt.figure(figsize=(12, 5))
        max_years = 0
        for data in all_data:
            ndvi = data["ndvi"]["mean_series"]
            dates = data["dates"]

            x, y, years = self.build_seasonal_trendline(dates, ndvi)
            if x is None:
                continue

            max_years = max(max_years, len(years))

            plt.plot(
                x,
                y,
                color="tab:green",
                alpha=0.15,
                linewidth=1
            )

        # Year separators
        for i in range(max_years):
            plt.axvline(
                i * SEASON_LENGTH,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.1
            )

        # X ticks
        xticks = [i * SEASON_LENGTH for i in range(max_years)]
        plt.xticks(xticks, [f"Jun Y{i+1}" for i in range(max_years)])

        plt.title("Seasonal NDVI trendlines across polygons")
        plt.ylabel("NDVI (trend only)")
        plt.xlabel("Seasonal time (Jun–Aug stacked)")
        plt.grid(alpha=0.3)

        out = savedir / "ndvi_trendline_overlay.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    def plot_global_seasonal_trendlines(self, all_data):
        """
        Plots one global NDVI trendline per polygon, fitted across the full
        stacked Jun–Jul–Aug time series. Only trendlines are plotted (no dots),
        overlaid with low alpha to visualize common trends.

        Parameters
        ----------
        all_data : iterable
            Iterable of per-polygon data dicts with keys:
            - "ndvi"]["mean_series"]
            - "dates"
        """

        savedir = ut.make_savedir("global_trends", path=self.savedir)
        SEASON_LENGTH = 92  # Jun 1 – Aug 31

        # ---------------------------------------------------------
        # Determine global year range (for consistent x-axis)
        # ---------------------------------------------------------
        all_years = []
        for data in all_data:
            all_years.extend(d.year for d in data["dates"])

        unique_years = sorted(set(all_years))
        year_to_index = {y: i for i, y in enumerate(unique_years)}

        # ---------------------------------------------------------
        # Prepare plot
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5))

        # ---------------------------------------------------------
        # Plot one global trendline per polygon
        # ---------------------------------------------------------
        for data in all_data:
            dates = data["dates"]
            y = np.asarray(data["ndvi"]["std_series"])

            # Build stacked seasonal x-axis
            season_x = []
            season_years = []

            for d in dates:
                season_start = datetime.date(d.year, 6, 1)
                season_x.append((d - season_start).days)
                season_years.append(d.year)

            x = np.array([
                season_x[i] + year_to_index[season_years[i]] * SEASON_LENGTH
                for i in range(len(dates))
            ])

            # Require enough points for a stable fit
            if len(x) < 10:
                continue

            # Global linear trend across all seasons
            a, b = np.polyfit(x, y, deg=1)

            # Evaluate over this polygon's valid time span only
            x_fit = np.arange(x.min(), x.max() + 1)
            y_fit = a * x_fit + b

            ax.plot(
                x_fit,
                y_fit,
                color="tab:green",
                alpha=0.15,
                linewidth=1.5
            )

        # ---------------------------------------------------------
        # Year separators
        # ---------------------------------------------------------
        for i in range(len(unique_years)):
            ax.axvline(
                i * SEASON_LENGTH,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.3
            )

        # ---------------------------------------------------------
        # X-axis ticks and labels (same as original)
        # ---------------------------------------------------------
        major_ticks = []
        major_labels = []
        minor_ticks = []

        for i, y in enumerate(unique_years):
            base = i * SEASON_LENGTH

            # Major tick: June
            major_ticks.append(base)
            major_labels.append(f"Jun {y}")

            # Minor ticks: July & August
            minor_ticks.append(base + 30)
            minor_ticks.append(base + 61)

        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels)

        ax.set_xticks(minor_ticks, minor=True)
        ax.tick_params(axis="x", which="minor", length=4, color="0.5")

        # ---------------------------------------------------------
        # Styling
        # ---------------------------------------------------------
        ax.set_title("Global NDVI trendlines across polygons (Jun–Aug stacked)")
        ax.set_ylabel("NDVI (global trend only)")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        out = savedir / "ndvi_global_trendlines.png"
        plt.savefig(out)
        plt.close()

if __name__=="__main__":
    cube=zc.Zarr(ZARR,GPKGS/"final02-20.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
        
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    data_stat=data_statistics(cube,dataname="data_final.pkl")
    data_stat.restoration_status_years()
    plt.savefig(REPORT_FIGS/"histograms"/"restoration_status.png")

    #pca_pastures = pd.read_csv(GPKGS/"pca_05_18.csv")
    #ax1="PC1"
    #ax2="PC3"
    ##data_stat.plot_PCA_2d_restored_vs_non_restored(pca_pastures,ax1=ax1,ax2=ax2)
    #data_stat.plot_PCA_2d_centroids_by_time_to_rest(pca_pastures,ax1=ax1,ax2=ax2)
    #plt.savefig(REPORT_FIGS/"PCA"/f"centroids_trajectory_plot_{ax1}_{ax2}.png")
    #plt.close()
#

    #X, metadata=data_stat.full_dataset_feature_vector(feature_type=4,glob_norm=True)
    #data_stat.PCA(X,metadata,filename="pca_glob_05_18.csv")
    #data_stat.create_and_save_full_data(filename="data_final.pkl")
    #ax=data_stat.restoration_status_years()
    #plt.savefig(data_stat.savedir/"histograms"/"Restoration_color2.png")

   # data_stat.plot_PCA_3d_centroids_by_time_to_rest(pca_pastures)
    #data_stat.plot_PCA_start3d(pca_pastures)
    #plt.savefig(data_stat.savedir/"PCA"/"new_cluster_test3d.png")

    #data_stat.plot_global_seasonal_trendlines(data_stat.data_collection)

    #
    #seq_diff=data_stat.seq_len_diff()
    #data_stat.plot_seq_len_hist(seq_diff,title="sequence_length_difference_hist_new_20.png")
#
    #cube=zc.Zarr(ZARR,GPKGS/"final02-20.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_15_15_10.npy")
    #data_stat=data_statistics(cube)
    #seq_diff=data_stat.seq_len_diff()
    #data_stat.plot_seq_len_hist(seq_diff,title="sequence_length_difference_hist_new_15.png")
#
    #cube=zc.Zarr(ZARR,GPKGS/"final02-20.gpkg")
    #data_stat=data_statistics(cube)
    #seq_diff=data_stat.seq_len_diff()
    #data_stat.plot_seq_len_hist(seq_diff,title="sequence_length_difference_hist_old.png")
%
    ##data_stat.plot_seq_len_hist(data_stat.seq_len_diff())
    ##data_collection=data_stat.data_collection
    #
    ##filename=data_stat.savedir/"outliers"/"outliers.pkl"
    ##data_stat.start_vs_diff_plot()
    ##X,metadata=data_stat.full_dataset_feature_vector()
    ##pca_pastures=data_stat.PCA(X,metadata)
    ##pca_pastures = pd.read_csv(GPKGS/"pca04_20.csv")
    ##data_stat.plot_PCA_2d_centroids_by_time_to_rest(pca_pastures)
    #data_stat=data_statistics(cube,dataname="data_20.pkl")
    #data_collection=data_stat.data_collection
#
    ##data_stat.start_vs_diff_plot()
    #indices=["ndvi","evi","ndmi","msavi"]
    #data_stat.index_end_histograms(data_collection)
    #data_stat.index_start_histograms(data_collection)
    #data_stat.index_diff_histograms(data_collection)
    #for i in range(0,4):
    #    data_stat.start_vs_diff_plot(data_collection,index=indices[i])
    #    data_stat.starts_vs_ends_plot(data_collection,index=indices[i])
    #    
    #    
    #    for k in range(i+1,4):
    #        data_stat.index_diff_vs_diff_plot(data_collection,x_index=indices[i],y_index=indices[k])
 
    #gdf=cube.get_gdf()
    #print(len(gdf[gdf["calc_area"]<5000])/len(gdf))
    #data_stat.area_statistics(title="Number of pixels for each polygon for polygons with less than 500 pixels")
    # ax=data_stat.year_after_stats()

    # plt.savefig(data_stat.savedir/"year_stats"/"Year_after_histogram.png")
    #data_stat.reflectance_stats(cloud_free=True)

    # Just Linnea down here, don't mind me
    #savedir = ut.make_savedir("linnea_testar_grejer_runs", additional_info="Duration hist")
    #ax = data_stat.restoration_duration_stats() 
    #plt.savefig(savedir/"Histogram")