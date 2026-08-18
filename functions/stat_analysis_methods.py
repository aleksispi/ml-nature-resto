import numpy as np
import pandas as pd
import os
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import geopy.distance


def calculate_ndvi(im):
    red = im[3,:,:] 
    nir = im[7,:,:]
    return (nir - red) / (nir + red)

def calculate_ndvi_series(ims):
    """Calculates ndvi for time series of images
    """
    red = ims[:,3,:,:] 
    nir = ims[:,7,:,:]
    return (nir - red) / (nir + red+ 1e-8)


def calculate_evi_series(ims):
    """Calculates evi for time series of images
    """
    red = ims[:,3,:,:] 
    blue= ims[:,1,:,:]
    nir = ims[:,7,:,:]
    return 2.5*(nir - red) / (nir + 6*red-7.5*blue+1+ 1e-8)


def calculate_ndmi_series(ims):
    """Calculates ndmi for time series of images
    """
    swir = ims[:,10,:,:] 
    nir = ims[:,7,:,:]
    return (nir - swir) / (nir + swir+ 1e-8)

def calculate_msavi_series(ims):
    """Calculates ndvi for time series of images
    """
    red = ims[:,3,:,:] 
    nir = ims[:,7,:,:]
    
    #Needed to ensure nan always stays nan
    term = (2*nir + 1)**2 - 8*(nir - red)
    term = np.maximum(term, 0.0)

    return (2*nir +1-np.sqrt(term)) / (2+ 1e-8)



def ndvi_median_series(ims):
    """Takes a time series and takes the median, returning a time-series
    of single nvdi values 
    """
    nvdis=calculate_ndvi_series(ims)
    median=np.nanmedian(nvdis,axis=(1,2))
    return median

def ndvi_mean_series(ims):
    """Takes a time series and takes the mean, returning a time-series
    of single nvdi values 
    """
    nvdis=calculate_ndvi_series(ims)
    mean=np.nanmean(nvdis,axis=(1,2))
    return mean

def ndvi_std_series(ims):
    """Takes a time series and takes the mean spatially, returning a time-series
    of single nvdi values 
    """
    nvdis=calculate_ndvi_series(ims)
    std=np.nanstd(nvdis,axis=(1,2))
    return std

def ndvi_max_series(ims):
    """Takes a time series and takes the max, returning a time-series
    of single nvdi values 
    """
    nvdis=calculate_ndvi_series(ims)
    max=np.nanmax(nvdis,axis=(1,2))
    return max

def evi_median_series(ims):
    """Takes a time series and takes the median, returning a time-series
    of single evi values 
    """
    evis=calculate_evi_series(ims)
    median=np.nanmedian(evis,axis=(1,2))
    return median

def evi_mean_series(ims):
    """Takes a time series and takes the mean, returning a time-series
    of single evi values 
    """
    evis=calculate_evi_series(ims)
    mean=np.nanmean(evis,axis=(1,2))
    return mean

def evi_std_series(ims):
    """Takes a time series and takes the std, returning a time-series
    of single evi values 
    """
    evis=calculate_evi_series(ims)
    std=np.nanstd(evis,axis=(1,2))
    return std

def evi_max_series(ims):
    """Takes a time series and takes the max, returning a time-series
    of single evi values 
    """
    evis=calculate_evi_series(ims)
    max=np.nanmax(evis,axis=(1,2))
    return max

def ndmi_median_series(ims):
    """Takes a time ndmries and takes the median, returning a time-series
    of single ndmi values 
    """
    ndmis=calculate_ndmi_series(ims)
    median=np.nanmedian(ndmis,axis=(1,2))
    return median

def ndmi_mean_series(ims):
    """Takes a time ndmries and takes the mean, returning a time-series
    of single ndmi values 
    """
    ndmis=calculate_ndmi_series(ims)
    mean=np.nanmean(ndmis,axis=(1,2))
    return mean

def ndmi_std_series(ims):
    """Takes a time ndmries and takes the std, returning a time-series
    of single ndmi values 
    """
    ndmis=calculate_ndmi_series(ims)
    std=np.nanstd(ndmis,axis=(1,2))
    return std

def ndmi_max_series(ims):
    """Takes a time ndmries and takes the max, returning a time-series
    of single ndmi values 
    """
    ndmis=calculate_ndmi_series(ims)
    max=np.nanmax(ndmis,axis=(1,2))
    return max

def msavi_median_series(ims):
    """Takes a time series and takes the median, returning a time-series
    of single msavi values 
    """
    msavis=calculate_msavi_series(ims)
    median=np.nanmedian(msavis,axis=(1,2))
    return median

def msavi_mean_series(ims):
    """Takes a time series and takes the mean, returning a time-series
    of single msavi values 
    """
    msavis=calculate_msavi_series(ims)
    mean=np.nanmean(msavis,axis=(1,2))
    return mean

def msavi_std_series(ims):
    """Takes a time series and takes the std, returning a time-series
    of single msavi values 
    """
    msavis=calculate_msavi_series(ims)
    std=np.nanstd(msavis,axis=(1,2))
    return std

def msavi_max_series(ims):
    """Takes a time series and takes the max, returning a time-series
    of single msavi values 
    """
    msavis=calculate_msavi_series(ims)
    max=np.nanmax(msavis,axis=(1,2))
    return max



def calculate_ndwi(im):
    swir = im[10,:,:]
    nir = im[7,:,:]
    return (nir - swir)/(nir + swir)

def channel_means(ims):
    """Takes a time series and takes spatianl mean, returning a time-series of single
    mean values
    """
    return np.nanmean(ims,axis=(2,3))

# def median_of_median_thingy(im1,im2):
#     # Calculate NDVI for im1 and im2
#     ndvi1 = calculate_ndvi(im1)
#     ndvi2 = calculate_ndvi(im2)
#     # Calculate "median of median thingy"
#     median1 = np.median(ndvi1,axis=0) # Median over first dim (height)
#     median1 = np.median(median1,axis=0) # Median over second dim
#     median2 = np.median(ndvi2,axis=0)
#     median2 = np.median(median2,axis=0)

#     # Now the medians should be scalars 
#     print("Median of median thingy:")
#     print("Median in image 1: ", str(median1))
#     print("Median in image 2: ", str(median2))
#     print("Difference: ", str(np.abs(median1-median2)))

# def median_of_flattened(im1,im2):
#     # Just checking if it makes a difference to first flatten the images (default if no axis is specified in np.median)
#     # Calculate NDVI for im1 and im2, which are assumed to be RGB images
#     ndvi1 = calculate_ndvi(im1)
#     ndvi2 = calculate_ndvi(im2)
#     median1 = np.median(ndvi1)
#     median2 = np.median(ndvi2)
#     print("Median of flattened image:")
#     print("Median in image 1: ", str(median1))
#     print("Median in image 2: ", str(median2))
#     print("Difference: ", str(np.abs(median1-median2)))

def create_correlation_matrix(im_list):
    print("Constructing correlation matrix ...")
    features = []
    for im in im_list:
        # im is one satellite image with C different channels
        im = im.astype(np.float32) / 10000.0      # Scaling
        medians_of_bands = np.median(im,axis=0)
        medians_of_bands = np.median(medians_of_bands,axis=0) # Should now be of shape 1XC
        # Adding other vegetation indices
        im_ndvi = calculate_ndvi(im)
        im_ndwi = calculate_ndwi(im)
        im_fts = np.append(medians_of_bands,np.median(np.median(im_ndvi,axis=0),axis=0))
        im_fts = np.append(im_fts,np.median(np.median(im_ndwi,axis=0),axis=0))
        features.append(im_fts)
  

    features = np.array(features)
    features_df = pd.DataFrame(features, columns=['B01 (Aerosol)', 'B02 (Blue)', 'B03 (Green)','B04(Red)','B05 (Red edge)','B06','B07','B08 (NIR)', 'B8A', 'B09', 'B11','B12','NDVI','NDWI'])  # Note that there is no band 10
    # Constructing a correlation matrix
    return features_df.corr()
