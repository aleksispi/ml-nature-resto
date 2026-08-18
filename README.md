#  Monitoring Pasture Restoration from Satellite Image Time Series: Caveats and Opportunities

Official code repository for the [CV4Ecology](https://cv4e-workshop.github.io/) (ECCV 2026 workshop) paper [Monitoring Pasture Restoration from Satellite Image Time Series: Caveats and Opportunities](INSERT LINK). The work was conducted by [RISE Research Institutes of Sweden](https://www.ri.se/en), [Lund University](https://www.lunduniversity.lu.se/) and the [Swedish Board of Agriculture (SBA)](https://jordbruksverket.se/languages/english/swedish-board-of-agriculture), with funding from the [Swedish National Space Agency (SNSA)](https://www.rymdstyrelsen.se/en/), project number 2025-00013. Code mainly developed by Linnea Sartorius and Isak Randahl. We thank the SBA, and [Niklas Boke Olen](https://scholar.google.com/citations?user=aUvgSgoAAAAJ&hl=en) in particular, for their valuable contributions, including key parts of the data used for this work.

## NEWS
* Paper accepted (in proceedings) at the [3rd Workshop on Computer Vision for Ecology](https://cv4e-workshop.github.io/) at [ECCV 2026](https://eccv.ecva.net/)!
* Our work got presented at the [Swedish Climate Symposium 2026](https://swedishclimatesymposium.com/) in Lund!

## Data
The polygons and associated labels in this project were obtained from the SBA -- please contact their GIS department (Gis.Support@jordbruksverket.se) if you want to ask for access to their polygons and labels, including the ones we used in this project. However, satellite data can be freely downloaded from the [_Digital Earth Sweden_](https://digitalearth.se/) platform using scripts provided within this code repository, where you could of course specify regions of interest with polygons of your own.

## Model weights
Pretrained model weights (for both the best-performing CNN and CNN-LSTM models) can be downloaded [here](https://drive.google.com/drive/folders/1RpOVOphp9IcDONkIgssxORzfYim8O76l?usp=sharing).

## Contents of this repository
- File overview
  - explains existing files, which ones are more important as well as necessary details
- Data preprocessing
  - Contains instructions on basic steps of going from directory of nc-files + a gpkg file to a ready dataset
- ML pipeline
  - Contains instructions on how to use our ML pipeline
- File descriptions
  - Contains more detailed info about specific files
- Data-file descriptions
  - Many scripts and methods need a path to specific data. This section is to clear that up slightly

## File overview

### Overview
Brief overview of the main files. These uses are mentioned later in File descriptions. Note that `zarr_class.py` is the most important file if you want to know how to use the dataset, whereas `load_data.py` (in folder `ml_scripts`) is the most important for using the data in ML.
- **classes**
  - `zarr_class.py` (in folder `classes`): main wrapper class for the zarr cube. Most of the functionality needed for the dataset lives here
  - `gdf_class.py` (in folder `classes`): class for filtering the gdf
  - `zarr_dataset_class.py` (in folder `classes`): Used in the ML pipeline to extract samples without having to load all into memory
  - `data_statistics.py` (in folder `classes`): Class for analyzing and visualizing the dataset. Also contains functionality for generating auxiliary data products used
  in ML, such as: (i) outlier definitions (`outliers.pkl`); (ii) statistics used for bias filtering. **Note:** Some of these outputs are used during training (e.g. in
  `cube.remove_outliers`) but are not part of the core preprocessing pipeline. They are mainly intended to reduce temporal bias rather than improve rawperformance.
- **ml_scripts**
  - `ml_main.py` (in folder `ml_scripts`): Main class for running training and testing.
  - `ml_main_timeseries` (in folder `ml_scripts`): CNN+LSTM version of `ml_main`.
  - `load_data.py` (in folder `ml_scripts`): Class where everything in sample construction, dataset split, and dataloader creation is handled.
  - All other files in the folder `ml_scripts` are just basic pytorch architecture.
- **functions**
  - `gdf_utils.py` (in folder `functions`): a collection of methods on gdfs. Most of the functionallity used in `gdf_class` lives here or in `gdf_tests.py`
  - `cloud_filter.py` (in folder `functions`): function for applying cloud filter on the timeseries of a polygon, and then returning the mask and filtered timeseries.
  Contains commented out code for visualization. **Note:** Different cloud thresholds introduce a trade-off:
  stricter filtering (e.g. 15) --> lower bias | looser filtering (e.g. 20) --> higher performance
- **main_scripts**
  - `zarr_creation.py` (in folder `main_scripts`): Script for creating the Zarr cube from our format of gdf. Can be used as a blueprint of zarr creation or can be generalized with a few changes.

## Data preprocessing 

### Overview
```
Satellite images (.nc)          Polygon GDF (.gpkg)
            │                            │
            │                GDF preprocessing class (Step 1)
            │                            │
            │                Stored GPKG (filtered GDF)
            │               (kept for reuse / metadata)
            │                            │
            └───────────────┬────────────┘
                            │
                 Zarr creation (Step 2)
                            │
               Zarr wrapper class (Step 3)
                            │
                Create cloud mask (Step 4)
                            │
                Bias filtering (Step 5)
                            │
                     Ready dataset
```
### Step 1: Preprocessing of polygon GDF
**Script:** `gdf_class.py` (in folder `classes`)

**What the above script does:** Filters polygon GDF, and matches polygons to nc filenames.

**What you must do:**
- Create instance of GDFFilter with a path for the intended gpkg-file
- Manually select filter steps or use the `full_filtering` method
- Call the save method using intended save location

**Notes:**
- Methods for filtering / testing exist in the class for verifying that polygons match to the correct amount of files. As these are task specific, those were not added to the  `full_filtering` method but it is recommended to examine such aspects.
- Clustering should be done at some point if intending to use `load_data.py`. However it is not strictly necessary to do before Zarr-creation. Later the Zarr-wrapper class
uses a gdf/gkpg-file and one can then select a gdf with a "cluster" column if that is needed.
- If loading a gpkg that has gone through all filtering, it will contain a `Years_Areas_json` column. It is recommended to the call the `load_years_areas` method in the class.

### Step 1 - alternative (original solution), skip this if you followed the other Step 1 above.
If the full_filtering function in `gdf_class.py` does not work, you may need to save the gdf after filtering and then use it in `align_gpkg_dataset_centered.py` that can be found in `main_scripts`. 

**What you must do:** 
- Save GDF after filtering steps
- Update paths in `align_gpkg_dataset_centered.py` (save directory, GPKG path, etc)
- This new GDF will have the `Years_Areas_json` column

### Step 2: Zarr creation
**Script:** `zarr_creation.py` (in folder `main_scripts`)

**What the above script does:** Creates a Zarr storage (.zarr) for the dataset. 

**What you must do:** In the beginning of the script are some variables that need to updated for the relevant ones you're using. For example the `years` can easily be changed to some other folders.

**Notes:**
- It extracts images and creates the dataset via the `Years_Areas` column in the gdf. `Years_Areas` is used to point to the filenames where relevant images exist for each polygon
- This design is for our specific dataset, however this script can be used as a blueprint for creating similar zarr datasets. 

### Step 3: Zarr wrapper class
**Script:** `zarr_class.py` (in folder `classes`)

**What the above script does:** A wrapper class for the zarr storage. This is the main way of interacting with the zarr dataset and an object of the Zarr class is needed
for most scripts in the project. An instance of the Zarr class is usually refered to as "cube".

**What you must do:** The Zarr class requires a path to the zarr storage used as well as a path to the gpkg that is used for polygon-data. It is important that these two
match so that e.g. polygon 5 in the gpkg/gdf is the same as would appear in the images of `p=5` in  the zarr store. The combination of these two is what creates a useful
dataset. All the polygon data comes from the gpkg/gdf an all satellite data comes from the zarr storage.

**Notes:**
- in `__init__` you'll see multiple attributes. Some of these (such as means and stds) are hard-coded from a file. If that file doesn't exist, fallbacks are in place.
- When a cloud mask exists, it is advisable to initialize the Zarr wrapper class object with a path to this cloud mask. The class should work regardless but methods where
"cloud free" images can be specified will just return valid images, regardless of cloudiness (look at the file description of `zarr_class.py` for more information)

### Step 4: Creating cloud mask
**Script:** `cloud_mask_creation.py` (in folder `main_scripts`)

**What the above script does:** Creates and saves a cloud mask with specified parameters. It requires an instance of the Zarr wrapper class (cube) to work. 

**What you must do:**
- Create a "cube" object of the Zarr wrapper class from `zarr_class.py`.
- Update save directory.
- Specify file name and other input parameters to `cloud_filter_and_save`.

**Notes:**
- The current script modifies the `cloud_mask` attribute directly in the cube and then saves it. It might thus in future be best to change this so that the
cloud filtering instead creates the mask first and then saves it when the mask is fully populated.
- Different cloud thresholds introduce a trade-off: stricter filtering (e.g. 15) --> lower bias | looser filtering (e.g. 20) --> higher performance

### Step 5: Bias filtering
After cloud masking, additional filtering steps may be applied before ML training:

- Removal of "outliers" (see `data_statistics.py`), e.g. in the `__main__` function of `ml_main.py` there is a line `cube.remove_outliers(..)`
- Removal of images with negative values, e.g. in the `__main__` function of `ml_main.py` there is a line `cube.remove_negatives(..)`

**Note:**
- These steps are not primarily for improving raw data quality or performance.
- They are intended to reduce **temporal bias**, as certain artifacts are overrepresented in pre-2022 data.
- Omitting these steps typically does not significantly affect validation accuracy, but may increase the risk of the model learning temporal shortcuts (e.g. distinguishing years rather than ecological signals).

## ML pipeline

### Overview
```
                 Cluster GDF (Step 1)
                           │
               Create DataLoaders (Step 2)
                           │
           ┌───────────────┴───────────────┐
           │                               │
    Train 2D CNN (Step 3)            Train CNN-LSTM (Step 3)
           │                               │
    Evaluate (Step 4)                Evaluate (Step 4)
```
### Step 1 (optional): Cluster GDF
**Script:** `gdf_class.py` (in folder `classes`)

**What the above script does:** Adds a `cluster` column to the gdf. The cluster column contains which cluster the given polygon is part of. The clusters are created
by choosing a distance in meters (in thesis: 1500 meters). All polygons within that distance are assigned the same cluster (with chaining).

**What you must do: **
- Load the filtered GDF from previous steps.
- Set a cluster distance. A distance of 1500 m guarantees that all polygons that could be in the same image belong to the same cluster (for 1000x1000m images).
- Use the `cluster_by_distance` function to modify the GDF.
- Save to disk (update path).

### Step 2: Create DataLoaders
**Script:** `load_data.py` (in folder `ml_scripts`)

**What the above script does:** Specifies how the data is loaded during training. Whereas `zarr_class.py` is the most important script for working with the dataset,
`load_data.py` contains all the important functionallity for using the zarr dataset in ML. **The key method** in `load_data.py` is `load_data` which returns dataloaders
for the train, val and test sets. In the _File descriptions_ section further down, this module is more thoroughly explained.

**What you must do:**
- Create an object of the Zarr wrapper class in `zarr_class.py`. This requires a path to a zarr storage (where the images are stored), and a path to a matched gpkg-file, where each row corresponds to the same _p_ in the zarr. Meaning e.g. row 2 in the gpkg (or gdf once it's loaded) corresponds to `poly_idx=2`. It is advisable to specify a path to a cloud mask when creating the object.
- Select a sample method. These are sample instructions for the DataLoader, explaining label and what parts of the data is used for a given sample. Examples of sample methods are `__first_and_last_year__` and `__first_and_last_year_monthly__`.
- Select a builder. This is a method that takes the instructions from the sample method and builds the actual sample at run time. Examples of builders are
`__build_global_normalized_masked_composite__` and `__build_masked_composite_monthly_global_normalization__`.
- Set a batch size
- To visualize the split, uncomment lines in the `load_data` function where `plot_split_map` is called. Update the save directory.

**Notes:**
- You can also get rid of the seeding if you don't want to use our split. The seed 16 is only checked to be a good split when using our clustering. It holds no other significance.
- Would in future be nice to add `plot_split_map=True/False` as input to `load_data`, instead of uncommenting as is currently done. 
- You can uncomment e.g. `overlap_test` to assure that train, test and val to ensure they have no overlap. 

### Step 3: Train network
**Script:** `ml_main.py` or `ml_main_timeseries.py` (in folder `ml_scripts`)

**What the above scripts do:** Defines and trains ML models. Results are saved in folders.

**What you must do:** 
- Update save directory in `main()` function
**Note: everything below is done in `__main__`.**
- Create a cube object, and modify as desired (set cloud filter, remove outliers). **Important:** The removal of outliers and negative-value samples is primarily
intended to reduce **temporal bias** in the dataset (e.g. pre- vs post-2022 differences). These steps are **not** strictly required for achieving good validation
performance, but may affect how much the model relies on dataset-specific artifacts.

- Call `main()` function to train. Fill in `additional_info`, `sample_method`, `builder`, `model_and_layers` and all other hyperparameters in the call. 
- Results are automatically printed and saved in a folder, located in the specified save directory.

**Notes:** 
- Instead of calling `main()`, `run_variants()` can be called to use the model-input combinations from the report. **Only implemented for 2D CNNs.**
- It is recommended to have a gdf/gpkg with a `clusters` collumn in the Zarr wrapper class object (cube) as the data split is then made to avoid as much local leakage as possible
- The result save folders are automatically named after the input argument `additional_info`.
- Defaults in our project is to use either `build_resnet50` or `build_genesis` as `model_and_layers` in `ml_main`. In `ml_main_timeseries`, we instead use `build_genesis_cnn_lstm`.
- **IMPORTANT:** Functionality in what is saved for models and how it is saved need some examining/revising after changes were made late in the project. The `network_info.txt` could need some modifications especially if the sample method or builder is not picked as a method but rather an object (as can be done for the `multi_year_bias_builder` etc.)

### Step 4: Evaluate performance
**Script:** `ml_main.py` or `ml_main_timeseries.py` (in folder `ml_scripts`)

**What the above scripts do:** Run a tester class in order to evaluate performance.

**What you must do:**
- Create an object of the `Test_runs` class, e.g. called tester. Needs the path to a ML run folder / model weights (note that model weights can be downloaded [here](https://drive.google.com/drive/folders/1RpOVOphp9IcDONkIgssxORzfYim8O76l?usp=sharing)). 
- Choose test to be run by calling a method in the `Test_runs` class. For example `tester.validation_set_test()`, but there are many others. 
- Results will be printed and/or saved in the save folder (same as the input path to create tester object)

**Notes:** 
- In `ml_main_timeseries`, the tester class is named `Test_runs_ts`.

## File descriptions

### zarr_class.py
Wrapper class for handling useful functions for using our zarr file storage. All functionality of the dataset lives here. If you want to use the Zarr dataset, 
this is where you do it.

**Important attributes:**
- `im_id`: this is the `image_index matrix` and has shape (P,T). For a given (p,t), the value is used to extract location and original file. As explained
in Zarr_creation. Not all timestamps contains images for every polygon. In `im_id`, these have value -1
- `x0`: corner coordinate of a given (p,t)
- `y0`: corner coordinate of a given (p,t)
- `timestamps`: Array of all dates (For example "2018-07-10")
- `p_dim`: number of polygons
- `t_dim`: total number of timesteps
- `gdf`: **THIS IS IMPORTANT.** The indices in the gdf is what is considered the polygon-indices. So all polygon geometries + metadata lives here. It is important that it is
the same gdf (or has the same rows) that was used to create the zarr dataset
- `cloud_mask`: This is a boolean mask used to indicate which images are cloudy. If a `cloud_path` is not sent in at initialization, this will mimic `im_ids` by setting all
"invalid images" to "cloudy".
- `precomputed_masks` and `mask_idx_map`: These are dictionaries where all masks for each timestep for each polygon are cached. This is important as masks are not consistent
for a given polygon over time. It is consistent inside a single year but not always across years. This is due to how the satellite images were downloaded in the
nc-files as well as how the polygons were matched with these.

**Bellow are some examples of how the zarr class can be used:**
- First an example of commonly used steps (each of these are delved deeper into later)
```code

    #Create cube object
    cloud_path=CLOUD_MASKS/"cloud_mask_04_17_thresholds_20_15_10.npy"
    cube=Zarr(ZARR,GPKGS/"final02-20.gpkg",cloud_path=cloud_path)
    
    #Choose a polygon
    polygon=15

    #Get timesteps for all cloud free images from the full restoration process
    ts=cube.cloud_free_range_ts(polygon)

    #Load all images from these ts for the given polygon and mask them as well as normalize them using the global mean and std (which are preloaded / need to be calculated) (the mask cache also needs to have been created)
    masked_imgs=cube.get_global_normalized_masked_images(polygon,ts)
```
- Ways of getting relevant time steps from the cube
```code
    #Create cube object
    cloud_path=CLOUD_MASKS/"cloud_mask_04_17_thresholds_20_15_10.npy"
    cube=Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path=cloud_path)
  
    #Choose polygon index
    polygon=10

    #Get all ts where polygon 10 has valid images (where ts are the time indices) if you remember: (p,t) -> a single image
    all_ts=cube.valid_time(polygon)

    #Or all cloud free time steps that have valid images
    all_cloud_free_ts=cube.cloud_free_time(polygon)

    #Or for a collection of ts, return only those for which the polygon has existing images for (same thing can be done for cloud_free_time).
    collection_of_ts=[10,11,12,13,14,15]
    valid_collection_of_ts=cube.valid_time(polygon,collection_of_ts)

    #Get all valid or cloud free ts from the first and last year of a polygons restoration process
    first_ts, last_ts=cube.first_and_year_after_ids(polygon)
    first_ts, last_ts=cube.first_and_year_after_ids(polygon,cloud_free=True) #Instead only gives those that are not cloudy according to the cloud mask

    #Get valid (or cloud free) ts from a given year of a pasture
    year=2022
    valid_ts_2022=cube.ids_from_year(polygon,2022)
       
    #Get all cloud free timesteps from the full restoration process of a pasture
    cloud_free_ts=cube.cloud_free_range_ts(polygon)

    #Get all cloud free timesteps between 2 specified years for a pasture
    cloud_free_ts=cube.cloud_free_range_ts(polygon,2018,2022)

    #Choose start date and end date
    start_date="2018-06-22"
    end_date="2020-07-01"

    #Get corresponding ts for all existing timesteps between these 
    range_of_ts=cube.range_to_indices(start_date,end_date)
```

### gdf_class.py
A wrapper class for gdf's (which is how the gpkg files are read).
- Example use of class
```code
    # Steps to commit full filtering of the gdf
    path_to_gpkg = "/home/aleksispi/Projects/nature-arla/data-from-jv/joined_allhist_with_arslager.gpkg"
    image_directory_path = "/home/aleksispi/Projects/nature-arla/sen2a-data-mark-georg"
    save_path = "/home/aleksispi/Projects/final_filtered_aleksis.gpkg"
    gdf_filter = GDFFilter(path_to_gpkg)
    gdf_filter.full_filtering(image_directory_path)
    gdf_filter.save_gdf(save_path)
```
### ml_main.py
File to run and evaluate 2D CNN models
- Example use of class

```
    # Define data cube. The cube can be modified in any way before starting to train (e.g. changing cloud mask)
    cube=zarr_class.Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_15_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/neg_img_mask.npy")

    # The following line will start training a network 
    main("EXAMPLE RUN", opt_method = torch.optim.AdamW, learning_rate=0.0001, weight_decay=0.0 ,scheduler=None,cube=cube ,loss_fn=nn.BCEWithLogitsLoss(), number_epochs=100,
    minibatch_size=64,sample_method=ld.first_and_last_year_balanced, builder=ld.build_global_normalized_masked_composite, model_and_layers=build_genesis)
    # This does the exact same thing
    run_variants(cube,models=["genesis"],variants=["baseline"])
    # This will run all the input-model combinations used in the report
    run_variants()
    # If you want to do a specific run, but modify it slightly, an ovveride can be used:
    overrides={"builder": ld.build_masked_composite_monthly_pasture_normalization_random, "epochs": 150,"weight_decay":0.0001}  # <---- not standard settings
    run_variants(cube,models=["genesis"],variants=["monthly stack pasture norm"],overrides=overrides)

    # In order to run tests on a previously trained model, we define a "tester" object as following
    tester = Test_runs("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/best_models/CNN/genesis_monthly stack pasture norm_wd0.01/")
    # For example, a test on the validation set is done by
    tester.validation_set_test()    
    # Testing pasture norm bias
    results = tester.pasture_bias_test_stats(og_builder="stack",set_loader="test")
    tester.plot_pasture_bias(results,set_loader="test")
    # Testing in which year a pasture is classified as restored
    results = tester.when_restored_test_stats(og_builder="stack",set_loader="val")
    tester.plot_when_restored(results,set_loader="test")
    # But more tests are available (!). See Test_runs class for more info
```
**Some additional notes relative to the above (ml_main) AND below (ml_main_timeseries)**

* Before trying commands such as
```
# The following line will start training a network 
    main("EXAMPLE RUN", opt_method = torch.optim.AdamW, learning_rate=0.0001, weight_decay=0.0 ,scheduler=None,cube=cube ,loss_fn=nn.BCEWithLogitsLoss(), number_epochs=100,
    minibatch_size=64,sample_method=ld.first_and_last_year_balanced, builder=ld.build_global_normalized_masked_composite, model_and_layers=build_genesis)
```
and noted so far, it might be convenient to first run `conda activate nature_res` [TODO: MUST ADD THE ENV!].
* The specific way to run a command is e.g. `python -m ml_scripts.ml_main` (i.e. a module).

### ml_main_timeseries.py
File to run and evaluate CNN-LSTM models
- Example use of class

```
    # Define data cube. The cube can be modified in any way before starting to train (e.g. changing cloud mask)
    cube=zarr_class.Zarr(ZARR,GPKGS/"polygons_with_clusters_dist_1500.gpkg",cloud_path="/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/cloud_masks/cloud_mask_04_17_thresholds_20_15_10.npy")
    cube.remove_outliers("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/img_runs/data_statistics/outliers/outliers.pkl")
    cube.remove_negatives("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/linnea_experimentation/neg_img_mask.npy")

    # The following line will start training a network 
    main("EXAMPLE RUN", opt_method = torch.optim.AdamW, learning_rate=0.0001, num_classes=1, weight_decay=0.0,loss_fn=nn.BCEWithLogitsLoss(), number_epochs=100, minibatch_size=12, sample_method=ld.first_and_last_year_balanced,
          builder=ld.build_raw_masked_pasture, model_and_layers=build_genesis_cnn_lstm, cube=cube, metrics={"accuracy": BinaryAccuracyMetric()},temp_mode="final")  # temp_mode is final hidden state or attention
    
    # The tests are run similarly to ml_main
    tester = Test_runs_ts("/home/aleksispi/Projects/nature-arla/ml_landscape_arla/ml_runs_timeseries/2026-05-27_06:27 Fifty fifty, pasture + final hidden state/",temp_mode="final")  # <-- NOTE Test_runs_ts instead of Test_runs
    tester.test_set_test()
```

### zarr_creation.py
Script for combining nc-files with corresponding polygons into a zarr-dataset. The dataset shares a global time axis, created in the beginning of the script. For timestamps where a given polygon doesn't contain an image, the data is filled with -9999 instead and its `image_index` is set to -1. Doing this, all valid times for all polygons can be found as they have all image `index=-1`.

## Data-file descriptions

### Gpkgs / gdf
These exist in the directory: `data/gpkg_files/` and the main ones are `final02-20.gpkg` which is the polygon data used widly in the project, as well as
`polygons_with_clusters_dist_1500.gpkg` which is the same file but with the `cluster` column added. Note that `polygons02-20.gpkg` is the raw starting point, i.e the one that (after filtering etc) becomes `final02-20`.

### Cloud masks
These exist in the directory: `data/cloud_masks/` and the main ones are `cloud_mask_04_17_thresholds_20_15_10.npy` and `cloud_mask_04_17_thresholds_15_15_10.npy`.
They are both unbiased (no -1000 in cloud filtering), but have slightly different thresholds in the cloud detection. See `main_scripts/cloud_mask_creation.py` for more info.

### Means and standard deviations
These exist in the directory: `data/mean_std/` and the main ones are `training_set_means_stds_with_range_ts_05_14.npy` (used for global normalization), `pasture_means.npy`
and `pasture_stds.npy` (the latter two used for per-pasture normalization).

## Citation
If you use this repository and/or find our paper useful, please cite the following:

    @article{sartorius2026monitoring,
      title={Monitoring Pasture Restoration from Satellite Image Time Series: Caveats and Opportunities},
      author={TODO},
      journal={TODO},
      year={2026}
    }

## License
This project is released under the **MIT License**.  
Copyright (c) 2025, RISE Research Institutes of Sweden.

See the full text in [the license file](https://github.com/aleksispi/ml-landscape-arla/blob/resto-cam-ready/license.md).
