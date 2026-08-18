import geopandas as gpd
import functions.gdf_utils as gu
import tests.gdf_tests as gt
import functions.utils as ut 
import matplotlib.pyplot as plt
import netCDF4
import json
import networkx as nx
import fiona
import os, sys
from shapely.geometry import box
import numpy as np

class GDFFilter:
    """
    Class for handling filtering and tests of a gdf to make it Zarr-Ready. The GPKG used is assumed to only contain 1 layer and
    that data is in EPSG:3006
    """

    def __init__(self,path_to_gpkg):
        """
        Reading GPKG file and calculating the area manually of each polygon and adding that as a column
        """
        self.gdf=gpd.read_file(path_to_gpkg)

        self.gdf["calc_area"]=self.gdf.geometry.area
        self.path_to_gpkg=path_to_gpkg
    
    def extract_original_gdf(self):
        """
        Returns: the unmodified gdf
        """
        return gpd.read_file(self.path_to_gpkg)
    
    def get_gdf(self):
        """
        Returns: the modified gdf
        """
        return self.gdf

    def filter_to_polygons(self):
        """
        Iterates through the rows, getting rid of all geometries that aren't polygons, multipolygons or geometrycollections.
        For multipolygons and geometry collections, extracts each individual polygon to a separate row. Then recalculates the Area as small polygons can be separated from old ones.
        
        Returns: list of the indices in the original gdf that contained multiple polygons
        """
        self.gdf, multi_polys=gu.get_polygon_gdf(self.gdf)
        self.gdf["calc_area"]=self.gdf.geometry.area
        return multi_polys
    
    def info(self):
        """
        Overview of the gdf contents
        """
        self.gdf.info()

    def prune_small_area(self,thresh=100):
        """
        Filters out polygons that have an area that is too small
        """
        self.gdf=self.gdf[self.gdf["calc_area"]>=thresh]

    def reset(self):
        """
        Resets the indicies of the gdf, so that they are ordered from 0 to len
        """
        self.gdf=self.gdf.reset_index(drop=True)

    def duplicate_polygon_test(self):
        """
        Checks if there are duplicates of any polygons

        Returns: False if there are no duplicates
                 List of matching polygon_indicies and id_origs if there are duplicates
        """
        return gt.check_duplicate_polygons(self.gdf)
        
    def empty_polygons(self,printing=True):
        """
        Checks if any polygons don't match with any images in the nc files by looking at the collumn "Year_Areas"

        Returns: False if all polygons are matched with images
                 List of indices of polygons that have empty "Years_Areas" columns
        """
        return gt.no_image_polygons(self.gdf,printing=printing)
    
    def len(self,printing=True):
        """
        Prints current lenght of the gdf

        Returns Int of the length
        """
        if printing:
            print(f"Number Rows: {len(self.gdf)}")
        return(len(self.gdf))
    
    def print_year_info(self, sample):
        """
        Prints relevant info of a gdf/sample of a gdf
        """
        for _,row in sample.iterrows():
            lst = row["Years_Areas"]
            print("First Year", row["firstYear"])
            print("Last Year", row["lastYear"])
            print("Year areas:", lst)
            print("List length:", len(lst))
            print("AREA SIZE: ", row["calc_area"])
            print("-" * 40)

    def year_length_match(self):
        """
        Checks if the "Years_Areas" column contain the correct amount of entries

        Returns: True if all contain correct amount of entries
                 List indices of faulty rows if there are any
        """
        return gt.year_length_match(self.gdf)

    def no_year_after(self):
        """
        Checks if the "Years_Areas" column contain the entry "year-after-data"
        
        Returns: False if there are no rows with no "year-after-data" entry
                 List of indices of rows with no "year-after-data" entry
        """
    
        return gt.no_year_after(self.gdf)
    
    def filter_no_year(self):
        """
        Filters out all rows that don't contain "year-after-data" in "Years_Areas" column
        """
        target="year-after-data"
        gdf=self.gdf
        mask=gdf["Years_Areas"].apply(lambda lst: any(target in s for s in lst))
        self.gdf=gdf[mask]
    
    def ensure_years(self,filter=False):
        """Ensures that each polygon in the gdf contains all years from first to last + year after
        and returns True if that is true for all polygons. Otherwise returns a gdf of the faulty ones. If filter=True,
        instead returns the gdf with the faulty rows filtered out.
        """

        if filter:
            self.gdf=gt.ensure_years(self.gdf,filter=filter)
        else:
            return gt.ensure_years(self.gdf,filter=filter)
        
    def cluster_by_distance(self,distance):
        """
        Method for clustering the polygons by a given distance. The clustering is done with chaining. If distance 1000 is chosen, all polygons within 1000 meters of eachother are assigned to
        the same cluster. If polygons A, B, C exist and are in a horizontal row with 800m between A-B and B-C, all three are assigned to the same cluster

        Returns: The gdf but with the cluster column added
        """

        gdf=self.gdf

        #Creates a gdf where each geomety is "padded" with the distance
        buffered=gdf.copy()
        buffered["geometry"]=buffered.geometry.buffer(distance)

        #Returns gdf where each column called "XXX_left" is from buffered, and "XXX_right" is corresponding index from the
        #Original gdf
        joined=gpd.sjoin(buffered,gdf,how="left",predicate="intersects")
        joined = joined.reset_index().rename(columns={"index": "index_left"})

        #Extracts only index columns
        joined=joined[["index_left","index_right"]]

        #builds graph, adds one node for each idx
        G=nx.Graph()
        for idx in gdf.index:
            G.add_node(idx)
        
        #Adds edge between overlapping ids (False to not include the "index")
        for left, right in joined.itertuples(index=False):
            G.add_edge(left, right)
        
        #Generator of connected components in the graph
        components = nx.connected_components(G)

        #Assigns to cluster
        cluster_id = {}
        for cid, comp in enumerate(components):
            for idx in comp:
                cluster_id[idx] = cid

        #Adds cluster column to gdf
        gdf["cluster"] = gdf.index.map(cluster_id)
    
        return gdf
    
    def align_gpkg_centered(self,image_directory,folders=None):
        """
        Extracting all images from the NC files for each folder and checking wether the polygons exist within each image, 
        and saving the area name of the image in which each polygon is most aligned for every folder. Adds a "Years_Areas" column.
        Args:
            image_directory (str): path to the nc files.
            folders (string list): folders in image_directory (e.g. [2018, 2019, 2020])
        """

        # Copy gdf to avoid modifying original
        gdf = self.gdf.copy()

        # Initialize storage: for each polygon, a list of (year, area)
        gdf["Years_Areas"] = [[] for _ in range(len(gdf))]

        # If folders not specified → use all year folders
        if folders is None:
            folders = [name for name in os.listdir(image_directory)
                    if os.path.isdir(os.path.join(image_directory, name))]

        def _read_bbox(filename):
            # Inner function used to get x-y-span of an nc-image
            im_nc = netCDF4.Dataset(filename)
            x = im_nc.variables['x'][:]
            y = im_nc.variables['y'][:]
            return x, y

        # Build spatial index (for fast spatial queries)
        spatial_index = gdf.sindex

        # Precompute centroids (to avoid recomputing in inner loop below)
        centroids = gdf.geometry.centroid

        # Loop over years
        for folder_idx, folder in enumerate(folders):

            # Build path for this year
            load_path = os.path.join(image_directory, folder)
            if not os.path.exists(load_path):
                print(f"Error: The path {load_path} does not exist.")
                sys.exit()

            # Collect and sort all files (sorted by area index)
            files = [os.path.join(load_path, filepath) for filepath in os.listdir(load_path)]
            files = sorted(files, key=lambda x: int(x.split('/')[-1].split('_')[1]))

            # Keep only valid 10x10 files that have corresponding 20x20 and 60x60 versions
            files20 = [file for file in files if "20x20" in file]
            files60 = [file for file in files if "60x60" in file]
            files = [file for file in files if "10x10" in file]
            files = [file for file in files if file.replace("10x10", "20x20") in files20
                                            and file.replace("10x10", "60x60") in files60]

            # For each polygon: track best matching image (closest to center)
            area_names = np.full(len(gdf), None)
            min_dists = np.full(len(gdf), 999999)

            # Loop over all image tiles in this year
            for file_idx in range(len(files)):

                # Progress print (very useful for long runs)
                print("align_gpkg_centered, folder progress: %d / %d (%s), file progress: %d / %d"
                    % (folder_idx, len(folders), folder, file_idx, len(files)))

                # Read x-y-span of image
                x, y = _read_bbox(files[file_idx])

                # Compute bounding box of the image tile
                im_min_lon = np.min(x)
                im_min_lat = np.min(y)
                im_max_lon = np.max(x)
                im_max_lat = np.max(y)

                # Convert bounding box → shapely polygon
                im_bbox = [im_min_lon, im_min_lat, im_max_lon, im_max_lat]
                im_poly = box(*im_bbox)
                center = im_poly.centroid

                # Extract area name from filename
                base = os.path.basename(files[file_idx])
                area = base.split("_10x10")[0]

                # Query spatial index to get candidate polygons intersecting bbox
                candidate_idxs = list(spatial_index.intersection(im_poly.bounds))

                # Loop only over candidates
                for idx in candidate_idxs:

                    gpkg_poly = gdf.geometry.iloc[idx]

                    # Precise check (index only gives bounding box intersection)
                    if gpkg_poly.within(im_poly):

                        # Get precomputed centroid
                        centroid = centroids.iloc[idx]
                        dist = centroid.distance(center)

                        # Keep the tile where polygon is most centered
                        if dist < min_dists[idx]:
                            min_dists[idx] = dist
                            area_names[idx] = area

            # After processing all tiles for this year → store assignments
            valid_idxs = np.flatnonzero(area_names != None)
            for idx in valid_idxs:
                gdf.at[idx, "Years_Areas"].append(os.path.join(folder, area_names[idx]))

        return gdf

    def save_gdf(self, save_path):
        """
        Method for saving the gdf as a gpkg file. As lists are objects and cannot be saved, they need to be converted to JSON formatted strings
        """

        # Convert to json
        self.gdf["Years_Areas_json"] = self.gdf["Years_Areas"].apply(json.dumps)

        # Drop the collumn with lists
        gdf_to_save = self.gdf.drop(columns=["Years_Areas"])

        gdf_to_save.to_file(
            save_path,
            layer="polygons",
            driver="GPKG"
        )

    def load_years_areas(self):
        """
        Method for extracting the "Years_Areas" collumn when working with a gdf that has already been filtered and therefore contains the 
        "Years_Areas_json" collumn when loaded
        """
        self.gdf["Years_Areas"] = self.gdf["Years_Areas_json"].apply(json.loads)
        self.gdf=self.gdf.drop(columns=["Years_Areas_json"])

    def full_filtering(self,image_directory,area_thresh=400):
        """
        Method for going through all important filtering of the gpkg from start to finish
        """
        
        # Filter out all geometries so that only polygons remain. For multipolygons and GeometryCollections, separates into individual polygon rows
        # multi_poly_ids contains the ids of original rows that contained multiple polygons
        self.filter_to_polygons()

        # Prunes small pastures from the gdf
        if area_thresh is not None:
            self.prune_small_area(thresh=area_thresh)

        # Resets the index
        self.reset()

        # Checks for duplicate geometries
        self.duplicate_polygon_test

        # Matches rows to image-series
        self.gdf = self.align_gpkg_centered(image_directory)

if __name__=="__main__":
    from paths import GPKGS

    """
    Comments (in Swedish) by Isak regarding slight diff in size (1.9 mb, vs orig by Isak and Linnea at 1.8 mb):

    Den vi använder som start är polygons02-20. Har 2 gissningar för skillnaden och tror mer på den 1a:

    1) i funktionen som gör "alla" filtersteg så valde vi att inte inkludera filtrering som var specifik till vår setup.
    Alltså att alla ska ha åren mellan sin first och last och sedan year-after. Det finns en metod "ensure years" eller nåt
    som gör ett extra filtersteg som man kan göra efteråt. Skulle kunna vara det som gör skillnaden.
    Vi skippar att inkludera detta eftersom vi inte visste om era senare samples kommer ha samma antal år/samma range eller
    va skapta med samma tanke.

    2) Ifall size-thresholden är annorlunda känns det som det också kan bli skillnad 
    """

    # Steps
    path_to_gpkg = "/home/aleksispi/Projects/nature-arla/ml_landscape_arla/data/gpkg_files/polygons02-20.gpkg"  # 1.9 mb result (so far closest to the 1.8 mb result of final02-20.gpkg) -- this one originally used by Isak and Linnea
    image_directory_path = "/home/aleksispi/Projects/nature-arla/sen2a-data-mark-georg"
    save_path = "/home/aleksispi/Projects/final_filtered_aleksis.gpkg"
    gdf_filter = GDFFilter(path_to_gpkg)
    print("GOT HERE")
    gdf_filter.full_filtering(image_directory_path)
    gdf_filter.save_gdf(save_path)
    print("GOT HERE TOO")
    sys.exit()
    #gdf_filter
    #
    # Perform clustering
    #cluster_distance = 1500
    #gdf=gdf.cluster_by_distance(cluster_distance)
    #gdf.info()
    # print(gdf.sample(10)["cluster"])
    # Save polygons with cluster column to disk
    # out_path = GPKGS/f"polygons_with_clusters_dist_{cluster_distance}.gpkg"
    # gdf.to_file(
    #    out_path,
    #    layer="polygons",
    #    driver="GPKG"
    # )

    #gdf_og=gdf.extract_og(path_to_gpkg)
    #gdf.gdf["Years_Areas"] = gdf.gdf["Years_Areas_json"].apply(json.loads)
    #gdf.info()
    #gdf.len()
    #multipolys=gdf.filter_to_polygons()
    #gdf.len()
    ##df.prune_small_area(thresh=400)
    #gdf.len()
    #gdf.reset()
    #gdf.duplicate_polygon_test()
    #missing=gdf.empty_polygons(printing=False)
    #missing.info()
    #faulty=gdf.year_length_match()
    ##gdf.print_year_info(faulty.sample(20))
    #no_year=gdf.no_year_after()
    #gdf.print_year_info(no_year.sample(10))
    #print(len(no_year))
    #gdf.len()
    #gdf.filter_no_year()
    #gdf.len()
    #no_year=gdf.no_year_after()
    #gdf.reset()
    #gdf=gdf.get_gdf()
    #gdf.info()
    #gdf["Years_Areas_json"] = gdf["Years_Areas"].apply(json.dumps)

    #gdf_to_save = gdf.drop(columns=["Years_Areas"])
    #out_path=GPKGS/"final02-20.gpkg"
    #gdf_to_save.to_file(
    #    out_path,
    #    layer="polygons",
    #    driver="GPKG"
    #)
    #
    ##gdf=gdf.get_gdf()
    ##for idx in multipolys:
    #    row=gdf_og.iloc[[idx]]
    #    gu.plot_multi_geom_row(f"Row_{idx}",row,"Gdf_class_exp",fig=None,ax=None)
    ##out_path=GPKGS/"filtered_polygons02-20.gpkg"
    #gdf.to_file(
    #    out_path,
    #    layer="polygons",
    #    driver="GPKG"
    #)