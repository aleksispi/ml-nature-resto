import geopandas as gpd
import geopy.distance
import numpy as np
import shapely as sh
import geopandas as gpd
from shapely.strtree import STRtree
import pandas as pd

def check_duplicate_values(gdf):
    """
    Checking how many duplicate values there are of different collumns
    """
    
    for i in range(0,len(gdf.columns)-2):
        extracted_column=gdf.iloc[:,i].to_numpy()
        unq, unq_idx, unq_cnt = np.unique(extracted_column, return_inverse=True, return_counts=True)
        mask=unq_cnt>1
        dups=unq[mask]

        print(f"Number duplicates in {gdf.columns[i]}: ", len(dups))   
    #Can add return later if that is wished


def check_duplicate_polygons(gdf):
    """
    Takes a list of geometries and returns a list of matching polygon_indicies and id_origs if there are any
    and returns False otherwise
    """
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    matches = []
    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if i < j and geom.equals(geoms[j]):
                id1 = gdf.iloc[i]["id_orig"]
                id2 = gdf.iloc[j]["id_orig"]

    
                matches.append((i, j, id1,id2))
    if len(matches)>0:
        print("There are multiple identical polygons!")
        return matches
    else:
        print("No identical polygons")
        return False

def single_year_test(gdf):
    """
    Checking if any polygons appear in multiple images
    
    Returns True if all polygons only contain one image-series per year
            False if any there are polygons with multiple image-series in a year
    """
    print("----------------------------------------------")
    print("Checking if any contain multiple nc-images of the same years")
    no_dups=0
    targets=["2018/","2019/","2020/","2021/","2022/","2023/","2024/","2025/"]
    for target in targets:

        mask = gdf["Years_Areas"].apply(
            lambda lst: sum(word.count(target) for word in lst) > 1
        )
        filtered_gdf = gdf[mask]
        no_dups+=len(filtered_gdf)
        try: 
            for i in range(10):
                row = filtered_gdf.sample(1).iloc[0]
                lst = row["Years_Areas"]

                matches = [x for x in lst if target in x]

                print("Full list:", lst)
                print("Matches:", matches)
                print("-" * 40)
        except Exception as xp:
            pass

    print("Total dups: ", no_dups)
    if no_dups!=0:
        print("All polygons do not contain only one image per year")
        return False
    else:
        print("All polygons contain only 1 image per year")
        return True

def no_image_polygons(gdf,printing=True):
    """
    Checks if any polyogns don't contain a single image

    returns True if all polygons are non-empty
            gdf with missing rows otherwise
    """

    print("------------------------------------------------------------")
    print("Checking for missing row")
    mask=gdf["Years_Areas"].apply(lambda lst: len(lst)==0)
    missing=gdf[mask]
    if len(missing)==0:
        print("No polygons have empty Year-Areas")
        return True
    else:
        print("There are polygons with no images")
        if printing:
            for _,row in missing.iterrows():
                lst = row["Years_Areas"]
                print("First Year", row["firstYear"])
                print("Last Year", row["lastYear"])
                print("Year areas:", lst)
                print("List length:", len(lst))
                print("AREA SIZE: ", row["calc_area"])
            print("-" * 40)
        
        return missing
    
def no_year_after(gdf):
    """
    Checks if the "Years_Areas" column contain the entry "year-after-data"

    Returns: False if there are no rows with no "year-after-data" entry
             gdf containing rows with no image of year after if there are any
    """
    
    target="year-after-data"
    mask=gdf["Years_Areas"].apply(lambda lst: not any(target in s for s in lst))
    no_year=gdf[mask]
    if len(no_year)==0:
        print("All polygons have year after")
        return True
    else:
        print("There are polygons with no image of year after")
        return no_year
    
def ensure_years(gdf,filter=False):
    """Ensures that each polygon in the gdf contains all years from first to last + year after
    and returns True if that is true for all polygons. Otherwise returns a gdf of the faulty ones. If filter=True,
    instead returns the gdf with the faulty rows filtered out.
    """
    mask=gdf.apply(row_matches, axis=1)
    faulty=gdf[~mask]
    if len(faulty)==0:
        print("All polygons have correct setup of years")
        return True
    else:
        print("There are polygons with wrong years or missing years")
        if not filter:
            return faulty
        else:
            return gdf[mask]

def compute_targets(row):
    """Helper method for ensure year. Computes targets for each row depending on first and last year
    """
    first=int(row["firstYear"])
    last=int(row["lastYear"])
    return [str(y)+"/" for y in range(first,last+1) ]+["year-after-data"]

def row_matches(row):
    """Helper method for ensure year. Returns True if row contains all years from first to last + year after
    """
    lst = row["Years_Areas"]
    targets=compute_targets(row)
    return all(any(t in s for s in lst) for t in targets)


def year_length_match(gdf):
    """
    Checks if there are expected amount of entries in "Years_Areas" column

    returns True if all have correct amount of images
            gdf with the faulty length rows if there are any
    """
    mask=gdf.apply(lambda row: (int(row["lastYear"])-int(row["firstYear"])+2)!=len(row["Years_Areas"]),axis=1)
    faulty=gdf[mask]
    if len(faulty)==0:
        print("All have correct ammount of images")
        return True
    else:
        print("There are polygons with wrong amount of images")
        return faulty