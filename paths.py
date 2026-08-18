from pathlib import Path

"""
Path ground truth. Use this to import paths

"""


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GPKGS= DATA / "gpkg_files"
ZARR = DATA / "satellite.zarr"
NC=ROOT.parent / "sen2a-data-mark-georg"
COT_MODEL = ROOT.parent.parent/"cot-model/skogs_models"
IMGS = ROOT / "img_runs"
CLOUD_MASKS=ROOT/ DATA /"cloud_masks"
POLY_MASKS=DATA/"poly_masks"
REPORT_FIGS=ROOT/"report_figs"

"""
Script to do a print of the paths to check for correct behaviour
"""

if __name__ == "__main__":
    print("ROOT: ", ROOT)
    print("DATA: ", DATA)
    print("GPKGS: ", GPKGS)
    print("ZARR: ", ZARR)
    print(NC)