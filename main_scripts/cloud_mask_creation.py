import numpy as np
from functions.cloud_filter import cloud_filter
from classes import zarr_class
from paths import GPKGS, ZARR, CLOUD_MASKS
import os, sys


def cloud_filter_and_save(cube, save_dir, filename,
                          thresh_thick, thresh_thin, poly_frac_thresh,
                          poly_ids):
    """
    Applies cloud filter to selected polygons and saves mask to disk.
    """

    # Reset mask
    cube.cloud_mask = np.zeros_like(cube.im_id, dtype=bool)

    # Iterate over polygons
    for poly_id in poly_ids:
        print(f"Processing polygon {poly_id} / {cube.p_dim}")

        im_series = cube.get_normalized_images(poly_id, np.arange(cube.t_dim))
        im_ids = cube.im_id[poly_id, :]


        # Initialize polymask stack
        poly_masks = np.ones((cube.t_dim, cube.W, cube.H), dtype=bool)

        for t in cube.valid_time(poly_id):
            _, poly_mask = cube.align_poly(poly_id, t)
            poly_masks[t, :, :] = poly_mask

        # Apply cloud filter
        _, cloudy_flag = cloud_filter(
            im_series,
            poly_masks,
            im_ids,
            THRESHOLD_THICKNESS_IS_CLOUD=thresh_thick,
            THRESHOLD_THICKNESS_IS_THIN_CLOUD=thresh_thin,
            POLY_CLOUD_FRAC_THRESH=poly_frac_thresh
        )

        # Update cube mask
        cube.update_cloud_mask(cloudy_flag, poly_id)

    # Save mask
    cube.save_cloud_mask_to_disk(save_dir / filename)
    print(f"Cloud mask saved to disk: {save_dir / filename}")


if __name__ == "__main__":

    # Set up some things
    path_to_gpkg = GPKGS / 'final02-20.gpkg'
    ZARR = ZARR.with_name(ZARR.stem + "_aleksis.zarr")
    cube = zarr_class.Zarr(ZARR, path_to_gpkg)
    poly_ids = range(cube.p_dim)

    # Setup save path
    save_dir = CLOUD_MASKS
    os.makedirs(save_dir, exist_ok=True)

    # First mask
    cloud_filter_and_save(
        cube,
        save_dir,
        "cloud_mask_aleksis_15_15_10.npy",
        thresh_thick=0.015,
        thresh_thin=0.015,
        poly_frac_thresh=0.01,
        poly_ids=poly_ids
    )

    # Second mask
    cloud_filter_and_save(
        cube,
        save_dir,
        "cloud_mask_aleksis_20_15_10.npy",
        thresh_thick=0.020,
        thresh_thin=0.015,
        poly_frac_thresh=0.01,
        poly_ids=poly_ids
    )