from treesegmentation.treeseg_lib import *

import treeseg as ts


def handle_c_stage(grid):
    sample = ts.sample_grid(10, grid)
    print("Sampled grid:")
    print(sample)
    print("Grid:")
    print(grid)

    # return { "grid": sample }


def handle_vector_test(grid, labeled_grid):
    print("Vector test begin!")
    ts.vector_test(grid, labeled_grid)
    print("Vector test end!")

    print("== Grid")
    print(grid)
    print()
    print("== Labeled Grid")
    print(labeled_grid)
    print()


def handle_label_patches(grid):
    labeled_grid = ts.label_grid(grid).astype("int")

    print("== Labeled Grid")
    print(labeled_grid)
    
    return {
        "labeled_grid": labeled_grid
    }

py_pipeline = Pipeline(verbose=True) \
    .then(handle_create_file_names_and_paths) \
    .then(handle_read_las_data) \
    .then(handle_las2img) \
    .then(handle_gaussian_filter) \
    .then(handle_grid_height_cutoff) \
    .then(handle_save_grid_raster) \
    \
    .then(handle_compute_patches) \
    .then(handle_compute_patches_labeled_grid) \
    .then(handle_compute_patch_neighbors) \
    .then(handle_save_patches_raster) \
    \
    .then(handle_vector_test)


c_pipeline = Pipeline(verbose=True) \
    .then(handle_create_file_names_and_paths) \
    .then(handle_read_las_data) \
    .then(handle_las2img) \
    .then(handle_gaussian_filter) \
    .then(handle_grid_height_cutoff) \
    .then(handle_save_grid_raster) \
    \
    .then(handle_label_patches) \
    \
    .then(handle_save_patches_raster) \
    \
    .then(handle_vector_test)

