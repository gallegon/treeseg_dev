from treesegmentation.treeseg_lib import *

import treeseg as ts
import os


def handle_c_stage(grid):
    sample = ts.sample_grid(10, grid)
    print("Sampled grid:")
    print(sample)
    print("Grid:")
    print(grid)

    # return { "grid": sample }


def handle_vector_test(grid, labeled_grid, weight_level_depth,
                                 weight_node_depth, weight_shared_ratio,
                                 weight_top_distance, weight_centroid_distance, weight_threshold):
    print("Vector test begin!")

    weight_params = [weight_level_depth, weight_node_depth, weight_shared_ratio, weight_top_distance, weight_centroid_distance, weight_threshold]
    ts.vector_test(grid, labeled_grid, weight_params)
    
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

def handle_label_las(input_file_path, output_folder_path, grid):
    output_file_path = os.path.join(output_folder_path, "test_output.las")
    ts.label_las(input_file_path, output_file_path, grid)

def handle_discretize_points(input_file_path, resolution, discretization):
    grid = ts.discretize_points(input_file_path, resolution, discretization)
    
    return {
        "grid": grid
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
    .then(handle_patches_to_dict) \
    .then(handle_compute_patches_labeled_grid) \
    .then(handle_compute_patch_neighbors) \
    .then(handle_save_patches_raster) \
    .then(handle_compute_hierarchies) \
    # .then(handle_find_connected_hierarchies)


c_pipeline = Pipeline(verbose=True) \
    .then(handle_create_file_names_and_paths) \
    .then(handle_discretize_points) \
    \
    .then(handle_gaussian_filter) \
    .then(handle_grid_height_cutoff) \
    .then(handle_save_grid_raster) \
    \
    .then(handle_label_patches) \
    \
    .then(handle_save_patches_raster) \
    \
    .then(handle_vector_test)

