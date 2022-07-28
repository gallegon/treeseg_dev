from treesegmentation.treeseg_lib import *

import treeseg as ts


def handle_c_stage(grid):
    sample = ts.sample_grid(100, grid)
    print("Sampled grid:")
    print(sample)
    print("Grid:")
    print(grid)


c_pipeline = Pipeline(verbose=True) \
    .then(handle_create_file_names_and_paths) \
    .then(handle_read_las_data) \
    .then(handle_las2img) \
    .then(handle_gaussian_filter) \
    .then(handle_grid_height_cutoff) \
    .then(handle_save_grid_raster) \
    \
    .then(handle_c_stage) \
    # \
    # .then(handle_compute_patches) \
    # .then(handle_patches_to_dict) \
    # .then(handle_compute_patches_labeled_grid) \
    # .then(handle_compute_patch_neighbors) \
    # .then(handle_save_patches_raster) \
    # .then(handle_compute_hierarchies) \
    # .then(handle_find_connected_hierarchies) \
    # .then(handle_calculate_edge_weight) \
    # .then(handle_partition_graph) \
    # .then(handle_trees_to_labeled_grid) \
    # .then(handle_save_partition_raster) \
    # .then(handle_label_point_cloud) \
    # .then(handle_save_context_file)
