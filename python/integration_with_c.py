from treesegmentation.ts_api import *
import treeseg_ext as ts

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

    dt = np.dtype('int32')

    weight_params = [weight_level_depth, weight_node_depth, weight_shared_ratio, weight_top_distance, weight_centroid_distance, weight_threshold]
    hierarchy_labels = ts.vector_test(grid.astype(dt), labeled_grid.astype(dt), weight_params)
    
    print("Vector test end!")

    print("== Grid")
    print(grid)
    print()
    print("== Labeled Grid")
    print(labeled_grid)
    print()
    print("== Segmented Hierarchies")
    print(hierarchy_labels)
    print()

    return {
        "labeled_partitions": hierarchy_labels
    }


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

def handle_read_and_discretize_points(input_file_path, resolution, discretization):
    grid = ts.discretize_points(input_file_path, resolution, discretization)
    
    return {
        "grid": grid
    }

c_pipeline = Pipeline().then([
    handle_create_file_names_and_paths,
    handle_read_and_discretize_points,
    handle_gaussian_filter,
    handle_grid_height_cutoff,
    handle_save_grid_raster,
    handle_label_patches,
    handle_save_patches_raster,
    handle_vector_test,
    handle_save_partition_raster
]) \
.transform([
    transform_print_stage_info,
    transform_print_runtime,
    transform_print_newline
])
