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

import cv2

def handle_label_patches(grid):
    image = grid.astype(np.uint8)
    labeled_grid = np.ndarray(grid.shape, dtype=np.uint8)
    connectivity = 4
    num_labels, labels = cv2.connectedComponents(image, labeled_grid, connectivity)

    # print(f"{num_labels=}")
    # print(f"Labels:")
    # print(f"{labels}")
    # print(f"labeled_grid:")
    # print(f"{labeled_grid}")

    # return {
    #     "labeled_grid":
    # }



c_pipeline = Pipeline(verbose=True) \
    .then(handle_create_file_names_and_paths) \
    .then(handle_read_las_data) \
    .then(handle_las2img) \
    .then(handle_gaussian_filter) \
    .then(handle_grid_height_cutoff) \
    .then(handle_save_grid_raster) \
    \
    .then(handle_c_stage) \
    \
    .then(handle_compute_patches) \
    .then(handle_patches_to_dict) \
    .then(handle_compute_patches_labeled_grid) \
    \
    .then(handle_vector_test) \
    \
    .then(handle_save_patches_raster)

