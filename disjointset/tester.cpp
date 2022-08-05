#include "disjointsets.hpp"
#include <iostream>


void print_disjointsets(DisjointSets ds) {
    for (std::map<PatchID, PatchNode>::iterator it = ds.nodes.begin(); it != ds.nodes.end(); ++it) {
        PatchID id = it->first;
        PatchNode node = it->second;
        std::cout << "ID [" << id << "]" << std::endl;
        std::cout << "height = " << node.height << std::endl;
        std::cout << "size = " << node.size() << std::endl;
        std::cout << "parent = " << ds.parent_of(node.id) << std::endl;
        std::cout << std::endl;
    }
}


void label_grid(DisjointSets ds, int* grid, int width, int height) {
    std::cout << "Begin label!" << std::endl;
    for (std::map<PatchID, PatchNode>::iterator it = ds.nodes.begin(); it != ds.nodes.end(); ++it) {
        std::cout << "Looking at node: " << it->first << std::endl;
        PatchNode patch = it->second;
        std::vector<Cell> cells = patch.cells;
        for (std::vector<Cell>::iterator cit = cells.begin(); cit != cells.end(); ++cit) {
            int x = cit->x;
            int y = cit->y;
            std::cout << "(" << x << ", " << y << ")" << std::endl;
            grid[x + y * width] = ds.parent_of(patch.id);
        }
    }

    std::cout << "End label" << std::endl;
}


void print_grid(int* grid, int width, int height) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            std::cout << grid[i + j * width] << ", ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char** argv) {
    std::cout << "Hello, DisjointSets!" << std::endl;

    const int width = 3;
    const int height = 3;
    int data[width * height] = {
        5, 5, 5,
        5, 3, 1,
        3, 1, 1
    };

    DisjointSets ds;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int h = data[i + j * width];
            PatchID patch = ds.make_patch({i, j}, h);

            if (i > 0 && data[(i - 1) + j * width] == h) {
                ds.union_patches(ds.find_patch({i - 1, j}), patch);
            }

            if (j > 0 && data[i + (j - 1) * width] == h) {
                ds.union_patches(ds.find_patch({i, j - 1}), patch);
            }
        }
    }

    // PatchID a = ds.make_patch({0, 0}, 4);
    // PatchID b = ds.make_patch({0, 1}, 4);
    // PatchID c = ds.make_patch({1, 0}, 4);
    // PatchID d = ds.make_patch({1, 1}, 2);

    // ds.union_patches(a, b);
    // ds.union_patches(c, b);

    print_disjointsets(ds);

    int labeled_grid[width * height];

    label_grid(ds, labeled_grid, width, height);

    print_grid(labeled_grid, width, height);

    print_grid(data, width, height);
}