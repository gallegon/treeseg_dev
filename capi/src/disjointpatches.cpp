#include "disjointpatches.hpp"
#include <iostream>

// #define Ptr2D(array, i, j) ((int*) PyArray_GETPTR2(array, i, j))
// #define Get2D(array, i, j) (*((int*) PyArray_GETPTR2(array, i, j)))

DisjointPatches::DisjointPatches(Grid<int>& levels, Grid<int>& labels) : levels(levels), labels(labels) {
    // npy_intp* dims = PyArray_DIMS(levels);
    // width = dims[0];
    // height = dims[1];
    width = levels.width;
    height = levels.height;

    patch_count = 0;
}

PatchID DisjointPatches::make_patch(int x, int y, int height) {
    PatchID id = ID_NEXT++;
    parents[id] = id;
    // *Ptr2D(labels, x, y) = patch.id;
    return id;
}

PatchID DisjointPatches::union_patches(PatchID a, PatchID b) {
    PatchID ida = parent_of(a);
    PatchID idb = parent_of(b);

    if (ida == idb) {
        return ida;
    }

    if (idb < ida) {
        PatchID tmp = ida;
        ida = idb;
        idb = tmp;
    }

    parents[idb] = ida;
    return ida;
}

PatchID DisjointPatches::find_patch(int x, int y) {
    // return Get2D(labels, x, y);
    return *labels.at(x, y);
}

PatchID DisjointPatches::parent_of(PatchID id) {
    PatchID parent = parents[id];

    if (parent != id) {
        parent = parent_of(parent);
        parents[id] = parent;
    }

    return parent;
}

int DisjointPatches::size() {
    return patch_count;
}



void DisjointPatches::compute_patches() {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            // int h = Get2D(levels, i, j);
            int h = *levels.at(i, j);

            // Cells at height 0 have been marked below the height cutoff.
            if (h == 0) {
                // *Ptr2D(labels, i, j) = 0;
                *labels.at(i, j) = 0;
                continue;
            }

            // Check if left neighbor is at the same height
            // bool connectLeft = i > 0 && Get2D(levels, i - 1, j) == h;
            bool connectLeft = i > 0 && *levels.at(i - 1, j) == h;
            // Check if top neighbor is at the same height
            // bool connectTop = j > 0 && Get2D(levels, i, j - 1) == h;
            bool connectTop = j > 0 && *levels.at(i, j - 1) == h;

            // Determine patch ID for the current cell.
            PatchID id;
            if (connectLeft && connectTop) {
                // PatchID leftId = Get2D(labels, i - 1, j);
                PatchID leftId = *labels.at(i - 1, j);
                // PatchID topId = Get2D(labels, i, j - 1);
                PatchID topId = *labels.at(i, j - 1);
                id = union_patches(leftId, topId);
            }
            else if (connectLeft) {
                // id = Get2D(labels, i - 1, j);
                id = *labels.at(i - 1, j);
            }
            else if (connectTop) {
                // id = Get2D(labels, i, j - 1);
                id = *labels.at(i, j - 1);
            }
            else {
                id = make_patch(i, j, h);
            }

            // *Ptr2D(labels, i, j) = id;
            *labels.at(i, j) = id;
        }
    }

    // Re-label the patch grid with their top-most parent ids.
    // Map top-level parents to linearly increasing id.
    std::map<int, int> idMap;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // int* ptr = Ptr2D(labels, i, j);
            int* ptr = labels.at(i, j);
            if (*ptr == 0) {
                continue;
            }

            int parent = parent_of(*ptr);
            if (idMap.find(parent) == idMap.end()) {
                int id = 1 + idMap.size();
                idMap[parent] = id;
            }

            *ptr = (int) idMap[parent];
        }
    }

    patch_count = idMap.size();
}
