#pragma once

#include <vector>
#include <map>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL treeseg_ARRAY_API
#include "numpy/arrayobject.h"

#include "grid.hpp"

typedef int PatchID;

class DisjointPatches {
private:
    PatchID ID_NEXT = 1;
    int width;
    int height;
    int patch_count;
    Grid<int>& levels;
    Grid<int>& labels;
    std::map<PatchID, PatchID> parents;

public:
    DisjointPatches(Grid<int>& levels, Grid<int>& labels);
    PatchID make_patch(int x, int y, int height);
    PatchID union_patches(PatchID a, PatchID b);
    PatchID find_patch(int x, int y);
    PatchID parent_of(PatchID id);
    int size();

    void compute_patches();
    void compute_linear_ids();
};