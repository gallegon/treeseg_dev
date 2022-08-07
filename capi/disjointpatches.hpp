#pragma once

#include <vector>
#include <map>

#include "numpy/arrayobject.h"

typedef int PatchID;

class DisjointPatches {
private:
    PatchID ID_NEXT = 1;
    int width;
    int height;
    int patch_count;
    PyArrayObject* levels;
    PyArrayObject* labels;
    std::map<PatchID, PatchID> parents;

public:
    DisjointPatches(PyArrayObject* levels, PyArrayObject* labels);
    PatchID make_patch(int x, int y, int height);
    PatchID union_patches(PatchID a, PatchID b);
    PatchID find_patch(int x, int y);
    PatchID parent_of(PatchID id);
    int size();

    void compute_patches();
    void compute_linear_ids();
};