#pragma once

#include <vector>
#include <map>

#include "numpy/arrayobject.h"

typedef int PatchID;


struct Cell {
    int x;
    int y;
};


class PatchNode {
public:
    PatchID id;
    int height;
};


class DisjointSets {
private:
    PatchID ID_NEXT = 1;

public:
    PyArrayObject* levels;
    PyArrayObject* labels;
    std::map<PatchID, PatchNode> nodes;
    std::map<PatchID, PatchID> parents;

    DisjointSets(PyArrayObject* levels, PyArrayObject* labels);
    PatchID make_patch(int x, int y, int height);
    PatchID union_patches(PatchID a, PatchID b);
    PatchID find_patch(int x, int y);
    PatchID parent_of(PatchID id);
};