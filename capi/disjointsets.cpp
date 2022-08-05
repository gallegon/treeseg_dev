#include "disjointsets.hpp"
#include <iostream>

#define Ptr2D(array, i, j) ((int*) PyArray_GETPTR2(array, i, j))
#define Get2D(array, i, j) (*((int*) PyArray_GETPTR2(array, i, j)))

DisjointSets::DisjointSets(PyArrayObject* levels, PyArrayObject* labels) {
    this->levels = levels;
    this->labels = labels;
}

PatchID DisjointSets::make_patch(int x, int y, int height) {
    PatchNode patch;
    patch.id = ID_NEXT++;
    patch.height = height;
    parents[patch.id] = patch.id;
    // *Ptr2D(labels, x, y) = patch.id;
    return patch.id;
}

PatchID DisjointSets::union_patches(PatchID a, PatchID b) {
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

PatchID DisjointSets::find_patch(int x, int y) {
    return Get2D(labels, x, y);
}

PatchID DisjointSets::parent_of(PatchID id) {
    PatchID parent = parents[id];

    if (parent != id) {
        parent = parent_of(parent);
        parents[id] = parent;
    }

    return parent;
}
