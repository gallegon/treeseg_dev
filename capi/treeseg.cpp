#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <iostream>
#include <map>

#include "Patch.hpp"
#include "Hierarchy.hpp"
#include "disjointsets.hpp"


static PyObject* label_grid(PyObject* self, PyObject* args) {
    PyObject* argGrid;
    if (!PyArg_ParseTuple(args, "O", &argGrid)) {
        return NULL;
    }

    PyArrayObject* levels = (PyArrayObject*) PyArray_FROM_OTF(argGrid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (levels == NULL) {
        return NULL;
    }

    int ndims = PyArray_NDIM(levels);
    npy_intp* dims = PyArray_DIMS(levels);
    int width = dims[0];
    int height = dims[1];

    PyArrayObject* labels = (PyArrayObject*) PyArray_SimpleNew(ndims, dims, NPY_INT);


    // Compute Patch sets
    DisjointSets ds(levels, labels);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int h = Get2D(levels, i, j);

            // Cells at height 0 have been marked below the height cutoff.
            if (h == 0) {
                *Ptr2D(labels, i, j) = 0;
                continue;
            }
            // Check if left neighbor is at the same height
            bool connectLeft = i > 0 && Get2D(levels, i - 1, j) == h;
            // Check if top neighbor is at the same height
            bool connectTop = j > 0 && Get2D(levels, i, j - 1) == h;

            // Determine patch ID for the current cell.
            PatchID id;
            if (connectLeft && connectTop) {
                PatchID leftId = Get2D(ds.labels, i - 1, j);
                PatchID topId = Get2D(ds.labels, i, j - 1);
                id = ds.union_patches(leftId, topId);
            }
            else if (connectLeft) {
                id = Get2D(ds.labels, i - 1, j);
            }
            else if (connectTop) {
                id = Get2D(ds.labels, i, j - 1);
            }
            else {
                id = ds.make_patch(i, j, h);
            }

            *Ptr2D(ds.labels, i, j) = id;
        }
    }

    // Re-label the patch grid with their top-most parent ids.
    // Map top-level parents to linearly increasing id.
    std::map<int, int> idMap;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int* ptr = Ptr2D(labels, i, j);
            if (*ptr == 0) {
                continue;
            }
            int parent = ds.parent_of(*ptr);
            if (idMap.find(parent) == idMap.end()) {
                int id = 1 + idMap.size();
                idMap[parent] = id;
            }
            *ptr = (int) idMap[parent];
        }
    }

    std::cout << "Total number of patches = " << idMap.size() << std::endl;

    return PyArray_Return(labels);
}


static PyObject* vector_test(PyObject* self, PyObject* args) {
    PyObject* argGrid;
    PyObject* argLabels;
    PyObject* argWeights;

    // Used to get data from the patch creation step
    struct PdagData pdag;
    struct HierarchyData hierarchyContext;

    if (!PyArg_ParseTuple(args, "OOO", &argGrid, &argLabels, &argWeights)) {
        return NULL;
    }

    PyArrayObject* arrayGrid = (PyArrayObject*) PyArray_FROM_OTF(argGrid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* arrayLabels = (PyArrayObject*) PyArray_FROM_OTF(argLabels, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* arrayParams = (PyArrayObject*) PyArray_FROM_OTF(argWeights, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (arrayGrid == NULL || arrayLabels == NULL || arrayParams == NULL) {
        return NULL;
    }

    int ndims = PyArray_NDIM(arrayGrid);
    npy_intp* dims = PyArray_DIMS(arrayGrid);
    int ddims[] = {dims[0], dims[1]};


    // Used to get data from the patch creation step
    //struct PdagData pdag;

    create_patches(arrayLabels, arrayGrid, ddims, pdag);
    compute_hierarchies(pdag, hierarchyContext);

    Py_RETURN_NONE;
}

static PyMethodDef treesegMethods[] = {
    {"label_grid", label_grid, METH_VARARGS, "Label contiguous patches"},
    {"vector_test", vector_test, METH_VARARGS, "Neighbors on neighbors"},
    {NULL, NULL, NULL, NULL}
};

static PyModuleDef treesegModule = {
    PyModuleDef_HEAD_INIT,
    "treeseg",
        "Tree segmentation documentation.\n"
        "Now it's in C!",
    -1,
    treesegMethods
};

PyMODINIT_FUNC PyInit_treeseg(void) {
    PyObject* module;
    // Normal Python functions already specified in treesegModule definition.
    module = PyModule_Create(&treesegModule);

    import_array();
    import_umath();

    return module;
}
