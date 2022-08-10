#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <iostream>
#include <map>

#include "Patch.hpp"
#include "Hierarchy.hpp"
#include "disjointpatches.hpp"
#include "pdalfilter.hpp"




static PyObject* label_las(PyObject* self, PyObject* args) {
    char* filename;
    PyObject* argGrid;
    if (!PyArg_ParseTuple(args, "sO", &filename, &argGrid)) {
        return NULL;
    }

    PyArrayObject* grid = (PyArrayObject*) PyArray_FROM_OTF(argGrid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (grid == NULL) {
        return NULL;
    }

    int width = PyArray_DIM(grid, 0);
    int height = PyArray_DIM(grid, 1);

    using namespace pdal;

    StageFactory factory;

    // Create stages
    Stage* reader = factory.createStage("readers.las");
    Stage* filter = factory.createStage("filters.customfilter");
    Stage* writer = factory.createStage("writers.las");

    std::cout << "Created stages" << std::endl;
    
    // Construct the pipeline
    filter->setInput(*reader);
    writer->setInput(*filter);

    Options options;
    options.add("filename", filename);
    reader->setOptions(options);

    Options optWriter;
    optWriter.add("filename", "test_output.las");
    // Adds the header, scale, offset, and vlr from the input .las file to the output .las file.
    optWriter.add("forward", "all");
    writer->setOptions(optWriter);

    PointTable table;
    table.layout()->registerDim(Dimension::Id::X);
    table.layout()->registerDim(Dimension::Id::Y);
    table.layout()->registerDim(Dimension::Id::Z);

    std::cout << "Registered dims" << std::endl;

    writer->prepare(table);

    std::cout << "Prepared table; executing PDAL pipeline" << std::endl;

    writer->execute(table);

    std::cout << "Finished las_label" << std::endl;

    Py_RETURN_NONE;
}

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


    DisjointPatches ds(levels, labels);
    ds.compute_patches();

    std::cout << "Total number of patches = " << ds.size() << std::endl;

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
    {"label_las", label_las, METH_VARARGS, "Label .las files by overlaying a 2d grid of ids, top-down on the points."},
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
