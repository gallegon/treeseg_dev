#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <iostream>

#include "Patch.hpp"


static PyObject* vector_test(PyObject* self, PyObject* args) {
    PyObject* argGrid;
    PyObject* argLabels;
    if (!PyArg_ParseTuple(args, "OO", &argGrid, &argLabels)) {
        return NULL;
    }

    PyArrayObject* arrayGrid = (PyArrayObject*) PyArray_FROM_OTF(argGrid, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* arrayLabels = (PyArrayObject*) PyArray_FROM_OTF(argLabels, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (arrayGrid == NULL || arrayLabels == NULL) {
        goto fail;
    }

    int ndims = PyArray_NDIM(arrayGrid);
    npy_intp* dims = PyArray_DIMS(arrayGrid);
    int ddims[] = {dims[0], dims[1]};

    int* dataGrid = (int*) PyArray_DATA(arrayGrid);
    int* dataLabels = (int*) PyArray_DATA(arrayLabels);

    create_patches(arrayLabels, arrayGrid, ddims);

    // Py_INCREF(Py_None);
    // Py_XDECREF(arrayGrid);
    // Py_XDECREF(arrayLabels);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(arrayGrid);
    Py_XDECREF(arrayLabels);

    return NULL;
}


static PyObject* sample_grid(PyObject* self, PyObject* args) {
    long count;
    PyObject* argArray;
    if (!PyArg_ParseTuple(args, "lO", &count, &argArray)) {
        return NULL;
    }

    std::cout << "Parsed arguments from tuple!" << std::endl;
    std::cout << "Got a long: " << count << std::endl;
    PyArrayObject* array = (PyArrayObject*) PyArray_FROM_OTF(argArray, NPY_INT, NPY_ARRAY_IN_ARRAY);
    std::cout << "Got an array!" << std::endl;
    if (array == NULL || count == NULL) {
        goto fail;
    }


    std::cout << "Parsed arguments into C objects!" << std::endl;

    int ndim = PyArray_NDIM(array);
    npy_intp* dims = PyArray_DIMS(array);
    npy_intp grid_width = dims[0];
    npy_intp grid_height = dims[1];
    int size = PyArray_SIZE(array);
    int* grid = (int*) PyArray_DATA(array);

    std::cout << "Got the grid!" << std::endl;

    npy_intp sample_dims[] = {count, count};
    PyArrayObject* sample_array = (PyArrayObject*) PyArray_SimpleNew(ndim, sample_dims, NPY_INT);
    int* sample = (int*) PyArray_DATA(sample_array);

    std::cout << "Discretized grid sample (N=" << count << ")" << std::endl;

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            int grid_index = j + i * grid_height;
            int cell = grid[grid_index];

            int sample_index = j + i * count;
            sample[sample_index] = cell;
        }
    }

    std::cout << "Finished grid sampling!" << std::endl;

    // Py_DECREF(array);
    // Py_DECREF(array);
    // Py_DECREF(sample_array);
    // Py_XDECREF(array);
    // Py_XDECREF(sample_array);

    return PyArray_Return(sample_array);

fail:
    Py_XDECREF(array);
    Py_XDECREF(sample_array);
    return NULL;
}

// Python compatible function.
static PyObject* array_sum(PyObject* self, PyObject* args) {
    // Parse arguments from Python into C objects.
    PyObject* arg;
    if (!PyArg_ParseTuple(args, "O", &arg))
        return NULL;

    // Interpret the first argument as a numpy array.
    // Specifying the type and any constraints/flags the array has.
    PyObject* arr;
    // Note: PyArray_FROM_OTF increments reference count of arr!
    arr = PyArray_FROM_OTF(arg, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL)
        goto fail;

    // Get the number of dimensions, shape, and size of the numpy array.
    int ndim = PyArray_NDIM(arr);
    npy_intp* shape = PyArray_DIMS(arr);
    int size = PyArray_SIZE(arr);

    // View the raw data of this array.
    int* data = (int*) PyArray_DATA(arr);

    // Perform some computation with the data.
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }

    // Decrement reference count (auto incremented with PyArray_FROM_OTF).
    Py_DECREF(arr);

    // Return a Python representation of the computed value.
    return Py_BuildValue("i", sum);

fail:
    Py_XDECREF(arr);
    return NULL;
}

static PyMethodDef treesegMethods[] = {
    {"sample_grid", sample_grid, METH_VARARGS, "Sample some elements on a 2d grid"},
    {"vector_test", vector_test, METH_VARARGS, "Neighbors on neighbors"},
    // {"array_sum", array_sum, METH_VARARGS, "Sums all the elements of the given array."},
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
