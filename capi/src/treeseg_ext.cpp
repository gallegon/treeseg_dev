#define PY_SSIZE_T_CLEAN
#include "Python.h"
#define PY_ARRAY_UNIQUE_SYMBOL treeseg_ARRAY_API
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <iostream>
#include <map>

#include "Patch.hpp"
#include "Hierarchy.hpp"
#include "disjointpatches.hpp"
#include "HDAG.hpp"
#include "pdalfilter.hpp"
#include "disjointtrees.hpp"
#include "grid.hpp"

#include "debug.hpp"



static PyObject* discretize_points(PyObject* self, PyObject* args) {
    char* filepath_in;
    double resolution;
    int discretization;
    if (!PyArg_ParseTuple(args, "sdi", &filepath_in, &resolution, &discretization)) {
        return NULL;
    }

    using namespace pdal;

    // PointTable holds all of the points, allows accessing points (read/write).
    PointTable table;
    // Register the "default" dimensions we care about.
    table.layout()->registerDim(Dimension::Id::X);
    table.layout()->registerDim(Dimension::Id::Y);
    table.layout()->registerDim(Dimension::Id::Z);
    // Create and register a new dimension "TreeID" of type uint64.
    // table.layout()->assignDim("TreeID", Dimension::Type::Unsigned64);

    StageFactory factory;

    // Create stages
    Stage* reader = factory.createStage("readers.las");
    CustomFilter* filter = dynamic_cast<CustomFilter*>(factory.createStage("filters.customfilter"));
    Stage* writer = factory.createStage("writers.las");

    DPRINT("Created stages");

    Options options;
    options.add("filename", filepath_in);
    reader->setOptions(options);

    DPRINT("-- Running pipeline...");

    filter->withResolution(resolution);
    filter->withDiscretization(discretization);
    filter->withReader(*reader);
    filter->setInput(*reader);
    filter->prepare(table);
    filter->execute(table);

    DPRINT("-- Finished running reader & filter");

    auto grid = filter->getGrid();
    const npy_intp dims[] = {grid->width, grid->height};


    PyArrayObject* discretized_grid = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_INT, grid->data);

    return PyArray_Return(discretized_grid);
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

    Grid<int> grid_levels(width, height, PyArray_DATA(levels));
    Grid<int> grid_labels(width, height, PyArray_DATA(labels));
    
    DisjointPatches ds(grid_levels, grid_labels);
    ds.compute_patches();

    DPRINT("Total number of patches = " << ds.size());

    return PyArray_Return(labels);
}


static PyObject* vector_test(PyObject* self, PyObject* args) {
    PyObject* argGrid;
    PyObject* argLabels;
    PyObject* argWeights;

    // Used to get data from the patch creation step
    struct PdagData pdag;
    struct HierarchyData hierarchyContext;
    std::vector<DirectedWeightedEdge> partitioned_edge_list;

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

    Grid<int> labels(dims[0], dims[1], PyArray_DATA(arrayLabels));
    Grid<int> levels(dims[0], dims[1], PyArray_DATA(arrayGrid));

    create_patches(labels, levels, pdag);
    compute_hierarchies(pdag, hierarchyContext);
    //calculateHAC(pdag, hierarchyContext);
    adjust_patches(hierarchyContext, pdag);
    map_cells_to_hierarchies(hierarchyContext, pdag);

    int mapped_cc = 0;
    for (auto h_itr = hierarchyContext.hierarchies.begin(); h_itr != hierarchyContext.hierarchies.end(); ++ h_itr) {
        mapped_cc += h_itr->second.get_adjusted_cells().size();
    }

    //std::cout << "***************MAPPED CELL COUNT = " << mapped_cc << std::endl;

    float weights[6] = {
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 0))),
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 1))),
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 2))),
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 3))),
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 4))),
        *(static_cast<float*>(PyArray_GETPTR1(arrayParams, 5)))
    };
    create_HDAG(partitioned_edge_list, hierarchyContext, pdag, weights);

    /*
    std::set<int> mapped_hierarchies;

    for (auto it = partitioned_edge_list.begin(); it != partitioned_edge_list.end(); ++it) {
        int parent_id = it->parent;
        int child_id = it->child;
        double weight = it->weight;
        mapped_hierarchies.insert(parent_id);
        mapped_hierarchies.insert(child_id);
    }

    std::set<int> unmapped_patches;
    for (auto it = pdag.patches.begin(); it != pdag.patches.end(); ++it) {
        auto patch_id = it->first;
        auto patch = it->second;
        unmapped_patches.insert(patch_id);
    }
    */
    /*
    for (auto it = mapped_hierarchies.begin(); it != mapped_hierarchies.end(); ++it) {
        auto hierarchy = hierarchyContext.hierarchies.at(*it);
        auto patches = hierarchy.getPatchIDs();
        for (auto pit = patches.begin(); pit != patches.end(); ++pit) {
            unmapped_patches.erase(*pit);
        }
    }


    std::cout << "== UNMAPPED PATCHES" << std::endl;
    int _count = 0;
    for (auto it = unmapped_patches.begin(); it != unmapped_patches.end(); ++it) {
        std::cout << "    " << *it << std::endl;
        if (_count++ > 25) {
            break;
        }
    }
    */

    DPRINT(
        std::endl
        << "== Partitioned Edge List" << std::endl
        << "size: " << partitioned_edge_list.size()
    );

    // std::cout << "Checking for invalid patches!" << std::endl;
    // for (auto pit = pdag.patches.begin(); pit != pdag.patches.end(); ++pit) {
    //     int patch_id = pit->first;
    //     Patch patch = pit->second;
    //     if (patch.associated_hierarchies.size() != 1) {
    //         std::cout << "Patch with multiple hierarchies: " << patch_id << std::endl;
    //     }
    //     if (patch.get_closest_hierarchy() <= 0) {
    //         std::cout << "Patch with invalid closest hierarchy: " << patch_id << " (hierarchy id = " << patch.get_closest_hierarchy() << ")" << std::endl;
    //     }
    // }

    using HierarchyID = int;

    std::map<HierarchyID, std::vector<HierarchyID>> parent_map;
    std::map<HierarchyID, std::vector<HierarchyID>> child_map;

    for (std::vector<DirectedWeightedEdge>::iterator vit = partitioned_edge_list.begin(); vit != partitioned_edge_list.end(); ++vit) {
        int parent_id = vit->parent;
        int child_id = vit->child;
        parent_map[child_id].push_back(parent_id);
        child_map[parent_id].push_back(child_id);
    }

    /*
    for (auto cit = child_map.begin(); cit != child_map.end(); ++cit) {
        HierarchyID child_id = cit->first;
        auto parents = cit->second;
        if (parents.size() > 1) {
            std::cout << "== Child: " << child_id << std::endl;
            for (auto pit = parents.begin(); pit != parents.end(); ++pit) {
                std::cout << "    -- " << *pit << std::endl;
            }
        }
    }
    */

    // Attempt to quick fix missing hierarchies by adding parentless hierarchies to roots
    auto hierarchies = &(hierarchyContext.hierarchies);
    std::unordered_set<int> parentless_hierarchies;

    // Iterate through all the hierarchies and add all the IDs to a set
    for (auto it = hierarchies->begin(); it != hierarchies->end(); ++it) {
        parentless_hierarchies.insert(it->second.get_id());
    }

    for (auto edge : partitioned_edge_list) {
        auto child = edge.child;
        //if (parentless_hierarchies.find(child) == parentless_hierarchies.end()) {
        //    continue;
        //}
        //else {
            parentless_hierarchies.erase(child);
        //}
    }

    struct DirectedWeightedEdge temp;

    for (auto parentless: parentless_hierarchies) {
        temp.parent = parentless;
        temp.child = parentless;
        temp.weight = 0;
        partitioned_edge_list.push_back(temp);
    }
    // for (auto pit = parent_map.begin(); pit != parent_map.end(); ++pit) {
    //     HierarchyID parent_id = pit->first;
    //     auto children = pit->second;
    //     if (children.size() > 1) {
    //         std::cout << "== Parent: " << parent_id << std::endl;
    //         for (auto cit = children.begin(); cit != children.end(); ++cit) {
    //             std::cout << "    -- " << *cit << std::endl; 
    //         }
    //     }
    // }


    DisjointTrees dt;
    for (std::vector<DirectedWeightedEdge>::iterator vit = partitioned_edge_list.begin(); vit != partitioned_edge_list.end(); ++vit) {
        
        /*
        int parent_id = std::get<0>(*vit);
        int child_id = std::get<1>(*vit);
        double weight = std::get<2>(*vit);
        */

        int parent_id = vit->parent;
        int child_id = vit->child;
        double weight = vit->weight;

        if (parent_id == 0 || child_id == 0) {
            std::cout << "Parent, Child :: " << parent_id << ", " << child_id << std::endl;
            continue;
        }

        // Hierarchy parent = hierarchyContext.hierarchies[parent_id];
        // Hierarchy child = hierarchyContext.hierarchies[child_id];

        TreeID tree_parent = dt.tree_from_hierarchy(parent_id);
        TreeID tree_child = dt.tree_from_hierarchy(child_id);
        // std::cout << "Got trees from hierarchies" << std::endl;
        if (tree_parent == 0) {
            tree_parent = dt.make_tree(parent_id);
        }
        
        if (tree_child == 0) {
            tree_child = dt.make_tree(child_id);
        }
        // std::cout << "Made trees" << std::endl;
        // std::cout << "Pre-union: " << tree_parent << ", " << tree_child << std::endl;
        dt.union_trees(tree_parent, tree_child);
        // std::cout << "Unioned trees" << std::endl;
    }


    
    // PyArrayObject* hierarchy_labels = (PyArrayObject*) PyArray_SimpleNew(ndims, dims, NPY_INT);
    PyArrayObject* hierarchy_labels = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT, 0);

    Grid<int> hierarchy_grid(dims[0], dims[1], PyArray_DATA(hierarchy_labels));

    DPRINT("== Roots");
    auto roots = dt.roots();

    // add the found parentless hierarchies roots
    for (auto ph_id : parentless_hierarchies) {
        roots.insert(ph_id);
    }

    for (auto it = roots.begin(); it != roots.end(); ++it) {
        TreeID tree_id = *it;
        DPRINT("  " << *it);
        
        auto hs = dt.hierarchies_from_tree(tree_id);
        std::set<int> patches_for_tree;
        std::set<Cell> cells_for_tree;
        for (auto hit = hs.begin(); hit != hs.end(); hit++) {
            DPRINT("    " << *hit);
            Hierarchy hierarchy = hierarchyContext.hierarchies[*hit];
            auto cells = hierarchy.get_adjusted_cells();
            cells_for_tree.insert(cells.begin(), cells.end());
        }

        for (auto cit = cells_for_tree.begin(); cit != cells_for_tree.end(); ++cit) {
            int x = cit->first;
            int y = cit->second;
            // *Ptr2D(hierarchy_labels, x, y) = tree_id;
            *hierarchy_grid.at(x, y) = tree_id;
        }
    }

    // std::cout << std::endl;

    // Py_RETURN_NONE;
    return PyArray_Return(hierarchy_labels);
}

static PyMethodDef treesegMethods[] = {
    {"discretize_points", discretize_points, METH_VARARGS, "Discretize a set of points into a 2d grid."},
    {"label_grid", label_grid, METH_VARARGS, "Label contiguous patches"},
    {"vector_test", vector_test, METH_VARARGS, "Neighbors on neighbors"},
    {NULL, NULL, NULL, NULL}
};

static PyModuleDef treesegModule = {
    PyModuleDef_HEAD_INIT,
    "treeseg_ext",
        "Tree segmentation documentation.\n"
        "Now it's in C!",
    -1,
    treesegMethods
};

PyMODINIT_FUNC PyInit_treeseg_ext(void) {
    PyObject* module;
    // Normal Python functions already specified in treesegModule definition.
    module = PyModule_Create(&treesegModule);

    import_array();
    import_umath();

    return module;
}
