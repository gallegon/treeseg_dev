#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL treeseg_ARRAY_API
#include "numpy/arrayobject.h"

typedef int TreeID;

class DisjointTrees {
private:
    TreeID ID_NEXT = 1;
    PyArrayObject* tree_labels;
    std::map<TreeID, TreeID> parents;
    std::map<int, TreeID> hierarchy_map;
    
public:
    DisjointTrees();
    TreeID make_tree(int hierarchy_id);
    TreeID union_trees(TreeID ta, TreeID tb);
    TreeID parent_of(TreeID id);
    TreeID tree_from_hierarchy(int hierarchy_id);
    std::set<TreeID> roots();
    std::set<int> hierarchies_from_tree(TreeID id);
};
