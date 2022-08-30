#include "disjointtrees.hpp"

DisjointTrees::DisjointTrees() {
    
}

TreeID DisjointTrees::make_tree(int hierarchy_id) {
    TreeID id = ID_NEXT++;
    parents[id] = id;
    hierarchy_map[hierarchy_id] = id;
    return id;
}

TreeID DisjointTrees::union_trees(TreeID ta, TreeID tb) {
    TreeID ida = parent_of(ta);
    TreeID idb = parent_of(tb);

    // std::cout << "Got parents for union: " << ida << ", " << idb << std::endl;

    if (ida == idb) {
        return ida;
    }

    if (idb < ida) {
        TreeID tmp = ida;
        ida = idb;
        idb = tmp;
    }

    parents[idb] = ida;
    return ida;
}

TreeID DisjointTrees::parent_of(TreeID id) {
    // std::cout << "Attempting to get parent_of " << id << std::endl;
    TreeID parent = parents[id];
    // std::cout << "Got parent: " << parent << std::endl;

    if (parent != id) {
        parent = parent_of(parent);
        parents[id] = parent;
    }

    return parent;
}

TreeID DisjointTrees::tree_from_hierarchy(int id) {
    auto maybe_id = hierarchy_map.find(id);
    if (maybe_id == hierarchy_map.end()) {
        return 0;
    }
    return maybe_id->second;
}

std::set<TreeID> DisjointTrees::roots() {
    // A root is a Tree whose parent is itself.
    std::set<TreeID> root_set;
    for (auto tree_it = parents.begin(); tree_it != parents.end(); ++tree_it) {
        if (tree_it->first == tree_it->second) {
            root_set.insert(tree_it->first);
        }
    }
    return root_set;
}

std::set<int> DisjointTrees::hierarchies_from_tree(TreeID id) {
    std::set<int> hierarchies;
    for (auto h_it = hierarchy_map.begin(); h_it != hierarchy_map.end(); ++h_it) {
        int hierarchy = h_it->first;
        TreeID parent = parent_of(h_it->second);
        if (parent == id) {
            hierarchies.insert(hierarchy);
        }
    }
    return hierarchies;
}
