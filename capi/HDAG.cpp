#include "HDAG.hpp"

void create_HDAG(HierarchyData&, PdagData&) {
    // Calculate HAC for every hierarchy
    // adjust patches
    //      - map each patch to a hierarchy
    //      - adjust the cells of each
}

void adjust_patches(struct HierarchyData& hierarchyContext, struct PdagData& pdagContext) {
    std::map<int, Patch>::iterator patch_itr;
    std::vector<int>::iterator hier_itr;

    std::map<int, Patch> *patches_ptr = &(pdagContext.patches);
    std::map<int, Hierarchy> *hierarchies_ptr = &(hierarchyContext.hierarchies);

    Patch *current_patch;

    Centroid hac;

    std::vector<int> *hierarchies;

    for (patch_itr = patches_ptr->begin(); patch_itr != patches_ptr->end(); ++patch_itr) {
        current_patch = &(patch_itr->second);

        hierarchies = &(patch_itr->second.associated_hierarchies);
        std::cout << " Patch id: " << patch_itr->first;
        std::cout << " Hierarchies size: " << patch_itr->second.associated_hierarchies.size() << std::endl;
        for (hier_itr = hierarchies->begin(); hier_itr != hierarchies->end(); ++hier_itr) {
            hac = (*hierarchies_ptr)[*hier_itr].get_HAC();
            current_patch->adjust_hierarchy(*hier_itr, hac);
            // patch.check_closest(currentHAC, hierarchy_ID)
            // if closer, update closest id, update closeset distance
        }

    }

    for (patch_itr = patches_ptr->begin(); patch_itr != patches_ptr->end(); ++patch_itr) {
        std::cout << "Patch ID: " << patch_itr->first << " closest hierarchy id: " << patch_itr->second.get_closest_hierarchy() << std::endl;
    }
}
