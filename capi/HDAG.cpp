#include "HDAG.hpp"


// MIE = Maximal inbound edge
void init_MIE_map(std::map<int, MaxInboundEdge>& MIE_map) {
    int parent, child;
    double weight;
    std::map<int, MaxInboundEdge>::iterator it;

    // For this map the key is the child, and the first value of the pair is the
    // Parent.  The second value of the pair is the weight
    // MIE_map[child] = (parent, weight)
    for (it = MIE_map.begin(); it != MIE_map.end(); ++it) {
        child = it->first;
        parent = it->second.first;
        weight = it->second.second;
    }
}


/*
Return the direction of an edge between two hierarchies in the HDAG.
This function references section 2.2.4 of "The Paper".
*/
HierarchyPair get_direction(Hierarchy* h1, Hierarchy* h2) {
    std::vector<double> stats;
    std::vector<double>::iterator it;

    /*
    The direction is determined in this order:

    1.) Determine orientation based on height
        (higher hierarchy -> lower hierarchy)
    2.) If heights are equal determine based on the cells counts of each
        hierarchy (larger cell count -> smaller cell count)
    3.) If cell counts are equal determine the direction based on the cell
        counts of the top patches for their respective hierarchies.
        (larger top patch -> smaller top patch)
    4.) If the top patch cell counts are equal, determine the direction based
        on the orientation of their respective height adjusted centroids (HACs)
        a.) First try to find the rightmost hierarchy (larger "x-coordinate")
        b.) if equal try to find the bottom-most hierarchy (Larger "y-coordinate")

    TODO: explain the vector stuff -- too tire to do right Now
    basically subtract each statistic and look for the first result that is
    non zero and set the direction based on the sign of the result of the
    subtraction--it implies that one hierarchy has a greater value than the
    other...

    */
    stats.push_back(h1->get_height() - h2->get_height());
    stats.push_back(h1->get_cell_count() - h2->get_cell_count());
    stats.push_back(h1->get_tp_cell_count() - h2->get_tp_cell_count());
    stats.push_back(h1->get_HAC().first - h2->get_HAC().first);
    stats.push_back(h1->get_HAC().second - h2->get_HAC().second);

    for (it = stats.begin(); it != stats.end(); ++it) {
        // If the *it is not equal to zero, then something had a greater value
        if (*it != 0) {
            if (*it > 0) {
                return std::make_pair(h1->get_id(), h2->get_id());
            }
            else {
                return std::make_pair(h2->get_id(), h1->get_id());
            }
        }
    }

}

void create_HDAG(std::vector<DirectedWeightedEdge>& edges,
                 HierarchyData& hierarchy_context, PdagData& pdag_context,
                 PyArrayObject* weights) {
    // Calculate HAC for every hierarchy
    // adjust patches
    //      - map each patch to a hierarchy
    //      - adjust the cells of each
    Hierarchy *h1, *h2;
    int h1_id, h2_id;
    
    double weight_threshold = *((float *) PyArray_GETPTR1(weights, 5));

    //std::vector<DirectedWeightedEdge> edges;

    // Size of patches set for each hierarchy, and for the shared cell cell count
    int h1_p_size, h2_p_size, shared_cell_count;

    // nd->"Node Depth", ld->"Level Depth"
    int curr_min_nd, curr_min_ld, min_ld, min_nd;
    std::vector<int> h1_patches, h2_patches, patch_intersection;

    // Will be used to keep track of and compare depth statistics
    std::pair<int, int> h1_depths, h2_depths;
    int h1_ld, h2_ld, h1_nd, h2_nd;

    // patch ID iterator
    std::vector<int>::iterator p_it;

    // Get a reference to the connected hierarchies, also create an iterator
    std::set<std::pair<int, int>> *ch_ptr = &(hierarchy_context.connected_hierarchies);
    std::set<std::pair<int, int>>::iterator ch_it;

    // Pointer to the hierarchy map
    std::map<int, Hierarchy> *h_ptr = &(hierarchy_context.hierarchies);

    // Pointer to patches map
    std::map<int, Patch> *p_ptr = &(pdag_context.patches);


    // Keep track of the maximal inbound edge for each
    // node.  This is effectively partitioning the
    // graph at the same time as creation
    std::map<int, MaxInboundEdge> maximal_inbound_edges;
    std::map<int, MaxInboundEdge>::iterator mie_it;

    // Iterate through every connected hierarchy pair
    for (ch_it = ch_ptr->begin(); ch_it != ch_ptr->end(); ++ch_it) {
        h1_id = ch_it->first;
        h2_id = ch_it->second;

        // Get references to the Hierarchies with associated IDs h1_id and h2_id
        h1 = &(h_ptr->at(h1_id));
        h2 = &(h_ptr->at(h2_id));



        h1_patches = h1->getPatchIDs();
        h2_patches = h2->getPatchIDs();

        h1_p_size = h1_patches.size();
        h2_p_size = h2_patches.size();

        std::sort(h1_patches.begin(), h1_patches.end());
        std::sort(h2_patches.begin(), h2_patches.end());

        // Clear the intersection set to be used for the current patches
        patch_intersection.clear();

        // Compute the set intersection of patch IDs
        std::set_intersection(h1_patches.begin(), h1_patches.end(),
                              h2_patches.begin(), h2_patches.end(),
                              std::back_inserter(patch_intersection));

        // "Zero out" each of these values for use with the current hierarchy pair
        shared_cell_count = 0;
        min_ld = std::numeric_limits<int>::max();
        min_nd = std::numeric_limits<int>::max();

        /*
        This for loop computes the minimum node depths and level depths,
        as well as the shared cell count for each of the two hierarchies.

        Honestly, this can be it's own function that returns a tuple of ints,
        but for the time being just implementing it here.
        */
        for (p_it = patch_intersection.begin(); p_it != patch_intersection.end(); ++ p_it) {
            shared_cell_count += p_ptr->at(*p_it).getCellCount();
            h1_depths = h1->getPatchDepths(*p_it);
            h2_depths = h2->getPatchDepths(*p_it);

            // Assign depths so that they can be compared easier
            h1_ld = h1_depths.first;
            h1_nd = h1_depths.second;
            h2_ld = h2_depths.first;
            h2_nd = h2_depths.second;

            curr_min_ld = std::min(h1_ld, h2_ld);
            curr_min_nd = std::min(h1_nd, h2_nd);

            if ((curr_min_ld < min_ld) && (curr_min_ld != 0)) {  min_ld = curr_min_ld;  }
            if ((curr_min_nd < min_nd) && (curr_min_nd != 0)) {  min_nd = curr_min_nd;  }
        }


        double top_distance, centroid_distance;

        top_distance = get_distance(h1->get_TPC(), h2->get_TPC());
        centroid_distance = get_distance(h1->get_HAC(), h2->get_HAC());

        std::cout << std::endl;
        std::cout << "H1 ID: " << h1->get_id() << " || " << "H2 ID: " << h2->get_id() << std::endl;
        std::cout << "H1 HAC: (" << h1->get_HAC().first << ", " << h1->get_HAC().second << ") || ";
        std::cout << "H2 HAC: (" << h2->get_HAC().first << ", " << h2->get_HAC().second << ")" << std::endl;
        std::cout << "CD: " << centroid_distance << std::endl;

        /*
        The following represent the weights given in the initial parameters
        The weights are as follows:
        Level Depth, Node Depth, Shared Ratio, Top Distance, and Centroid Distance

        They are expected to be in the following order in the inital array:
        {ld_weight, nd_weight, sr_weight, td_weight, cd_weight}
        */
        double ld_weight, nd_weight, sr_weight, td_weight, cd_weight;

        double ld_score, nd_score, sr_score, td_score, cd_score;

        double edge_weight;

        // Get the weights from the NumPy array object
        ld_weight = *((float *) PyArray_GETPTR1(weights, 0));
        nd_weight = *((float *) PyArray_GETPTR1(weights, 1));
        sr_weight = *((float *) PyArray_GETPTR1(weights, 2));
        td_weight = *((float *) PyArray_GETPTR1(weights, 3));
        cd_weight = *((float *) PyArray_GETPTR1(weights, 4));
        std::cout << "Weights: " << ld_weight << ", " << nd_weight << ", " << sr_weight << ", " << td_weight << ", " << cd_weight << std::endl;
        std::cout << "Stats:   " << min_ld << ", " << min_nd << ", " << shared_cell_count << ", " << top_distance << ", " << centroid_distance << std::endl;
        ld_score = 1 / min_ld;
        nd_score = 1 / min_nd;
        sr_score = shared_cell_count / (h1->get_cell_count() + h2->get_cell_count() - shared_cell_count);
        td_score = 1 / top_distance;
        cd_score = 1 / centroid_distance;


        HierarchyPair edge = get_direction(h1, h2);

        std::cout<< "Directed Edge: (" << edge.first << ", " << edge.second << ") weight: " << edge_weight << std::endl;

        edge_weight = (ld_weight * ld_score) + (nd_weight * nd_score) +
                      (sr_weight * sr_score) + (td_weight * td_score) +
                      (cd_weight * cd_score);
        //std::cout << "(" << h1_id << ", " << h2_id << ") weight: " << edge_weight << std::endl;

        std::cout << std::endl;

        int parent = edge.first;
        int child = edge.second;

        double curr_max_weight;

        // If edge weight greater than weight_threshold...
        if (edge_weight > weight_threshold) {
            // if edge is maximal_inbound_edge
            mie_it = maximal_inbound_edges.find(child);
            if (mie_it != maximal_inbound_edges.end()) {
                curr_max_weight = mie_it->second.second;
                if (edge_weight > curr_max_weight) {
                    mie_it->second.first = parent;
                    mie_it->second.second = edge_weight;
                }
            }
            else {
                maximal_inbound_edges.insert(std::make_pair(child, std::make_pair(parent, edge_weight)));
            }
        }

    }

    std::cout << std::endl;
    std::cout << "Maximal inbound edges list" << std::endl;
    // Create a list of edges for the partitioend graph
    int parent, child;
    double weight;

    for (mie_it = maximal_inbound_edges.begin(); mie_it != maximal_inbound_edges.end(); ++mie_it) {
        parent = mie_it->second.first;
        child = mie_it->first;
        weight = mie_it->second.second;
        edges.push_back(std::make_tuple(parent, child, weight));
        std::cout << "(" << parent << ", " << child << ") Weight: " << weight << std::endl;
    }
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
        //std::cout << " Patch id: " << patch_itr->first;
        //std::cout << " Hierarchies size: " << patch_itr->second.associated_hierarchies.size() << std::endl;
        for (hier_itr = hierarchies->begin(); hier_itr != hierarchies->end(); ++hier_itr) {
            hac = (*hierarchies_ptr)[*hier_itr].get_HAC();
            current_patch->adjust_hierarchy(*hier_itr, hac);

            // patch.check_closest(currentHAC, hierarchy_ID)
            // if closer, update closest id, update closeset distance

        }

    }

    /*
    std::map<int, Hierarchy>::iterator hier_map_it;


    for (hier_map_it = hierarchies_ptr->begin(); hier_map_it != hierarchies_ptr->end(); ++hier_map_it) {
        cells = hier_map_it->second.get_adjusted_cells();
        std::cout << "Hierachy ID: " << hier_map_it->first << " Cells: ";
        for (c_it = cells.begin(); c_it != cells.end(); ++c_it) {
            std::cout << "(" << c_it->first << ", " << c_it->second << ") ";
        }
        std::cout << std::endl;
    }

    for (patch_itr = patches_ptr->begin(); patch_itr != patches_ptr->end(); ++patch_itr) {
        std::cout << "Patch ID: " << patch_itr->first << " closest hierarchy id: " << patch_itr->second.get_closest_hierarchy() << std::endl;
    }
    */
}

void map_cells_to_hierarchies(struct HierarchyData& hierarchyContext, struct PdagData& pdagContext) {
    std::map<int, Patch>::iterator patch_itr;

    std::vector<Cell>::iterator c_itr;

    std::vector<Cell> cells;

    std::map<int, Patch> *patches = &(pdagContext.patches);
    std::map<int, Hierarchy> *hierarchies = &(hierarchyContext.hierarchies);

    Hierarchy *hierarchy;
    int hierarchy_id;

    for (patch_itr = patches->begin(); patch_itr != patches->end(); ++patch_itr) {
        hierarchy_id = patch_itr->second.get_closest_hierarchy();
        cells = patch_itr->second.get_cells();
        hierarchy = &(hierarchies->at(hierarchy_id));
        hierarchy->add_adjusted_patch_id(patch_itr->first);

        for (c_itr = cells.begin(); c_itr != cells.end(); ++c_itr) {
            hierarchy->add_adjusted_cell(*c_itr);
        }
    }

    std::map<int, Hierarchy>::iterator hier_map_it;


    for (hier_map_it = hierarchies->begin(); hier_map_it != hierarchies->end(); ++hier_map_it) {
        cells = hier_map_it->second.get_adjusted_cells();
        std::cout << "Hierachy ID: " << hier_map_it->first << " Cells: ";
        for (c_itr = cells.begin(); c_itr != cells.end(); ++c_itr) {
            std::cout << "(" << c_itr->first << ", " << c_itr->second << ") ";
        }
        std::cout << std::endl;
    }
}
