#include "Hierarchy.hpp"
#include <chrono>

Hierarchy::Hierarchy(int id, int level) {
    this->id = id;
    this->height = level;
    this->cellCount = 0;
}


void Hierarchy::add_patch(int patchID, Patch patch, std::pair<int, int> depths) {
    (this->patches).insert(std::make_pair(patchID, patch));
    (this->patchDepthMap).insert(std::make_pair(patchID, depths));
}

void Hierarchy::add_patchID(int patchID, std::pair<int, int> depths) {
    this->patchIDs.push_back(patchID);
    (this->patchDepthMap).insert(std::make_pair(patchID, depths));
}

std::vector<int> Hierarchy::getPatchIDs() {
    return this->patchIDs;
}

std::pair<int, int> Hierarchy::getPatchDepths(int patchID) {
    return (this->patchDepthMap).at(patchID);
}

void Hierarchy::setHAC(double x, double y)
{
    this->heightAdjustedCentroid = std::make_pair(x, y);
}

void Hierarchy::setCellCount(int cellCount)
{
    this->cellCount = cellCount;
}


void calculateHAC(struct PdagData& pdagContext, struct HierarchyData& hierarchyContext) {
    std::map<int, Hierarchy>::iterator hierIt;
    std::vector<int>::iterator patchItr;
    std::pair<double, double> patchCentroid;
    std::vector<int> patchIDs;

    double hacNumeratorX = 0, hacNumeratorY = 0, hacDenominator = 0;
    double centroidX, centroidY, hacConstant, hacX, hacY;
    int cellCount, heightDiff, hierarchyID, hierarchyCellCount = 0;
    std::pair<double, double> heightAdjustedCentroid;

    // Initialize some patch to use as the current patch.
    Patch patch(0, 0);
    for (hierIt = hierarchyContext.hierarchies.begin(); hierIt != hierarchyContext.hierarchies.end(); ++hierIt) {
        hierarchyID = hierIt->first;
        patchIDs = hierarchyContext.hierarchies.at(hierarchyID).getPatchIDs();

        // this is the meat of the height adjusted centroid calculation -- 
        // review 2.2.4 (6) from "The Paper".  This performs the summations that are
        // described in the equation mentioned.
        hierarchyCellCount = 0;
        hacNumeratorX = 0; 
        hacNumeratorY = 0; 
        hacDenominator = 0;
        
        for (patchItr = patchIDs.begin(); patchItr != patchIDs.end(); ++patchItr) {
            // This loop inspects each patch in a hierarchy to compute the height adjusted
            // centroid and the 
            patch = pdagContext.patches.at(*patchItr);
            cellCount = patch.getCellCount();
            patchCentroid = patch.getCentroid();

            // These represent the Patch centroid's elements (x, y)
            centroidX = patchCentroid.first;
            centroidY = patchCentroid.second;

            // heightDiff is the difference in height between the current patch and the 
            // hierarchy's top patch.
            // hacConstant is a constant that is defined as the cell count of the current
            // patch multiplied by ().  This constant gets used in the numerator and denominator.
            heightDiff = hierIt->second.getPatchDepths(*patchItr).first;
            hacConstant = cellCount * (heightDiff + 1);

            // This sum will be divided by the denominator sum to give the hierarchy's
            // height adjusted centroid
            hacNumeratorX += hacConstant * centroidX;
            hacNumeratorY += hacConstant * centroidY;

            hacDenominator += hacConstant;

            // Increment the hierarchy's cell count by the number of cells in this patch
            hierarchyCellCount += cellCount;
        }

        hacX = hacNumeratorX / hacDenominator;
        hacY = hacNumeratorY / hacDenominator;

        // Set the height-adjusted centroid for the hierarchy
        hierIt->second.setHAC(hacX, hacY);
        hierIt->second.setCellCount(hierarchyCellCount);
    }
}

void compute_hierarchies(struct PdagData& pdagContext, struct HierarchyData& hierarchyContext) {
    // map from a hierarchy id to a Hierarchy object
    std::map<int, Hierarchy> hierarchies;

    //DirectedGraph g = pdagContext.graph;
    //std::map<int, Patch> patches = pdagContext.patches;

    // Use this dictionary to keep track of patches that have no parent.
    // These will be the "local maxima" patches
    //std::map<int, Patch> parentless_patches = pdagContext.parentless_patches;

    std::cout << "Num of parentless patches = " << pdagContext.parentless_patches.size() << std::endl;


    typedef boost::graph_traits<DirectedGraph>::vertex_descriptor vertex_descriptor;
    typedef boost::property_map<DirectedGraph, boost::vertex_index_t>::type IdMap;

    std::vector< int > d(num_vertices(pdagContext.graph));


    std::vector<vertex_descriptor> pred(boost::num_vertices(pdagContext.graph));
    boost::iterator_property_map<std::vector<vertex_descriptor>::iterator,
        IdMap,
        vertex_descriptor,
        vertex_descriptor&>
        predmap(pred.begin(), get(boost::vertex_index, pdagContext.graph));


    std::vector<int> distvector(num_vertices(pdagContext.graph));
    boost::iterator_property_map<std::vector<int>::iterator,
        IdMap,
        int,
        int&>
        distmap_vect(distvector.begin(), get(boost::vertex_index, pdagContext.graph));


    // Some debug/trace variables.
    int count = 0;
    int total_count = pdagContext.parentless_patches.size() - 1;
    int one_percent_amount = floor(total_count * 0.01);
    int next_count = one_percent_amount;
    int total_reachable = 0;

    // These are for hierarchy creation
    int hierarchy_id = 1;
    int hierarchy_level, nodeDepth, levelDepth;

    std::map<int, Patch>::iterator it;




    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (it = pdagContext.parentless_patches.begin(); it != pdagContext.parentless_patches.end(); ++it) {

        int vertex_id = it->first;

        if (vertex_id < 0 || vertex_id > pdagContext.patches.size() - 1) {
            std::cout << "WORKING WITH VERTEX: " << vertex_id << std::endl;
        }

        // Create a new Hierarchy object for each of the parentless nodes
        hierarchy_level = (pdagContext.patches.at(vertex_id)).get_level();
        (pdagContext.patches.at(vertex_id)).add_hierarchy(hierarchy_id, hierarchyContext.connected_hierarchies);

        Hierarchy h(hierarchy_id, hierarchy_level);
        hierarchies.insert(std::pair<int, Hierarchy>(hierarchy_id, h));

        vertex_descriptor s = vertex(vertex_id, pdagContext.graph);


        dijkstra_shortest_paths(pdagContext.graph, s,
            predecessor_map(predmap)
            .distance_map(distmap_vect));
        
        // -- Reachable patches
        boost::graph_traits<DirectedGraph>::vertex_iterator vi, vend;
        for (boost::tie(vi, vend) = vertices(pdagContext.graph); vi != vend; ++vi) {
            if (distvector[*vi] != 2147483647) {
                /*
                Level Depth: The difference in height between a patch and a Hierarchy top
                
                Node Depth: The minimum number of nodes needed to reach a given patch from a 
                Hierarchy top.

                See section 2.2.4. Weighted Graph of "The Paper" 
                */

                auto this_patch = pdagContext.patches.find((int)*vi);
                if (this_patch == pdagContext.patches.end()) {
                    std::cout << "Patch not found: " << this_patch->second.get_id() <<  " || " << (int) *vi << std::endl;
                    continue;
                }
                levelDepth = hierarchy_level - this_patch->second.get_level();
                nodeDepth = (int)distvector[*vi];
                h.add_patchID((int) *vi, std::make_pair(levelDepth, nodeDepth));
                total_reachable += 1;
            }
        }

        //Print the reachable patches
        /*
        std::vector<int> reachablePatches = h.getPatchIDs();

        std::vector<int>::iterator h_it;


        
        
        std::cout << "Hierarchy ID: " << hierarchy_id << std::endl;
        std::cout << "Reachable Patches by ID: ";
        for (h_it = reachablePatches.begin(); h_it != reachablePatches.end(); ++h_it) {
            std::cout << *h_it << ", ";
        }

        std::cout << std::endl;
        */
        //end test print


        // For debugging/tracing purposes; print an update every 1% of parentless patches processed.
        if (count >= next_count || count == total_count) {
            int p = ((float) count / total_count) * 100;
            std::cout << "Patch ID: " << vertex_id << " :: " << (count + 1) << "/" << pdagContext.parentless_patches.size() << " = " << p << "%" << std::endl;
            auto step_time = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count()) / 1000000.0;
            std::cout << "    -- Time since last update (seconds): " << step_time << std::endl;
            next_count += one_percent_amount;
            begin = std::chrono::steady_clock::now();
        }
        count += 1;
        hierarchy_id++;
    }

    std::cout << "Total reachable nodes from parentless patches = " << total_reachable << std::endl;
    std::cout << "Number of parentless patches = " << pdagContext.parentless_patches.size() << std::endl;
    std::cout << "Average reachable nodes per parentless patch = " << ((float) total_reachable / pdagContext.parentless_patches.size()) << std::endl;
    std::cout << std::endl;
}
