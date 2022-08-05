#include "Patch.hpp"

// For debugging elapsed time
#include <chrono>

/*

Class definition for a Patch.  A contigous patch of equal level cells in the
discretized grid.

*/
Patch::Patch(int id) {
    this->id = id;
    this->cell_count = 0;
}

/*

Add a cell to a patch.  Everytime this function is called 3 things happen:
    1. The index of the cell (i, j)->(row, col) is added to the cells vector of
    the patch
    2. The cell_count is incremented by 1
    3. the centroid is re-calculated by update_centroid()

 */
void Patch::add_cell(int i, int j) {
    std::pair<int, int> cell = std::make_pair(i, j);
    this->cells.push_back(cell);
    this->sum_x += i;
    this->sum_y += j;
    (this->cell_count)++;
    this->update_centroid();
}

// Everytime a cell is added this function is called to recalculate the centroid
// of the patch
void Patch::update_centroid() {
    this->centroid.first = this->sum_x / this->cell_count;
    this->centroid.second = this->sum_y / this->cell_count;
}

void Patch::print_cells() {
    std::vector<std::pair<int, int>>::iterator it;
    for (it = this->cells.begin(); it != this->cells.end(); ++it) {
        std::cout << " (" << it->first << ", " << it->second << ") ";
    }
}



void addDirectedNeighbor(std::vector<int>& neighbors,  int neighbor_i, int neighbor_j, int n,int current_id,
                        int current_level, PyArrayObject* labels,
                        PyArrayObject* levels) {
    // get the feature id and level of the neighboring cell
    int neighbor_id = Get2D(labels, neighbor_i, neighbor_j);
    int neighbor_level = Get2D(labels, neighbor_i, neighbor_j);

    // if the id of the feature is different the the current id, we know that
    // two patches are connected.
    if (neighbor_id != current_id) {
        if (neighbor_level > current_level) {
            // set the direction flag towards the higher neighbor
            neighbor_id = neighbor_id * -1;
        }
        neighbors.push_back(neighbor_id);

    }
}

#define TOP -1
#define BOTTOM 1
#define RIGHT 1
#define LEFT -1

std::vector<int> get_neighbors(int i, int j, int* dimensions, PyArrayObject* labels, PyArrayObject* levels) {
    // m is the i size of the array, n is the j size
    int m = dimensions[0];
    int n = dimensions[1];
    int neighbor_id;
    int neighbor_level;
    int neighbor_i, neighbor_j;
    // This is for the valid neighbors of a given cell.  Valid neighbors are
    // calculated below
    std::vector<int> neighbors;

    // TODO: find out a way to not have to cast from system int to npy_intp
    int current_id = Get2D(labels, i, j);
    int current_level = Get2D(labels, i, j);

    // Old way using non-NumPy arrays
    //int current_id = labels[IDX(i, j)];
    //int current_level = levels[IDX(i, j)];

    // get top neighbor
    if ((i - 1) >= 0) {
        neighbor_i = i + TOP;
        neighbor_j = j;

        addDirectedNeighbor(neighbors, neighbor_i, neighbor_j, n, current_id, current_level, labels, levels);
    }
    // get bottom neighboring cell
    if ((i + 1) < m) {
        neighbor_i = i + BOTTOM;
        neighbor_j = j;

        addDirectedNeighbor(neighbors, neighbor_i, neighbor_j, n, current_id, current_level, labels, levels);

    }
    // get left neighbor
    if ((j - 1) >= 0) {
        neighbor_i = i;
        neighbor_j = j + LEFT;

        addDirectedNeighbor(neighbors, neighbor_i, neighbor_j, n, current_id, current_level, labels, levels);
    }

    // get right neighbor
    if ((j + 1) < n) {
        neighbor_i = i;
        neighbor_j = j + RIGHT;

        addDirectedNeighbor(neighbors, neighbor_i, neighbor_j, n, current_id, current_level, labels, levels);
    }

    return neighbors;
}

void create_patches(PyArrayObject* labels, PyArrayObject* levels, int* dimensions, struct TreeSegData* context) {
    std::map<int, Patch> patches;

    // Use this dictionary to keep track of patches that have no parent.
    // These will be the "local maxima" patches
    std::map<int, Patch> parentless_patches;

    //c-style array of data

    // int* levelsData = (int*) PyArray_DATA(levels);
    // int* labelsData = (int*) PyArray_DATA(labels);



    int m = dimensions[0];
    int n = dimensions[1];
    int current_feature;

    DirectedGraph g;

    // iterator for patches map
    std::map<int, Patch>::iterator it;

    // TODO: implement this 2D loop with NumPy's array iterators.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            current_feature = Get2D(labels, i, j);
            // current_feature = (int) *(PyArray_GETPTR2(labels, (npy_intp)i, (npy_intp)j));
            // current_feature = labelsData[IDX(i, j)];

            // Don't compute a patch for a cell with that has id 0
            if (current_feature == 0) {
                continue;
            }
            // check if the feature has already been turned into a patch
            it = patches.find(current_feature);

            // If the feature is already a patch, just add the cell
            if (it != patches.end()) {
                it->second.add_cell(i, j);
            }
            // Else the feature is not a patch
            else {
                // create a patch, then add the cell to it
                Patch p(current_feature);
                p.add_cell(i, j);
                patches.insert(std::pair<int, Patch>(current_feature, p));
                parentless_patches.insert(std::pair<int, Patch>(current_feature, p));
            }

            // Get the current cell's neighbors, then iterate
            std::vector<int> neighbors = get_neighbors(i, j, dimensions, labels, levels);
            std::vector<int>::iterator neigh_it;
            for (neigh_it = neighbors.begin(); neigh_it != neighbors.end(); ++neigh_it) {
                int neighbor = *neigh_it;


                // Build the directed graph.  Direction is determined by the patch's height relative to a
                // neighbor.  If a neighbor in neighbors is negative, then the patch is a child to it's
                // neighbor
                if (neighbor < 0) {
                    boost::add_edge((neighbor * -1), current_feature, 1, g);
                    // Here the current feature is a child patch to some higher level patch.
                    // Therefore this patch cannot be a parentless patch.
                    parentless_patches.erase(current_feature);
                }
                // Here the patch is the parent to some lower patch.
                else {
                    boost::add_edge(current_feature, neighbor, 1, g);
                    parentless_patches.erase(neighbor);
                }
            }
        }
    }



    context->patches = patches;
    context->parentless_patches = parentless_patches;j
    context->PDAG = g;

    std::cout << "Num of parentless patches = " << parentless_patches.size() << std::endl;


    typedef boost::graph_traits<DirectedGraph>::vertex_descriptor vertex_descriptor;
    typedef boost::property_map<DirectedGraph, boost::vertex_index_t>::type IdMap;

    std::vector< int > d(num_vertices(g));


    std::vector<vertex_descriptor> pred(boost::num_vertices(g));
    boost::iterator_property_map<std::vector<vertex_descriptor>::iterator,
        IdMap,
        vertex_descriptor,
        vertex_descriptor&>
        predmap(pred.begin(), get(boost::vertex_index, g));

    std::vector<int> distvector(num_vertices(g));
    boost::iterator_property_map<std::vector<int>::iterator,
        IdMap,
        int,
        int&>
        distmap_vect(distvector.begin(), get(boost::vertex_index, g));

<<<<<<< HEAD
    std::map<int, Patch> reachable_patches;
    int patch_id;

=======
    // Some debug/trace variables.
    int count = 0;
    int total_count = parentless_patches.size() - 1;
    int one_percent_amount = floor(total_count * 0.01);
    int next_count = one_percent_amount;
    int total_reachable = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
>>>>>>> 8110fa2c777b64a370c26228676f3e17c6bdfde2
    for (it = parentless_patches.begin(); it != parentless_patches.end(); ++it) {

        int vertex_id = it->first;
        vertex_descriptor s = vertex(vertex_id, g);
        dijkstra_shortest_paths(g, s,
            predecessor_map(predmap)
            .distance_map(distmap_vect));

<<<<<<< HEAD
        // std::cout << std::endl;

        // std::cout << "Parentless Patch ID: " << vertex_id << std::endl;

        std::cout << "Reachable patches: ";

        // Clear the dictionary
        reachable_patches.clear();

        boost::graph_traits<DirectedGraph>::vertex_iterator vi, vend;
        for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {

            patch_id = *vi;
            //reachable_patches.insert(std::pair<int, Patch>(patch_id, patches.at(patch_id)));

            if (distvector[*vi] != 2147483647)
                std::cout << "Patch[" << *vi << "]-Distance: " << distvector[*vi] << ", ";
                // Insert the patch if reachable
                reachable_patches.insert(std::pair<int, Patch>(patch_id, patches.at(patch_id)));
        }
        std::cout << std::endl;

=======
        // -- Reachable patches
        boost::graph_traits<DirectedGraph>::vertex_iterator vi, vend;
        for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
            if (distvector[*vi] != 2147483647) {
                // std::cout << "Patch[" << *vi << "]-Distance: " << distvector[*vi] << ", ";
                total_reachable += 1;
            }
        }

        // For debugging/tracing purposes; print an update every 1% of parentless patches processed.
        if (count >= next_count || count == total_count) {
            int p = ((float) count / total_count) * 100;
            std::cout << "Patch ID: " << vertex_id << " :: " << (count + 1) << "/" << parentless_patches.size() << " = " << p << "%" << std::endl;
            auto step_time = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count()) / 1000000.0;
            std::cout << "    -- Time since last update (seconds): " << step_time << std::endl;
            next_count += one_percent_amount;
            begin = std::chrono::steady_clock::now();
        }
        count += 1;
>>>>>>> 8110fa2c777b64a370c26228676f3e17c6bdfde2
    }

    std::cout << "Total reachable nodes from parentless patches = " << total_reachable << std::endl;
    std::cout << "Number of parentless patches = " << parentless_patches.size() << std::endl;
    std::cout << "Average reachable nodes per parentless patch = " << ((float) total_reachable / parentless_patches.size()) << std::endl;
    std::cout << std::endl;
}
