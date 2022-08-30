#include "Patch.hpp"

// For debugging elapsed time
#include <chrono>

/*

Class definition for a Patch.  A contigous patch of equal level cells in the
discretized grid.

*/
Patch::Patch(int id, int level) {
    this->id = id;
    this->level = level;
    this->cell_count = 0;

    // Intialize this to use later when adjusting patches
    this->closest_hierarchy_id = -1;
    this->closest_hierarchy_dist = std::numeric_limits<double>::max();
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
    std::vector<std::pair<int, int> >::iterator it;
    for (it = this->cells.begin(); it != this->cells.end(); ++it) {
        std::cout << " (" << it->first << ", " << it->second << ") ";
    }
}

// This is used for keeping track of which hierarchies are connected.
void Patch::add_hierarchy(int hierarchy_id, std::set<std::pair<int, int> >& connected_hierarchies) {
    std::vector<int>::iterator it;
    this->associated_hierarchies.push_back(hierarchy_id);
    for (it = this->associated_hierarchies.begin(); it != this->associated_hierarchies.end(); ++it) {
        // This check insures that the hierarchies are not connected to
        // themselves.
        if (hierarchy_id != *it) {
            if (hierarchy_id > *it) {
                connected_hierarchies.insert(std::make_pair(hierarchy_id, *it));
            }
            else {
                connected_hierarchies.insert(std::make_pair(*it, hierarchy_id));
            }
        }
    }
}

void Patch::adjust_hierarchy(int hierarchy_id, Centroid hac)
{
    //std::cout << "pX: " << this->centroid.first << " pY: ";
    //std::cout << this->centroid.second << std::endl;
    //std::cout << "hacX: " << hac.first << " hacY: " << hac.second << std::endl;

    double distance = get_distance(this->centroid, hac);
    //std::cout << "distance: " << distance << std::endl;
    // TODO: think about float comparison more
    if (distance < this->closest_hierarchy_dist) {
        this->closest_hierarchy_id = hierarchy_id;
        this->closest_hierarchy_dist = distance;
    }
}

int Patch::get_closest_hierarchy()
{
    return this->closest_hierarchy_id;
}

int Patch::get_level() {
    return (this->level);
}

int Patch::getCellCount() {
    return (this->cell_count);
}

std::pair<double, double> Patch::getCentroid() {
    return (this->centroid);
}

std::vector<Cell> Patch::get_cells() {
    return this->cells;
}
void Patch::operator=(const Patch& patch) {
    this->id = patch.id;
    this->level = patch.level;
    this->cells = patch.cells;
    this->associated_hierarchies = patch.associated_hierarchies;
    this->centroid = patch.centroid;
    this->sum_x = patch.sum_x;
    this->sum_y = patch.sum_y;
    this->cell_count = patch.cell_count;
}
void addDirectedNeighbor(std::vector<int>& neighbors,  int neighbor_i, int neighbor_j, int n,int current_id,
                        int current_level, Grid<int>& labels,
                        Grid<int>& levels) {
    // get the feature id and level of the neighboring cell
    // int neighbor_id = Get2D(labels, neighbor_i, neighbor_j);
    // int neighbor_level = Get2D(levels, neighbor_i, neighbor_j);
    int neighbor_id = labels.get(neighbor_i, neighbor_j);
    int neighbor_level = levels.get(neighbor_i, neighbor_j);

    // if the id of the feature is different the the current id, we know that
    // two patches are connected.  Also check that the neighbor's level is
    // not equal to 0.  We don't want to consider level 0 as patches.
    if (neighbor_id != current_id && neighbor_level != 0) {
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

std::vector<int> get_neighbors(int i, int j, int* dimensions, Grid<int>& labels, Grid<int>& levels) {
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
    // int current_id = Get2D(labels, i, j);
    // int current_level = Get2D(levels, i, j);
    int current_id = labels.get(i, j);
    int current_level = levels.get(i, j);

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

void create_patches(Grid<int>& labels, Grid<int>& levels, struct PdagData& context) {
    //std::map<int, Patch> patches;

    // Use this dictionary to keep track of patches that have no parent.
    // These will be the "local maxima" patches
    //std::map<int, Patch> parentless_patches;

    //c-style array of data

    // int* levelsData = (int*) PyArray_DATA(levels);
    // int* labelsData = (int*) PyArray_DATA(labels);

    // This is an edge list of what will ultimately become the PDAG
    std::set<std::pair<int, int> > connected_patches;
    std::set<std::pair<int, int> >::iterator set_iter;


    int m = levels.width;
    int n = levels.height;
    int current_feature, current_level;
    int dimensions[2] = {m, n};

    DirectedGraph g;

    // iterator for patches map
    std::map<int, Patch>::iterator it;

    // TODO: implement this 2D loop with NumPy's array iterators.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // current_feature = Get2D(labels, i, j);
            // current_level = Get2D(levels, i, j);
            current_feature = labels.get(i, j);
            current_level = levels.get(i, j);
            // current_feature = (int) *(PyArray_GETPTR2(labels, (npy_intp)i, (npy_intp)j));
            // current_feature = labelsData[IDX(i, j)];

            // Don't compute a patch for a cell with that has id 0
            if (current_feature == 0) {
                continue;
            }
            // check if the feature has already been turned into a patch
            it = context.patches.find(current_feature);

            // If the feature is already a patch, just add the cell
            if (it != context.patches.end()) {
                it->second.add_cell(i, j);
            }
            // Else the feature is not a patch
            else {
                // create a patch, then add the cell to it
                Patch p(current_feature, current_level);
                p.add_cell(i, j);
                context.patches.insert(std::pair<int, Patch>(current_feature, p));
                context.parentless_patches.insert(std::pair<int, Patch>(current_feature, p));
                connected_patches.insert(std::make_pair(current_feature, current_feature));
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
                    //boost::add_edge((neighbor * -1), current_feature, 1, g);
                    std::cout << "adding edge (" << (neighbor * -1) << ", " << current_feature << ")" << std::endl;
                    connected_patches.insert(std::make_pair((neighbor * -1), current_feature));
                    // Here the current feature is a child patch to some higher level patch.
                    // Therefore this patch cannot be a parentless patch.
                    context.parentless_patches.erase(current_feature);
                }
                // Here the patch is the parent to some lower patch.
                else {
                    std::cout << "adding edge (" << current_feature << ", " << neighbor << ")" << std::endl;
                    //boost::add_edge(current_feature, neighbor, 1, g);
                    connected_patches.insert(std::make_pair(current_feature, neighbor));
                    context.parentless_patches.erase(neighbor);
                }
            }
        }
    }

    DPRINT("Completed edge list");
    
    // some abstraction for graph creation
    int parent, child;
    // add all the edges to the Boost graph container
    for (set_iter = connected_patches.begin(); set_iter != connected_patches.end(); ++set_iter) {
        parent = set_iter->first;
        child = set_iter->second;
        std::cout << "(" << parent << ", " << child << ")" << std::endl;
        boost::add_edge(parent, child, 1, context.graph);
    }

    // Moved to Hierarchy
    /*
    context.patches = patches;
    context.parentless_patches = parentless_patches;
    context.graph = g;
    */
    /*
    std::cout << "Num of parentless patches = " << context.parentless_patches.size() << std::endl;


    typedef boost::graph_traits<DirectedGraph>::vertex_descriptor vertex_descriptor;
    typedef boost::property_map<DirectedGraph, boost::vertex_index_t>::type IdMap;

    std::vector< int > d(num_vertices(context.graph));


    std::vector<vertex_descriptor> pred(boost::num_vertices(context.graph));
    boost::iterator_property_map<std::vector<vertex_descriptor>::iterator,
        IdMap,
        vertex_descriptor,
        vertex_descriptor&>
        predmap(pred.begin(), get(boost::vertex_index, context.graph));

    std::vector<int> distvector(num_vertices(context.graph));
    boost::iterator_property_map<std::vector<int>::iterator,
        IdMap,
        int,
        int&>
        distmap_vect(distvector.begin(), get(boost::vertex_index, context.graph));

    // Some debug/trace variables.
    int count = 0;
    int total_count = parentless_patches.size() - 1;
    int one_percent_amount = floor(total_count * 0.01);
    int next_count = one_percent_amount;
    int total_reachable = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (it = context.parentless_patches.begin(); it != context.parentless_patches.end(); ++it) {

        int vertex_id = it->first;
        vertex_descriptor s = vertex(vertex_id, context.graph);
        dijkstra_shortest_paths(context.graph, s,
            predecessor_map(predmap)
            .distance_map(distmap_vect));

        // -- Reachable patches
        boost::graph_traits<DirectedGraph>::vertex_iterator vi, vend;
        for (boost::tie(vi, vend) = vertices(context.graph); vi != vend; ++vi) {
            if (distvector[*vi] != 2147483647) {
                // std::cout << "Patch[" << *vi << "]-Distance: " << distvector[*vi] << ", ";
                total_reachable += 1;
            }
        }

        // For debugging/tracing purposes; print an update every 1% of parentless patches processed.
        if (count >= next_count || count == total_count) {
            int p = ((float) count / total_count) * 100;
            std::cout << "Patch ID: " << vertex_id << " :: " << (count + 1) << "/" << context.parentless_patches.size() << " = " << p << "%" << std::endl;
            auto step_time = (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count()) / 1000000.0;
            std::cout << "    -- Time since last update (seconds): " << step_time << std::endl;
            next_count += one_percent_amount;
            begin = std::chrono::steady_clock::now();
        }
        count += 1;
    }

    std::cout << "Total reachable nodes from parentless patches = " << total_reachable << std::endl;
    std::cout << "Number of parentless patches = " << context.parentless_patches.size() << std::endl;
    std::cout << "Average reachable nodes per parentless patch = " << ((float) total_reachable / context.parentless_patches.size()) << std::endl;
    std::cout << std::endl;
    */
}


double get_distance(Centroid c1, Centroid c2) {
    double x1, x2, y1, y2;
    int xs_squared, ys_squared;

    x1 = c1.first;
    x2 = c2.first;
    y1 = c1.second;
    y2 = c2.second;

    // Not true distance, but it's what we need
    return (sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)));
}
