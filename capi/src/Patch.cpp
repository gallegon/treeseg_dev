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

// This is an old function for print debugging.  This shouldn't be necessary to use
// with the GDB debugging setup.
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

    double distance = get_distance(this->centroid, hac);

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

int Patch::get_cell_count() {
    return (this->cell_count);
}

std::pair<double, double> Patch::get_centroid() {
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
void addDirectedNeighbor(std::vector<int>& neighbors, Cell neighbor, int current_id,
                        int current_level, Grid<int>& labels,
                        Grid<int>& levels) {
    // get the feature id and level of the neighboring cell
    int neighbor_i = neighbor.first;
    int neighbor_j = neighbor.second;

    int neighbor_id = labels.get(neighbor_i, neighbor_j);
    int neighbor_level = levels.get(neighbor_i, neighbor_j);

    /*
    If the id of the feature is different the the current id, we know that
    two patches are connected.  Also check that the neighbor's level is
    not equal to 0.  We don't want to consider level 0 as patches.
    */
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

/*
This function gets the valid neighboring cells if there are any.  A valid
neighbor is a neighbor that: isn't equal to the current cell's ID and is 
within bounds of the array.  This is a helper function to get_neighbors.
*/
std::vector<Cell> get_valid_neighbors(int i, int j, int* dimensions) {
    std::vector<Cell> valid_neighbors;

    int m = dimensions[0];
    int n = dimensions[1];

    if ((i - 1) >= 0)
        valid_neighbors.push_back(std::make_pair(i + TOP, j));

    if ((i + 1) < m)
        valid_neighbors.push_back(std::make_pair(i + BOTTOM, j));

    if ((j - 1) >= 0)
        valid_neighbors.push_back(std::make_pair(i, j + LEFT));

    if ((j + 1) < n)
        valid_neighbors.push_back(std::make_pair(i, j + RIGHT));

    return valid_neighbors;
}


/*
This will compute the patch neighbors based on a current cell
*/
std::vector<int> get_neighbors(int i, int j, int* dimensions, Grid<int>& labels, Grid<int>& levels) {
    // This is for the valid neighbors of a given cell.  Valid neighbors are calculated below
    std::vector<int> neighbors;

    std::vector<Cell> neighboring_cells;

    
    int current_id = labels.get(i, j);
    int current_level = levels.get(i, j);

    /*
    Get neighboring cells that are valid (within bounds of the array) and not part of the 
    current cell's patch if there are any.  neighboring_cells may be empty if the cell is 
    in the middle of a patch and only surrounded by cells within the same patch.
    */
    neighboring_cells = get_valid_neighbors(i, j, dimensions);

    for (auto nc_it = neighboring_cells.begin(); nc_it != neighboring_cells.end(); ++nc_it) {
        Cell neighbor = *nc_it;
        addDirectedNeighbor(neighbors, neighbor,current_id, current_level, labels, levels);
    }

    return neighbors;
}

void create_patches(Grid<int>& labels, Grid<int>& levels, struct PdagData& context) {
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
            current_feature = labels.get(i, j);
            current_level = levels.get(i, j);

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
                context.parentless_patches.insert(current_feature);
                connected_patches.insert(std::make_pair(current_feature, current_feature));
            }

            // Get the current cell's neighbors, then iterate
            std::vector<int> neighbors = get_neighbors(i, j, dimensions, labels, levels);
            for (auto neigh_it = neighbors.begin(); neigh_it != neighbors.end(); ++neigh_it) {
                int neighbor = *neigh_it;
                /*
                Build the directed graph.  Direction is determined by the patch's height relative to a
                neighbor.  If a neighbor in neighbors is negative, then the patch is a child to it's
                neighbor
                */
                if (neighbor < 0) {
                    connected_patches.insert(std::make_pair((neighbor * -1), current_feature));
                    // Here the current feature is a child patch to some higher level patch.
                    // Therefore this patch cannot be a parentless patch.
                    context.parentless_patches.erase(current_feature);
                }
                // Here the patch is the parent to some lower patch.
                else {
                    connected_patches.insert(std::make_pair(current_feature, neighbor));
                    context.parentless_patches.erase(neighbor);
                }
            }
        }
    }

    DPRINT("== Completed edge list");
    
    // some abstraction for graph creation
    int parent, child;

    // add all the edges to the Boost graph container to form the Patch Directed Acyclic Graph
    for (set_iter = connected_patches.begin(); set_iter != connected_patches.end(); ++set_iter) {
        parent = set_iter->first;
        child = set_iter->second;

        /*
        Add a self edge to the patch directed acyclic graph (PDAG), this is a fix 
        otherwise the program will crash when it uses the boost graph library. 
        This specifically happens during dijkstra_shortest_paths() from the BGL
        during the hierarchy creation step.  
        
        Technically this violates the "acyclic" rule of the PDAG, as there is a 
        cycle from each parentless patch to itself with a weight of 0.  However 
        this doesn't seem to cause any problems.
        */
        if (parent == child) {
            boost::add_edge(parent, child, 0, context.graph);
        }

        // Otherwise, add the edge
        else {
            boost::add_edge(parent, child, 1, context.graph);
        }
        

    }
}


/*
Compute the Euclidean distance between two Centroids.
*/
double get_distance(Centroid c1, Centroid c2) {
    double x1, x2, y1, y2;
    double xs_squared, ys_squared;

    x1 = c1.first;
    x2 = c2.first;
    y1 = c1.second;
    y2 = c2.second;

    xs_squared = pow(x1 - x2, 2);
    ys_squared = pow(y1 - y2, 2);

    return sqrt(xs_squared + ys_squared);
}
