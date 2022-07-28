// vector_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <iostream>
#include <vector>
#include <map>

typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty > DirectedGraph;
typedef boost::graph_traits<DirectedGraph>::edge_iterator edge_iterator;

// TODO centroid calculation
class Patch {
private:
    int id;
    std::vector<std::pair<int, int>> cells;
    std::pair<double, double> centroid;
    int sum_x = 0;
    int sum_y = 0;
    int cell_count = 0;
public:
    Patch(int);
    void add_cell(int, int);
    void print_cells();
    void update_centroid();
};

Patch::Patch(int id) {
    this->id = id;
    this->cell_count = 0;
}

void Patch::add_cell(int i, int j) {
    std::pair<int, int> cell = std::make_pair(i, j);
    this->cells.push_back(cell);
    this->sum_x += i;
    this->sum_y += j;
    (this->cell_count)++;
    this->update_centroid();
}

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

#define IDX(i, j) ((j) + (i) * (n))

std::vector<int> get_neighbors(int i, int j, int* dimensions, int* labels, int* levels) {
    // m is the i size of the array, n is the j size
    int m = dimensions[0];
    int n = dimensions[1];
    int neighbor_id;
    int neighbor_level;
    // This is for the valid neighbors of a given cell.  Valid neighbors are
    // calculated below
    std::vector<int> neighbors;

    int current_id = labels[IDX(i, j)];
    int current_level = levels[IDX(i, j)];

    if ((i - 1) >= 0) {
        neighbor_id = labels[IDX(i - 1, j)];
        neighbor_level = levels[IDX(i - 1, j)];

        if (neighbor_id != current_id) {
            if (neighbor_level > current_level) {
                // set the direction flag towards the higher neighbor
                neighbor_id = neighbor_id * -1;
            }
            neighbors.push_back(neighbor_id);

        }
        //neighbors.push_back(labels[i - 1][j]);
    }
    if ((i + 1) < m) {
        neighbor_id = labels[IDX(i + 1, j)];
        neighbor_level = levels[IDX(i + 1, j)];

        if (neighbor_id != current_id) {
            if (neighbor_level > current_level) {
                // set the direction flag towards the higher neighbor
                neighbor_id = neighbor_id * -1;
            }
            neighbors.push_back(neighbor_id);

        }
        
        //neighbors.push_back(labels[i + 1][j]);
    }
    if ((j - 1) >= 0) {
        neighbor_id = labels[IDX(i, j - 1)];
        neighbor_level = levels[IDX(i, j - 1)];

        if (neighbor_id != current_id) {
            if (neighbor_level > current_level) {
                // set the direction flag towards the higher neighbor
                neighbor_id = neighbor_id * -1;
            }
            neighbors.push_back(neighbor_id);

        }
        //neighbors.push_back(labels[i][j - 1]);
    }
    if ((j + 1) < n) {
        neighbor_id = labels[IDX(i, j + 1)];
        neighbor_level = levels[IDX(i, j + 1)];

        if (neighbor_id != current_id) {
            if (neighbor_level > current_level) {
                // set the direction flag towards the higher neighbor
                neighbor_id = neighbor_id * -1;
            }
            neighbors.push_back(neighbor_id);

        }
        //neighbors.push_back(labels[i][j + 1]);
    }

    return neighbors;
}

void create_patches(int* labels, int* levels, int* dimensions) {
    std::map<int, Patch> patches;

    // Use this dictionary to keep track of patches that have no parent.
    // These will be the "local maxima" patches
    std::map<int, Patch> parentless_patches;

    int m = dimensions[0];
    int n = dimensions[1];
    int current_feature;

    DirectedGraph g;

    // iterator for patches map
    std::map<int, Patch>::iterator it;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            current_feature = labels[j + i * n];
            // check if the feature has already been turned into a patch
            it = patches.find(current_feature);

            // If the feature is already a patch, just add the cell
            if (it != patches.end()) {
                it->second.add_cell(i, j);
                // TODO add centroid calculation here probably
            }
            // Else the feature is not a patch
            else {
                // create a patch, then add the cell to it
                Patch p(current_feature);
                p.add_cell(i, j);
                patches.insert(std::pair<int, Patch>(current_feature, p));
                parentless_patches.insert(std::pair<int, Patch>(current_feature, p));
            }

            // Get the current cell's neighbors, then iterat
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
                }
            }
        }
    }

    // Print the patches for testing
    // for (it = patches.begin(); it != patches.end(); ++it) {
    //     std::cout << it->first << std::endl;
    //     (it->second).print_cells();
    //     std::cout << std::endl;
    // }

    // Print the edges for testing
    // std::pair<edge_iterator, edge_iterator> ei = edges(g);
    // std::copy(ei.first, ei.second,
        // std::ostream_iterator<boost::adjacency_list<>::edge_descriptor>{
        // std::cout, "\n"});
}

void print_array(int arr[3][3], int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

struct patch_visitor {
    typedef boost::on_discover_vertex event_filter;
    template <class T, class Graph> void operator()(T t, Graph&) {}
};

// int main()
// {
//     int test_arr[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };

//     int test_levels[3][3] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
//     int test_arr2[3][3] = { {5, 1, 2},
//                             {1, 1, 2},
//                             {4, 3, 3}
//     };

//     int test_levels2[3][3] = {  {4, 3, 1}, 
//                                 {3, 3, 1}, 
//                                 {5, 7, 7} };

//     int dims[2] = { 3, 3 };

//     std::cout << "Labeled features sample" << std::endl;
//     print_array(test_arr2, 3, 3);

//     std::cout << std::endl;

//     std::cout << "Discrete levels sample" << std::endl;
//     print_array(test_levels2, 3, 3);

//     std::cout << std::endl;

// #if 0
//     int i = 0;
//     int j = 0;
//     int dims[2] = { 3, 3 };
//     std::vector<int> test = get_neighbors(i, j, dims, test_arr, test_levels);

//     for (int i : test)
//         std::cout << i << " ";

//     std::cout << std::endl;

// //# if 0
//     std::cout << "Hello World!\n";

//     DirectedGraph g;

//     boost::add_edge(0, 1, 8, g);
//     boost::add_edge(0, 3, 18, g);
//     boost::add_edge(1, 2, 20, g);
//     boost::add_edge(2, 3, 2, g);
//     boost::add_edge(3, 1, 1, g);
//     boost::add_edge(1, 3, 7, g);
//     boost::add_edge(1, 4, 1, g);
//     boost::add_edge(4, 5, 6, g);
//     boost::add_edge(2, 5, 7, g);

//     std::pair<edge_iterator, edge_iterator> ei = edges(g);

//     std::cout << "Number of edges = " << num_edges(g) << "\n";
//     std::cout << "Edge list:\n";

//     std::copy(ei.first, ei.second,
//         std::ostream_iterator<boost::adjacency_list<>::edge_descriptor>{
//         std::cout, "\n"});
// #endif

//     create_patches(test_arr2, test_levels2, dims);

//     return 0;

// }