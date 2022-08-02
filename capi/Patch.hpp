#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <iostream>
#include <vector>
#include <map>

typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty > DirectedGraph;
typedef boost::graph_traits<DirectedGraph>::edge_iterator edge_iterator;

// returns an index i, j for 1D array
#define IDX(i, j) ((j) + (i) * (n))

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

std::vector<int> get_neighbors(int i, int j, int* dimension,
                               int* labels, int* levels);

void create_patches(PyArrayObject* labels, PyArrayObject* levels, int* dimensions);
