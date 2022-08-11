#pragma once
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL treeseg_ARRAY_API
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
#include <utility>  // for std::pair == operator overload
#include <set>
//#include <chrono>

#define Ptr2D(array, i, j) ((int*) PyArray_GETPTR2(array, i, j))
#define Get2D(array, i, j) (*((int*) PyArray_GETPTR2(array, i, j)))

typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty > DirectedGraph;
typedef boost::graph_traits<DirectedGraph>::edge_iterator edge_iterator;

class Patch {
private:
    int id;
    int level;
    std::vector<std::pair<int, int> > cells;
    std::vector<int> associated_hierarchies;
    std::pair<double, double> centroid;
    int sum_x = 0;
    int sum_y = 0;
    int cell_count = 0;
public:
    Patch(int, int);
    void add_cell(int, int);
    void print_cells();
    void update_centroid();
    void add_hierarchy(int, std::set<std::pair<int, int> >&);
    int get_level();
    int getCellCount();
    std::pair<double, double> getCentroid();
    void operator = (const Patch&);

    int get_id() { return id; }
};

// This is a struct that will be used to pass data to the Hierarchy step
struct PdagData {
    std::map<int, Patch> patches;
    std::map<int, Patch> parentless_patches;
    DirectedGraph graph;
};

std::vector<int> get_neighbors(int i, int j, int* dimension,
                               PyArrayObject* labels, PyArrayObject* levels);

void create_patches(PyArrayObject* labels, PyArrayObject* levels, int* dimensions, struct PdagData&);
