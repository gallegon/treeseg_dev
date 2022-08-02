// vector_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <iostream>
#include <vector>
#include <map>

#include "Patch.hpp"

typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty > DirectedGraph;
typedef boost::graph_traits<DirectedGraph>::edge_iterator edge_iterator;

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