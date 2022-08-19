#pragma once
#include "Hierarchy.hpp"

typedef std::pair<int, double> MaxInboundEdge;

// This pair represents an edge between two hierarchies
// it is as follows: (parent, child)
typedef std::pair<int, int> HierarchyPair;

typedef std::tuple<int, int, double> DirectedWeightedEdge;
struct maximal_inbound_edge {
        int parent;
        double weight;
};

void create_HDAG(std::vector<DirectedWeightedEdge>&, struct HierarchyData&, struct PdagData&, PyArrayObject*);

void adjust_patches(struct HierarchyData&, struct PdagData&);

void map_cells_to_hierarchies(struct HierarchyData&, struct PdagData&);

void init_MIE_map(std::map<int, MaxInboundEdge>);

HierarchyPair get_direction(Hierarchy, Hierarchy);