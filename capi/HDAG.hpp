#pragma once
#include "Hierarchy.hpp"

struct maximal_inbound_edge {
        int parent;
        double weight;
};

void create_HDAG(struct HierarchyData&, struct PdagData&);

void adjust_patches(struct HierarchyData&, struct PdagData&);
