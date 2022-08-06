#pragma once
//#ifndef __HIERARCHY__HPP
//#define __HIERARCHY__HPP

#include "Patch.hpp"

class Hierarchy {
private:
    int id;
    int height;
    std::map<int, Patch> patches;
    std::vector<int> patchIDs;
    std::pair<double, double> heightAdjustedCentroid;
public:
    Hierarchy(int, int);
    void add_patch(int, Patch);
    void add_patchID(int);
    std::vector<int> getPatchIDs();
};

struct HierarchyData {
    std::map<int, Hierarchy> hierarchies;
    std::set<std::pair<int, int>> connected_hierarchies;
};

void compute_hierarchies(struct PdagData&, struct HierarchyData&);

//#endif
