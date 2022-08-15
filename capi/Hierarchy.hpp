#pragma once
//#ifndef __HIERARCHY__HPP
//#define __HIERARCHY__HPP

#include "Patch.hpp"

class Hierarchy {
private:
    int id;
    int height;
    int cellCount;
    std::map<int, Patch> patches;
    std::vector<int> patchIDs;
    std::pair<double, double> heightAdjustedCentroid;
    std::map<int, std::pair<int, int> > patchDepthMap;

public:
    Hierarchy();
    Hierarchy(int, int);
    void add_patch(int, Patch, std::pair<int, int>);
    void add_patchID(int, std::pair<int, int>);
    std::vector<int> getPatchIDs();
    std::pair<int, int> getPatchDepths(int patchID);
    void setHAC(double, double);
    std::pair<double, double> get_HAC();
    void setCellCount(int);
    void remove_patch(int);
};

struct HierarchyData {
    std::map<int, Hierarchy> hierarchies;
    std::set<std::pair<int, int> > connected_hierarchies;
};

void compute_hierarchies(struct PdagData&, struct HierarchyData&);

void calculateHAC(PdagData&, HierarchyData&);
//#endif
