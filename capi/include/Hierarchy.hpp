#pragma once
//#ifndef __HIERARCHY__HPP
//#define __HIERARCHY__HPP

#include "Patch.hpp"
#include "debug.hpp"

class Hierarchy {
private:
    int id;
    int height;
    int cellCount;
    int top_patch_cell_count;
    std::map<int, Patch> patches;
    std::vector<int> patchIDs;
    Centroid heightAdjustedCentroid;
    Centroid top_patch_centroid;
    std::map<int, std::pair<int, int> > patchDepthMap;
    std::vector<int> adjusted_patch_ids;
    std::vector<Cell> adjusted_cells;
public:
    Hierarchy();
    Hierarchy(int, int);
    void add_patch(int, Patch, std::pair<int, int>);
    void add_patchID(int, std::pair<int, int>);
    std::vector<int> getPatchIDs();
    std::pair<int, int> getPatchDepths(int patchID);
    void setHAC(double, double);
    void set_TPC(Centroid);
    void set_tp_cell_count(int);
    int get_tp_cell_count();
    Centroid get_TPC();
    Centroid get_HAC();
    void setCellCount(int);
    void remove_patch(int);
    void add_adjusted_cell(Cell);
    void add_adjusted_patch_id(int);
    int get_cell_count();
    int get_height();
    int get_id();
    std::vector<Cell> get_adjusted_cells();
};

struct HierarchyData {
    std::map<int, Hierarchy> hierarchies;
    std::set<std::pair<int, int> > connected_hierarchies;
};

void compute_hierarchies(struct PdagData&, struct HierarchyData&);

void calculateHAC(PdagData&, HierarchyData&);
//#endif
