#include <iostream>

#include "Patch.hpp"
#include "Hierarchy.hpp"
#include "disjointpatches.hpp"
#include "HDAG.hpp"
#include "pdalfilter.hpp"
#include "disjointtrees.hpp"
#include "grid.hpp"

#include "debug.hpp"

Grid<int>* discretize_points(char* filepath_in, double resolution, int discretization) {
    using namespace pdal;

    PointTable table;
    table.layout()->registerDim(Dimension::Id::X);
    table.layout()->registerDim(Dimension::Id::Y);
    table.layout()->registerDim(Dimension::Id::Z);

    StageFactory factory;
    Stage* reader = factory.createStage("readers.las");
    CustomFilter* filter = dynamic_cast<CustomFilter*>(factory.createStage("filters.customfilter"));
    Stage* writer = factory.createStage("writers.las");

    DPRINT("-- Created stages...");

    Options options;
    options.add("filename", filepath_in);
    reader->setOptions(options);

    DPRINT("-- Running pipeline...");

    filter->withResolution(resolution);
    filter->withDiscretization(discretization);
    filter->withReader(*reader);
    filter->setInput(*reader);
    filter->prepare(table);
    filter->execute(table);

    DPRINT("-- Finished running reader & filter");

    auto grid = filter->getGrid();
    
    return grid;
}

Grid<int>* label_grid(Grid<int>* grid_levels) {
    Grid<int>* grid_labels = new Grid<int>(grid_levels->width, grid_levels->height);
    DisjointPatches ds(*grid_levels, *grid_labels);
    ds.compute_patches();

    DPRINT("Total number of patches = " << ds.size());

    return grid_labels;
}

Grid<int>* vector_test(Grid<int>* grid_levels, Grid<int>* grid_patches, float weights[6]) {
    // Used to get data from the patch creation step
    struct PdagData pdag;
    struct HierarchyData hierarchyContext;
    std::vector<DirectedWeightedEdge> partitioned_edge_list;

    // Used to get data from the patch creation step
    //struct PdagData pdag;

    create_patches(*grid_patches, *grid_levels, pdag);
    compute_hierarchies(pdag, hierarchyContext);
    //calculateHAC(pdag, hierarchyContext);
    adjust_patches(hierarchyContext, pdag);
    map_cells_to_hierarchies(hierarchyContext, pdag);
    create_HDAG(partitioned_edge_list, hierarchyContext, pdag, weights);

    DPRINT(
        "== Partitioned Edge List" << std::endl
        << "size: " << partitioned_edge_list.size()
    );

    DisjointTrees dt;
    for (std::vector<DirectedWeightedEdge>::iterator vit = partitioned_edge_list.begin(); vit != partitioned_edge_list.end(); ++vit) {
        int parent_id = std::get<0>(*vit);
        int child_id = std::get<1>(*vit);
        double weight = std::get<2>(*vit);
        // std::cout << "(parent=" << parent_id << ", child=" << child_id << ", weight=" << weight << ")" << std::endl;
        TreeID tree_parent = dt.tree_from_hierarchy(parent_id);
        TreeID tree_child = dt.tree_from_hierarchy(child_id);
        // std::cout << "Got trees from hierarchies" << std::endl;
        if (tree_parent == 0) {
            tree_parent = dt.make_tree(parent_id);
        }
        
        if (tree_child == 0) {
            tree_child = dt.make_tree(child_id);
        }
        // std::cout << "Made trees" << std::endl;
        // std::cout << "Pre-union: " << tree_parent << ", " << tree_child << std::endl;
        dt.union_trees(tree_parent, tree_child);
        // std::cout << "Unioned trees" << std::endl;
    }

    Grid<int>* grid_hierarchy = new Grid<int>(grid_levels->width, grid_levels->height);

    DPRINT("== Roots");

    auto roots = dt.roots();

    for (auto it = roots.begin(); it != roots.end(); ++it) {
        TreeID tree_id = *it;

        DPRINT("  " << *it);

        auto hs = dt.hierarchies_from_tree(tree_id);
        std::set<int> patches_for_tree;
        std::set<Cell> cells_for_tree;
        for (auto hit = hs.begin(); hit != hs.end(); hit++) {
            
            DPRINT("    " << *hit);
            
            Hierarchy hierarchy = hierarchyContext.hierarchies[*hit];
            auto cells = hierarchy.get_adjusted_cells();
            cells_for_tree.insert(cells.begin(), cells.end());
        }

        for (auto cit = cells_for_tree.begin(); cit != cells_for_tree.end(); ++cit) {
            int x = cit->first;
            int y = cit->second;
            grid_hierarchy->set(x, y, tree_id);
        }
    }

    std::cout << std::endl;

    return grid_hierarchy;
}

int main(int argc, char* argv[]) {
    Grid<int>* grid_levels;
    Grid<int>* grid_patches;
    Grid<int>* grid_hierarchy;

    float weights[6] = {
        0.22, 0.22, 0.22, 0.22, 0.22, 0.22
    };

    double resolution = 1;
    int discretization = 5;

    char* fp_in = argv[1];

    std::cout << "File in: " << fp_in << std::endl;

    std::cout << "-- Discretize Points" << std::endl;
    grid_levels = discretize_points(fp_in, resolution, discretization);
    std::cout << "-- Label Grid" << std::endl;
    grid_patches = label_grid(grid_levels);
    std::cout << "-- Grid Hierarchy" << std::endl;
    grid_hierarchy = vector_test(grid_levels, grid_patches, weights);
    std::cout << "All done!" << std::endl;

    std::cout << std::endl;
}