#pragma once
#include "pdal/StageFactory.hpp"
#include "pdal/Stage.hpp"
#include "pdal/PointTable.hpp"
#include "pdal/Options.hpp"
#include "pdal/PointTable.hpp"
#include "pdal/Dimension.hpp"
#include "pdal/pdal_internal.hpp"
#include "pdal/Filter.hpp"
#include "pdal/Reader.hpp"
#include "pdal/Streamable.hpp"

#include "grid.hpp"
#include "debug.hpp"

namespace pdal {

class PDAL_DLL FilterLabelPoints : public Filter, public Streamable {
public:
    FilterLabelPoints() : Filter()
    {}
    std::string getName() const;
    
    void withReader(Stage& reader);
    void withGrid(Grid<int>* grid);
    void setResolution(double resolution);

private:
    Stage* reader;
    Grid<int>* grid;

    Dimension::Id dim_tree_id;
    double resolution;
    double scale_x, scale_y;
    double min_x, min_y, max_x, max_y;
    double cell_size_x, cell_size_y;

    virtual void ready(PointTableRef table);
    virtual void addDimensions(PointLayoutPtr layout);
    virtual void addArgs(ProgramArgs& args);
    virtual void filter(PointView& view);
    virtual bool processOne(PointRef& point);
    // virtual PointViewSet run(PointViewPtr view);

    // Not implemented
    FilterLabelPoints& operator=(const FilterLabelPoints&);
    FilterLabelPoints(const FilterLabelPoints&);
};

} // nammespace pdal

