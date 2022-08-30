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

class PDAL_DLL CustomFilter : public Filter, public Streamable {
public:
    CustomFilter() : Filter()
    {}
    std::string getName() const;

    void withReader(Stage& reader);
    void withResolution(float res);
    void withDiscretization(int disc);
    Grid<int>* getGrid();

private:
    Stage* reader;
    Grid<int>* grid;
    double resolution;
    double discretization;
    double scale_x;
    double scale_y;
    double scale_z;
    int cell_size_x;
    int cell_size_y;
    int grid_width;
    int grid_height;
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    int min_z;
    int max_z;

    virtual void ready(PointTableRef table);
    virtual void addDimensions(PointLayoutPtr layout);
    virtual void addArgs(ProgramArgs& args);
    virtual void filter(PointView& view);
    virtual bool processOne(PointRef& point);
    // virtual PointViewSet run(PointViewPtr view);

    // Not implemented
    CustomFilter& operator=(const CustomFilter&);
    CustomFilter(const CustomFilter&);
};

} // nammespace pdal

