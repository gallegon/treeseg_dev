#pragma once
#include "pdal/StageFactory.hpp"
#include "pdal/Stage.hpp"
#include "pdal/PointTable.hpp"
#include "pdal/Options.hpp"
#include "pdal/PointTable.hpp"
#include "pdal/Dimension.hpp"
#include "pdal/pdal_internal.hpp"
#include "pdal/Filter.hpp"

#include "Python.h"
#include "numpy/arrayobject.h"

namespace pdal {

class PDAL_DLL CustomFilter : public Filter {
public:
    CustomFilter() : Filter()
    {}
    std::string getName() const;

private:
    Dimension::Id dim_tree_id;

    virtual void addDimensions(PointLayoutPtr layout);
    virtual void addArgs(ProgramArgs& args);
    virtual PointViewSet run(PointViewPtr view);

    // Not implemented
    CustomFilter& operator=(const CustomFilter&);
    CustomFilter(const CustomFilter&);
};

} // nammespace pdal

