#include "FilterLabelPoints.hpp"

#include <iostream>

namespace pdal {

static PluginInfo const s_info {
    "filters.labelpoints",
    "Label points according to raster data",
    "link-to-documentation"
};

CREATE_STATIC_STAGE(FilterLabelPoints, s_info)

void FilterLabelPoints::setResolution(double resolution) {
    this->resolution = resolution;
}

void FilterLabelPoints::withReader(Stage& reader) {
    this->reader = &reader;
}

std::string FilterLabelPoints::getName() const {
    return s_info.name;
}

void FilterLabelPoints::withGrid(Grid<int>* grid) {
    this->grid = grid;
}

void FilterLabelPoints::addArgs(ProgramArgs& args) {

}

void FilterLabelPoints::addDimensions(PointLayoutPtr layout) {
    this->dim_tree_id = layout->registerOrAssignDim("TreeID", Dimension::Type::Unsigned64);
}

void FilterLabelPoints::ready(PointTableRef table) {
    auto metadata = reader->getMetadata();

    auto header = [&metadata](auto s) {
        return metadata.findChild(s);
    };

    scale_x = header("scale_x").value<double>();
    scale_y = header("scale_y").value<double>();
    min_x = (int) (header("minx").value<double>() / scale_x);
    max_x = (int) (header("maxx").value<double>() / scale_x);
    min_y = (int) (header("miny").value<double>() / scale_y);
    max_y = (int) (header("maxy").value<double>() / scale_y);
    cell_size_x = (int) ceil(resolution / scale_x);
    cell_size_y = (int) ceil(resolution / scale_y);
}

bool FilterLabelPoints::processOne(PointRef& point) {
    double x;
    double y;
    point.getField((char*) &x, Dimension::Id::X, Dimension::Type::Double);
    point.getField((char*) &y, Dimension::Id::Y, Dimension::Type::Double);
    x /= scale_x;
    y /= scale_y;

    int cell_x = (int) ((x - min_x) / cell_size_x);
    int cell_y = (int) ((y - min_y) / cell_size_y);
    
    int label = grid->get(cell_x, cell_y);

    // std::cout << "TreeID: " << (int) (this->dim_tree_id) << std::endl;
    // std::cout << "About to set (" << cell_x << ", " << cell_y << ") =" << label << std::endl;

    unsigned long h_label = (unsigned long) label;
    point.setField<unsigned long>(this->dim_tree_id, h_label);

    if (point.pointId() % 1000 == 0) {
        std::cout << point.pointId() << " :: " << point.getFieldAs<unsigned long>(this->dim_tree_id) << std::endl;
    }

    return true;
}

void FilterLabelPoints::filter(PointView& view) {
    PointRef point = view.point(0);

    DPRINT("Starting point loop");
    std::cout << "Labeling the points!" << std::endl;
    for (PointId pid = 0; pid < view.size(); ++pid) {
        // Changes the Point which the PointRef points to.
        // *NOT* changing the ID of the point itself!
        point.setPointId(pid);
        processOne(point);
    }

    DPRINT("Processed every point!");
}

} // namespace pdal
