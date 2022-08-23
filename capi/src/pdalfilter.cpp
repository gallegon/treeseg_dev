#include "pdalfilter.hpp"

#include <iostream>

namespace pdal {

static PluginInfo const s_info {
    "filters.customfilter",
    "Custom filter",
    "link-to-documentation"
};

CREATE_STATIC_STAGE(CustomFilter, s_info)

std::string CustomFilter::getName() const {
    return s_info.name;
}

PyArrayObject* CustomFilter::getGrid() {
    return grid;
}

void CustomFilter::withResolution(float res) {
    resolution = res;
}

void CustomFilter::withReader(Stage& reader) {
    this->reader = &reader;
}

void CustomFilter::withDiscretization(int disc) {
    this->discretization = disc;
}

void CustomFilter::addArgs(ProgramArgs& args) {

}

void CustomFilter::addDimensions(PointLayoutPtr layout) {
}

void CustomFilter::ready(PointTableRef table) {
    auto metadata = reader->getMetadata();

    auto header = [&metadata](auto s) {
        return metadata.findChild(s);
    };

    scale_x = header("scale_x").value<double>();
    scale_y = header("scale_y").value<double>();
    scale_z = header("scale_z").value<double>();
    min_x = (int) (header("minx").value<double>() / scale_x);
    max_x = (int) (header("maxx").value<double>() / scale_z);
    min_y = (int) (header("miny").value<double>() / scale_y);
    max_y = (int) (header("maxy").value<double>() / scale_y);
    min_z = (int) (header("minz").value<double>() / scale_z);
    max_z = (int) (header("maxz").value<double>() / scale_z);
    double range_x = (max_x - min_x) * scale_x;
    double range_y = (max_y - min_y) * scale_y;
    std::cout << "rangeX: (" << min_x << ", " << max_x << ") = " << range_x << std::endl;
    std::cout << "rangeY: (" << min_y << ", " << max_y << ") = " << range_y << std::endl;
    std::cout << "scaleX: " << scale_x << std::endl;
    std::cout << "scaleY: " << scale_y << std::endl;
    std::cout << "scaleZ: " << scale_z << std::endl;
    grid_width = (int) ceil(range_x / resolution);
    grid_height = (int) ceil(range_y / resolution);
    cell_size_x = (int) ceil(resolution / scale_x);
    cell_size_y = (int) ceil(resolution / scale_y);
    std::cout << "grid_size: (" << grid_width << ", " << grid_height << ")" << std::endl;
    std::cout << "cell_size: (" << cell_size_x << ", " << cell_size_y << ")" << std::endl;

    npy_intp dims[] = { grid_width, grid_height };
    PyArrayObject* newgrid = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_INT, 0);
    std::cout << "Created grid" << std::endl;
    this->grid = newgrid;

    // grid_size = np.ceil(range_xyz[:2] / resolution).astype("int")
    // cell_size = np.ceil(resolution / scale_xyz[:2]).astype("int")
    std::cout << "reader count = " << reader->getMetadata().findChild("count").value<double>() << std::endl;
}

bool CustomFilter::processOne(PointRef& point) {
    double x;
    double y;
    double z;
    point.getField((char*) &x, Dimension::Id::X, Dimension::Type::Double);
    point.getField((char*) &y, Dimension::Id::Y, Dimension::Type::Double);
    point.getField((char*) &z, Dimension::Id::Z, Dimension::Type::Double);
    x /= scale_x;
    y /= scale_y;
    z /= scale_z;

    int cell_x = (int) ((x - min_x) / cell_size_x);
    int cell_y = (int) ((y - min_y) / cell_size_y);
    int level = (int) round(z / max_z * discretization);
    
    int* grid_ptr = (int*) PyArray_GETPTR2(grid, cell_x, cell_y);
    if (level > *grid_ptr) {
        *grid_ptr = level;
    }

    return true;
}

void CustomFilter::filter(PointView& view) {
    PointRef point = view.point(0);

    for (PointId pid = 0; pid < view.size(); ++pid) {
        // Changes the Point which the PointRef points to.
        // *NOT* changing the ID of the point itself!
        point.setPointId(pid);
        processOne(point);
    }
}

} // namespace pdal
