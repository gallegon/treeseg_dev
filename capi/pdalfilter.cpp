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

void CustomFilter::addArgs(ProgramArgs& args) {
    std::cout << "CustomFilter.addArgs" << std::endl;
}

void CustomFilter::addDimensions(PointLayoutPtr layout) {
    std::cout << "CustomFilter.addDimensions" << std::endl;
}

PointViewSet CustomFilter::run(PointViewPtr input) {
    std::cout << "CustomFilter.run" << std::endl;
    // Use the PointViewSet to access point data (read/write).
    // getField and setField
    PointViewSet viewSet;
    viewSet.insert(input);
    return viewSet;
}

} // namespace pdal
