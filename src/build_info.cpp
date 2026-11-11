#include "diffxpbd/build_info.hpp"

#include <Eigen/Core>

#include <ostream>

namespace diffxpbd {

void print_build_info(std::ostream& output)
{
    output << "Eigen " << EIGEN_WORLD_VERSION << '.'
           << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION << '\n';
#ifdef DIFFXPBD_HAS_CUDA
    output << "Build configuration: CUDA enabled\n";
#else
    output << "Build configuration: CPU only\n";
#endif
}

} // namespace diffxpbd
