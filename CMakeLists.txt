cmake_minimum_required(VERSION 3.24)
project(DiffXPBD VERSION 0.1.0 LANGUAGES CXX)

option(DIFFXPBD_ENABLE_CUDA "Build the CUDA backend and smoke test" ON)

find_package(Eigen3 3.4.1 EXACT CONFIG QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Build Eigen's own tests")
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Build Eigen documentation")
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
        URL_HASH SHA256=b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

if(DIFFXPBD_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
endif()

add_library(diffxpbd_core STATIC
    src/build_info.cpp
    src/xpbd.cpp
    src/kinematic_collider.cpp
    src/linearization.cpp
    src/adjoint.cpp
)
add_library(DiffXPBD::core ALIAS diffxpbd_core)
target_sources(diffxpbd_core
    PUBLIC FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/diffxpbd/build_info.hpp
        include/diffxpbd/xpbd.hpp
        include/diffxpbd/kinematic_collider.hpp
        include/diffxpbd/contact.hpp
        include/diffxpbd/linearization.hpp
        include/diffxpbd/adjoint.hpp
)
target_link_libraries(diffxpbd_core PUBLIC Eigen3::Eigen)
target_compile_features(diffxpbd_core PUBLIC cxx_std_17)

add_executable(diffxpbd src/main.cpp)
target_compile_definitions(diffxpbd PRIVATE
    DIFFXPBD_DEFAULT_MESH_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/remeshed/skirt1252.obj"
)
target_link_libraries(diffxpbd PRIVATE DiffXPBD::core)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT diffxpbd)
set_target_properties(diffxpbd diffxpbd_core PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(diffxpbd_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    target_compile_options(diffxpbd PRIVATE /utf-8)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME scaffold_startup COMMAND diffxpbd)
endif()

if(DIFFXPBD_ENABLE_CUDA)
    target_sources(diffxpbd_core
        PRIVATE src/cuda/constraint_sweeps.cu
        PUBLIC FILE_SET HEADERS FILES include/diffxpbd/cuda/constraint_sweeps.hpp
    )
    target_compile_definitions(diffxpbd_core PRIVATE DIFFXPBD_HAS_CUDA=1)

    add_executable(cuda_smoke tests/cuda_smoke.cu)
    set_target_properties(diffxpbd_core cuda_smoke PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(diffxpbd_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
        target_compile_options(cuda_smoke PRIVATE -Xcompiler=/utf-8)
    endif()
    if(BUILD_TESTING)
        add_test(NAME cuda_smoke COMMAND cuda_smoke)
    endif()
endif()

if(CMAKE_GENERATOR MATCHES "^Visual Studio")
    # Browse the full source tree without adding files to a compile target.
    file(GLOB_RECURSE diffxpbd_browse_files CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
    )
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${diffxpbd_browse_files})
    add_custom_target(diffxpbd_project_files SOURCES ${diffxpbd_browse_files})
    set_target_properties(diffxpbd_project_files PROPERTIES PROJECT_LABEL "Project Files")
endif()
