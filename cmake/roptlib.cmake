cmake_minimum_required(VERSION 3.5)

project(roptlib-download NONE)

include(ExternalProject)
ExternalProject_Add(roptlib
        GIT_REPOSITORY    https://github.com/yuluntian/ROPTLIB.git
        GIT_TAG           feature/cmake
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlib-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlib-build"
        PATCH_COMMAND     "${CMAKE_COMMAND}" -DROPTLIB_SOURCE_DIR=${CMAKE_CURRENT_BINARY_DIR}/roptlib-src -P "@CMAKE_SOURCE_DIR@/cmake/patch_roptlib.cmake"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      "")
