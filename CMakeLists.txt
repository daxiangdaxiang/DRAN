# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 3.5)

project(DPGO)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # Options: Debug, Release, MinSizeRel, RelWithDebInfo
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type." FORCE)
endif()

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -march=native)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


include(GNUInstallDirs)
set(DPGO_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/DPGO)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place libraries built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# ROPTLIB
configure_file(cmake/roptlib.cmake roptlib-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/roptlib-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/roptlib-download")
add_subdirectory("${CMAKE_BINARY_DIR}/roptlib-src"
        "${CMAKE_BINARY_DIR}/roptlib-build")

# Dependencies
find_package(SPQR REQUIRED)
find_package(Cholmod REQUIRED)
find_package(BLAS REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost)
find_package(Eigen3 REQUIRED)
option(BUILD_DCCI_MPI "Build optional DCCI MPI communicator example" ON)
if(BUILD_DCCI_MPI)
	find_package(MPI QUIET)
endif()
set(DCCI_MPI_ENABLED OFF)
if(BUILD_DCCI_MPI AND MPI_CXX_FOUND)
	set(DCCI_MPI_ENABLED ON)
endif()
if(DCCI_MPI_ENABLED)
	find_program(DPGO_MPIRUN_EXECUTABLE mpirun)
	set(DPGO_TED_CCI_MPIEXEC "${MPIEXEC_EXECUTABLE}")
	set(DPGO_TED_CCI_MPIEXEC_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}")
	if(DPGO_MPIRUN_EXECUTABLE)
		set(DPGO_TED_CCI_MPIEXEC "${DPGO_MPIRUN_EXECUTABLE}")
		set(DPGO_TED_CCI_MPIEXEC_NUMPROC_FLAG "-np")
	endif()
endif()
message(STATUS "Found Eigen version: ${EIGEN3_VERSION}")
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

# Enable OpenMP (if available)
set(ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP (if available)")

# PERFORMANCE IMPROVEMENTS
if(${ENABLE_OPENMP})
find_package(OpenMP)
if(OPENMP_FOUND)
message(STATUS "Found OpenMP! Turning on support for parallelization\n")
endif()
endif()

message(STATUS "Boost version: " ${Boost_VERSION_STRING})

set(EXTERNAL_INCLUDES
	${CHOLMOD_INCLUDES}
	${EIGEN3_INCLUDE_DIR}
	${Boost_INCLUDE_DIRS}
	CACHE INTERNAL "")

add_library(DPGO SHARED
	src/manifold/LiftedSEManifold.cpp
	src/manifold/LiftedSEVariable.cpp
	src/manifold/LiftedSEVector.cpp
	src/ObjectConsensus.cpp
	src/GeodesicSE3.cpp
	src/GeodesicObjectProblem.cpp
	src/TEDCCI.cpp
	src/RIFTIF.cpp
	src/InProcessDCCICommunicator.cpp
	src/ObjectGaugeCoupling.cpp
	src/BoundarySchurPreconditioner.cpp
	src/DCCI_chordal_initialization.cpp
	src/DCCI_linear_operators.cpp
	src/FullEquivHybridOptimizer.cpp
	src/ManualDpgoMm.cpp
	src/ManualObjectDran.cpp
		src/PGOAgent.cpp
		src/ManualQuadraticOptimizer.cpp
		src/ReducedRotationQuadraticOptimizer.cpp
		src/QuadraticProblem.cpp
	src/QuadraticOptimizer.cpp
	src/RoptOptimizer.cpp
	src/DPGO_utils.cpp
	src/ObjectPGOData.cpp
    src/DPGO_robust.cpp
	src/PGOLogger.cpp)
if(DCCI_MPI_ENABLED)
	target_sources(DPGO PRIVATE
		src/MPIDCCICommunicator.cpp
		src/MPIAsyncFactorExchange.cpp)
endif()

target_include_directories(DPGO PUBLIC
	${EXTERNAL_INCLUDES}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	# INSTALL
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}>
	)
target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${Boost_LIBRARIES})
if(DCCI_MPI_ENABLED)
	target_link_libraries(DPGO MPI::MPI_CXX)
endif()

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
	set_target_properties(DPGO PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
	set_target_properties(DPGO PROPERTIES LINK_FLAGS "-fopenmp")
	target_link_libraries(DPGO)
endif()

# Build Distributed PGO example
add_executable(multi-robot-example
		examples/MultiRobotExample.cpp)
target_link_libraries(multi-robot-example DPGO)

add_executable(object-based-multi-robot-example
		examples/ObjectBasedMultiRobotExample.cpp)
target_link_libraries(object-based-multi-robot-example DPGO)

add_executable(consensus
		examples/test.cpp)
target_link_libraries(consensus DPGO)

add_executable(chordal-initialization-example
		examples/ChordalInitializationExample.cpp)
target_link_libraries(chordal-initialization-example DPGO)

add_executable(manual-dpgo-mm-example
		examples/ManualDpgoMmExample.cpp)
target_link_libraries(manual-dpgo-mm-example DPGO)

add_executable(manual-object-dran-example
		examples/ManualObjectDranExample.cpp)
target_link_libraries(manual-object-dran-example DPGO)

add_executable(bench-dcci
		benchmarks/bench_dcci.cpp)
target_link_libraries(bench-dcci DPGO)

if(DCCI_MPI_ENABLED)
	add_executable(run_dcci_mpi
			examples/run_dcci_mpi.cpp)
	target_link_libraries(run_dcci_mpi DPGO MPI::MPI_CXX)

	add_executable(run_ted_cci_mpi
			examples/run_ted_cci_mpi.cpp)
	target_link_libraries(run_ted_cci_mpi DPGO MPI::MPI_CXX)
endif()

# add_executable(single-robot-example
# 	examples/SingleRobotExample.cpp)
# target_link_libraries(single-robot-example DPGO)

############################### TESTS ##########################################
### Add testing
option(BUILD_DPGO_TESTS "Build tests" OFF)
if(BUILD_DPGO_TESTS)
	enable_testing()

	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest.cmake googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()

	# Prevent overriding the parent project's compiler/linker
	# settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	# Add googletest directly to our build. This defines
	# the gtest and gtest_main targets.
	add_subdirectory("${CMAKE_BINARY_DIR}/external/googletest-src"
                     "${CMAKE_BINARY_DIR}/external/googletest-build"
                     EXCLUDE_FROM_ALL)
	if(TARGET gtest)
		target_compile_options(gtest PRIVATE -Wno-error=maybe-uninitialized)
	endif()

	# The gtest/gtest_main targets carry header search path
	# dependencies automatically when using CMake 2.8.11 or
	# later. Otherwise we have to add them here ourselves.
	if (CMAKE_VERSION VERSION_LESS 2.8.11)
	  include_directories("${gtest_SOURCE_DIR}/include")
	endif()

	add_executable(
		testDPGO
		tests/testUtils.cpp
		tests/testEigenMap.cpp
		tests/testConstruction.cpp
		tests/testChordalInitialization.cpp
		tests/testDCCICommunicator.cpp
		tests/testDCCIInitialization.cpp
		tests/testDCCILinearOperators.cpp
		tests/testDCCIPCGSolver.cpp
		tests/testTEDCCIFactorBuilder.cpp
		tests/testTEDCCIBoundarySelector.cpp
		tests/testTEDCCILocalQRCondensation.cpp
		tests/testTEDCCIInterfaceDirectSolver.cpp
		tests/testTEDCCISeparatorTree.cpp
		tests/testTEDCCIAsyncFactorExchange.cpp
		tests/testTEDCCIOrchestrator.cpp
		tests/testTEDCCISolver.cpp
		tests/testRIFTIF.cpp
		tests/testLineGraph.cpp
		tests/testTriangleGraph.cpp
		tests/testOptimizationThread.cpp
		tests/testManualQuadraticOptimizer.cpp
		tests/testManualDpgoMm.cpp
		tests/testManualObjectDran.cpp
		tests/testGeodesicSE3.cpp
		tests/testGeodesicObjectProblem.cpp
		tests/testObjectGaugeCoupling.cpp
		)
	target_include_directories(testDPGO PUBLIC
		${EXTERNAL_INCLUDES}
		${CMAKE_CURRENT_SOURCE_DIR}/include>
		${CMAKE_SOURCE_DIR}/eigen>
	)
	target_link_libraries(
		testDPGO
		gtest_main
		DPGO
		)
	add_test(
		NAME testDPGO
		COMMAND $<TARGET_FILE:testDPGO>
		)
	add_test(
		NAME chordal_initialization
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*ChordalInitialization*
		)
	add_test(
		NAME dcci_rotation_operator
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*DCCIRotationOperator*
		)
	add_test(
		NAME dcci_translation_operator
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*DCCITranslationOperator*
		)
	add_test(
		NAME dcci_pcg_solver
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*DCCIPCG*
		)
	add_test(
		NAME dcci_communicator
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*DCCIInProcessCommunicator*
		)
	add_test(
		NAME dcci_initialization
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*DCCIInitialization*
		)
	add_test(
		NAME rift_if
		COMMAND $<TARGET_FILE:testDPGO> --gtest_filter=*RIFTIF*
		)
	if(DCCI_MPI_ENABLED)
		add_test(
			NAME dcci_mpi_synthetic_3d_np2
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
			        $<TARGET_FILE:run_dcci_mpi> --dataset synthetic_3d
			)
		add_test(
			NAME dcci_mpi_synthetic_3d_np4
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
			        $<TARGET_FILE:run_dcci_mpi> --dataset synthetic_3d
			)
		add_test(
			NAME dcci_mpi_synthetic_3d_np8
			COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
			        $<TARGET_FILE:run_dcci_mpi> --dataset synthetic_3d
			)
		add_test(
			NAME ted_cci_mpi_point_to_point_np2
			COMMAND ${DPGO_TED_CCI_MPIEXEC} ${DPGO_TED_CCI_MPIEXEC_NUMPROC_FLAG} 2
			        $<TARGET_FILE:run_ted_cci_mpi>
			)
		add_test(
			NAME ted_cci_mpi_point_to_point_np4
			COMMAND ${DPGO_TED_CCI_MPIEXEC} ${DPGO_TED_CCI_MPIEXEC_NUMPROC_FLAG} 4
			        $<TARGET_FILE:run_ted_cci_mpi>
			)
	endif()
endif(BUILD_DPGO_TESTS)

############################### INSTALL ##########################################
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/DPGO)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/DPGOConfigVersion.cmake
  VERSION 1.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/DPGOConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DPGOConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DPGOConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${DPGO_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)

install(TARGETS DPGO
		EXPORT 	DPGOTargets
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		)

install(EXPORT DPGOTargets
		FILE DPGOTargets.cmake
		DESTINATION ${INSTALL_CONFIGDIR}
		)

export(TARGETS DPGO FILE DPGOTargets.cmake)

############################### UNINSTALL ##########################################
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
