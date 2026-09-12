# First thing, we check if vcpkg is installed and if so, use it for dependencies
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE
        "${CMAKE_CURRENT_SOURCE_DIR}/.vcpkg/scripts/buildsystems/vcpkg.cmake"
    )
endif()

cmake_minimum_required(VERSION 3.24)

project(form VERSION 0.2.0 LANGUAGES CXX) # x-release-please-version

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)

option(FORM_BUILD_TESTS "Build FORM Tests" OFF)
option(FORM_BUILD_PYTHON "Build python evalio wrapper" OFF)
option(FORM_BUILD_BENCHMARKS "Build standalone scan replay benchmark" OFF)
option(FORM_ENABLE_CUDA "Build CUDA QR batch compression" OFF)

# ------------------------- Dependencies ------------------------- #
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(TBB REQUIRED)

include(FetchContent)
FetchContent_Declare(
    tsl
    GIT_REPOSITORY https://github.com/Tessil/robin-map.git
    GIT_TAG v1.4.1
		FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(tsl)

# ------------------------- form ------------------------- #
add_library(FORM
	form/form.cpp
	form/feature/factor.cpp
	form/feature/summary.cpp
	form/optimization/constraints.cpp
	form/mapping/keyscanner.cpp
)
target_sources(FORM
	PUBLIC
	FILE_SET form_headers
	TYPE HEADERS
	BASE_DIRS .
	FILES 
		# feature
		form/feature/factor.hpp
		form/feature/summary.hpp
		form/feature/cuda_qr.hpp
		form/feature/features.hpp
		form/feature/extraction.hpp
		# optimization
		form/optimization/constraints.hpp
		form/optimization/gtsam.hpp
		form/optimization/cuda_solver.hpp
		form/optimization/profile.hpp
		form/optimization/matcher.hpp
		# mapping
		form/mapping/map.hpp
		form/mapping/keyscanner.hpp
		# core
		form/form.hpp
		form/utils.hpp
)
target_link_libraries(FORM
	PUBLIC
	gtsam
	Eigen3::Eigen
	TBB::tbb
	tsl::robin_map
)

if(FORM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(FORM PRIVATE form/feature/cuda_qr.cu form/optimization/cuda_solver.cpp)
    target_link_libraries(FORM PUBLIC CUDA::cudart CUDA::cusolver)
    target_compile_definitions(FORM PUBLIC FORM_ENABLE_CUDA=1)
    set_target_properties(FORM PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

# Tests
if(${FORM_BUILD_TESTS})
	enable_testing()
	add_subdirectory(tests)
endif()

if(FORM_BUILD_BENCHMARKS)
    add_executable(form-replay benchmarks/replay.cpp)
    target_link_libraries(form-replay PRIVATE FORM)
    if(FORM_ENABLE_CUDA)
        add_executable(form-qr-benchmark benchmarks/qr_benchmark.cpp)
        target_link_libraries(form-qr-benchmark PRIVATE FORM)
        add_executable(form-qr-tiles benchmarks/qr_tile_benchmark.cpp)
        target_link_libraries(form-qr-tiles PRIVATE FORM)
    endif()
endif()

# python bindings
if(${FORM_BUILD_PYTHON})
	add_subdirectory(python)
endif()
