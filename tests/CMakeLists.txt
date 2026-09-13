# Setup gtest
find_package(GTest REQUIRED)
enable_testing() # enable ctest

add_executable(form-tests
    test_FeatureFactor.cpp
    test_BatchSummary.cpp
    test_Summary.cpp
    test_SummaryIntegration.cpp
    test_ManagerIntegration.cpp
    test_ResidentOptimizer.cpp
    test_FeatureExtraction.cpp
)
target_link_libraries(form-tests 
    GTest::gtest_main
    FORM 
)
if(FORM_ENABLE_CUDA)
    target_sources(form-tests PRIVATE test_CudaQr.cpp test_CudaSolver.cpp)
endif()
include(GoogleTest)
gtest_discover_tests(form-tests)

# Make runnable from "make check" or "make test"
add_custom_target(check COMMAND form-tests)
