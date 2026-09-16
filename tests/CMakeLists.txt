# Setup gtest
find_package(GTest REQUIRED)
enable_testing() # enable ctest

add_executable(form-tests
    test_Diagnostics.cpp
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
    target_sources(form-tests PRIVATE test_CudaExtraction.cpp test_CudaQr.cpp test_CudaSolver.cpp test_CudaMatcher.cpp test_CudaMatching.cpp)
endif()
include(GoogleTest)
gtest_discover_tests(form-tests)

# Make runnable from "make check" or "make test"


# Isolated process avoids the original constructor's permanent first-call TBB cap.
add_executable(form-extraction-parallel-tests test_ParallelExtraction.cpp)
target_link_libraries(form-extraction-parallel-tests PRIVATE GTest::gtest_main FORM)
gtest_discover_tests(form-extraction-parallel-tests)

if(FORM_ENABLE_CUDA)
    # The helper is CUDA-only code; the caller selects its own Eigen reduction policy.
    add_executable(form-extraction-scalar-tests test_FeatureExtraction.cpp)
    target_compile_definitions(form-extraction-scalar-tests PRIVATE EIGEN_DONT_VECTORIZE=1)
    target_link_libraries(form-extraction-scalar-tests PRIVATE GTest::gtest_main FORM)
    add_test(NAME FeatureExtraction.ScalarEigenParity COMMAND form-extraction-scalar-tests)
endif()

set(FORM_CHECK_TARGETS form-tests form-extraction-parallel-tests)
if(FORM_ENABLE_CUDA)
    list(APPEND FORM_CHECK_TARGETS form-extraction-scalar-tests)
endif()
foreach(test_target IN LISTS FORM_CHECK_TARGETS)
    list(APPEND FORM_CHECK_COMMANDS COMMAND $<TARGET_FILE:${test_target}>)
endforeach()
add_custom_target(check ${FORM_CHECK_COMMANDS} DEPENDS ${FORM_CHECK_TARGETS})
