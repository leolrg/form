include(FetchContent)

find_package(
    Python
    3.11
    REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.13.0
        FIND_PACKAGE_ARGS 2.13.0 EXACT CONFIG
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
    evalio
    GIT_REPOSITORY https://github.com/contagon/evalio.git
    GIT_TAG "v0.6.1"
        FIND_PACKAGE_ARGS 0.6.1 EXACT
)
FetchContent_MakeAvailable(evalio)

set_property(TARGET FORM PROPERTY POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_core STABLE_ABI MODULE bindings.cpp)
target_link_libraries(_core PRIVATE evalio FORM)
install(TARGETS _core DESTINATION form)

nanobind_add_stub(
    form_stubs
    INSTALL_TIME
    MODULE form._core
    OUTPUT form/_core.pyi # install directly to final location
    PYTHON_PATH "."
)