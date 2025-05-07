define_cxx_flags(cavity_detection_compile_flags cavity_detection_compile_definition cavity_detection_link_options)
define_cuda_flags(cavity_detection_cuda_flags)

enable_language(CUDA)

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/CMake"
        ${CMAKE_MODULE_PATH}
)

include(CMake/compile_to_ptx.cmake)

find_package(OptiX REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cavity-detection-gpu main.cpp utils.cu
        optix/context.cu
        optix/geometry.cu
        optix/pipeline.cu
        optix/geometry_pipeline.cu
        optix/program.cu
        optix/sphere_module.cu
)
target_compile_features(cavity-detection-gpu PRIVATE cxx_std_17)
target_compile_options(cavity-detection-gpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cavity_detection_compile_flags}>)
target_compile_definitions(cavity-detection-gpu PRIVATE ${cavity_detection_compile_definition})
target_link_libraries(cavity-detection-gpu PRIVATE ApoAppCommon apo polyscope)
target_link_options(cavity-detection-gpu PRIVATE ${cavity_detection_link_options})
target_include_directories(cavity-detection-gpu PRIVATE . ${OptiX_INCLUDE})

set_target_properties(cavity-detection-gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(cavity-detection-gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cavity_detection_cuda_flags}>)
set_target_properties(cavity-detection-gpu PROPERTIES CUDA_ARCHITECTURES native)

set(cavity-optix-shaders
        shaders/find_occlusion.cu
)

set(cavity-optix-shaders-includes -Ishaders/ -I${OptiX_INCLUDE})
get_target_property(apo_includes apo INCLUDE_DIRECTORIES)
get_target_property(cuda_helper_includes cuda_helper INCLUDE_DIRECTORIES)
foreach (includes IN LISTS apo_includes)
    set(cavity-optix-shaders-includes ${cavity-optix-shaders-includes} -I${includes})
endforeach ()

compile_to_ptx(
        SOURCES ${cavity-optix-shaders}
        TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ptx"
        GENERATED_FILES cavity-optix-ptx-shaders
        NVCC_OPTIONS ${cuda_flags} ${cavity-optix-shaders-includes} -std=c++17
)

add_custom_target(cavity-ptx-compilation DEPENDS ${cavity-optix-ptx-shaders})
add_dependencies(cavity-detection-gpu cavity-ptx-compilation)
