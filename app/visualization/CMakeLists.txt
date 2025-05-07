define_cxx_flags(visualization_compile_flags visualization_compile_definition visualization_link_options)
define_cuda_flags(visualization_cuda_flags)

enable_language(CUDA)

find_package(OpenMP REQUIRED) # Voronota relies on OpenMP for multithreading

add_executable(visualization main.cpp)
target_compile_features(visualization PRIVATE cxx_std_17)
target_compile_options(visualization PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${visualization_compile_flags}>)
target_compile_definitions(visualization PRIVATE ${visualization_compile_definition})
target_link_libraries(visualization PRIVATE ApoAppCommon apo polyscope OpenMP::OpenMP_CXX)
target_link_options(visualization PRIVATE ${visualization_link_options})
target_include_directories(visualization PRIVATE .)

set_target_properties(visualization PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(visualization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cavity_detection_cuda_flags}>)
set_target_properties(visualization PROPERTIES CUDA_ARCHITECTURES native)