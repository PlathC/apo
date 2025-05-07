define_cxx_flags(statistics_compile_flags statistics_compile_definition statistics_link_options)
define_cuda_flags(statistics_cuda_flags)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(statistics-benchmarks-memory main.cu)
target_compile_features(statistics-benchmarks-memory PRIVATE cxx_std_17)
set_target_properties(statistics-benchmarks-memory PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(statistics-benchmarks-memory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(statistics-benchmarks-memory PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-benchmarks-memory PRIVATE apo ApoAppCommon CUDA::cupti)
target_compile_options(statistics-benchmarks-memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
