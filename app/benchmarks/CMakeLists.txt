define_cxx_flags(statistics_compile_flags statistics_compile_definition statistics_link_options)
define_cuda_flags(statistics_cuda_flags)

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/../cavity/CMake"
        ${CMAKE_MODULE_PATH}
)

enable_language(CUDA)
find_package(OptiX REQUIRED)
find_package(OpenMP REQUIRED) # Voronota relies on OpenMP for multithreading

# statistics-apovsvoronota-proteins
add_executable(statistics-apovsvoronota-proteins full_benchmarks_proteins.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-proteins PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-proteins PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-proteins PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-apovsvoronota-proteins PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-proteins PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-proteins SYSTEM PRIVATE ../extern)

## CUDA
target_compile_options(statistics-apovsvoronota-proteins PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-proteins PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-proteins PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-proteins PRIVATE cuda_std_17)

# statistics-apovsvoronota-proteins-fp32
add_executable(statistics-apovsvoronota-proteins-fp32 full_benchmarks_proteins.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-proteins-fp32 PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-proteins-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-proteins-fp32 PRIVATE ${statistics_compile_definition} APO_REAL_SIZE=4)
target_link_libraries(statistics-apovsvoronota-proteins-fp32 PRIVATE apo-fp32 ApoAppCommon-fp32 OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-proteins-fp32 PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-proteins-fp32 SYSTEM PRIVATE ../extern)

## CUDA
target_compile_options(statistics-apovsvoronota-proteins-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-proteins-fp32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-proteins-fp32 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-proteins-fp32 PRIVATE cuda_std_17)

# statistics-apovsvoronota-clouds
add_executable(statistics-apovsvoronota-clouds full_benchmarks_clouds.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-clouds PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-clouds PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-clouds PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-apovsvoronota-clouds PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-clouds PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-clouds SYSTEM PRIVATE ../extern)

target_compile_options(statistics-apovsvoronota-clouds PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-clouds PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-clouds PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-clouds PRIVATE cuda_std_17)

# statistics-apovsvoronota-clouds-fp32
add_executable(statistics-apovsvoronota-clouds-fp32 full_benchmarks_clouds.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-clouds-fp32 PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-clouds-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-clouds-fp32 PRIVATE ${statistics_compile_definition} APO_REAL_SIZE=4)
target_link_libraries(statistics-apovsvoronota-clouds-fp32 PRIVATE apo-fp32 ApoAppCommon-fp32 OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-clouds-fp32 PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-clouds-fp32 SYSTEM PRIVATE ../extern)

target_compile_options(statistics-apovsvoronota-clouds-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-clouds-fp32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-clouds-fp32 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-clouds-fp32 PRIVATE cuda_std_17)

# statistics-apovsvoronota-dataset
add_executable(statistics-apovsvoronota-dataset full_benchmarks_dataset.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-dataset PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-dataset PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-dataset PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-apovsvoronota-dataset PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-dataset PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-dataset SYSTEM PRIVATE ../extern)

target_compile_options(statistics-apovsvoronota-dataset PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-dataset PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-dataset PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-dataset PRIVATE cuda_std_17)
add_dependencies(statistics-apovsvoronota-dataset DATASET_SAMPLES)

# statistics-apovsvoronota-dataset
add_executable(statistics-apovsvoronota-dataset-fp32 full_benchmarks_dataset.cpp benchmark.cu)

## C++
target_compile_features(statistics-apovsvoronota-dataset-fp32 PRIVATE cxx_std_17)
target_compile_options(statistics-apovsvoronota-dataset-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-apovsvoronota-dataset-fp32 PRIVATE ${statistics_compile_definition} APO_REAL_SIZE=4)
target_link_libraries(statistics-apovsvoronota-dataset-fp32 PRIVATE apo-fp32 ApoAppCommon-fp32 OpenMP::OpenMP_CXX)
target_link_options(statistics-apovsvoronota-dataset-fp32 PRIVATE ${statistics_link_options})
target_include_directories(statistics-apovsvoronota-dataset-fp32 SYSTEM PRIVATE ../extern)

target_compile_options(statistics-apovsvoronota-dataset-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-apovsvoronota-dataset-fp32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-apovsvoronota-dataset-fp32 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-apovsvoronota-dataset-fp32 PRIVATE cuda_std_17)
add_dependencies(statistics-apovsvoronota-dataset-fp32 DATASET_SAMPLES)

# statistics-benchmarks-cavity
add_executable(statistics-benchmarks-cavity benchmark_cavity.cu
        ../cavity/utils.cu
        ../cavity/optix/context.cu
        ../cavity/optix/geometry.cu
        ../cavity/optix/pipeline.cu
        ../cavity/optix/geometry_pipeline.cu
        ../cavity/optix/program.cu
        ../cavity/optix/sphere_module.cu)

## C++
target_compile_features(statistics-benchmarks-cavity PRIVATE cxx_std_17)
target_compile_options(statistics-benchmarks-cavity PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-benchmarks-cavity PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-benchmarks-cavity PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(statistics-benchmarks-cavity PRIVATE ${statistics_link_options})
target_include_directories(statistics-benchmarks-cavity SYSTEM PRIVATE ../cavity . ${OptiX_INCLUDE})
target_compile_options(statistics-benchmarks-cavity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-benchmarks-cavity PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-benchmarks-cavity PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-benchmarks-cavity PRIVATE cuda_std_17)
add_dependencies(statistics-benchmarks-cavity cavity-ptx-compilation)


# statistics-benchmarks-knn
add_executable(statistics-benchmarks-knn benchmark_knn.cu)

## C++
target_compile_features(statistics-benchmarks-knn PRIVATE cxx_std_17)
target_compile_options(statistics-benchmarks-knn PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-benchmarks-knn PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-benchmarks-knn PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(statistics-benchmarks-knn PRIVATE ${statistics_link_options})
target_compile_options(statistics-benchmarks-knn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-benchmarks-knn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-benchmarks-knn PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-benchmarks-knn PRIVATE cuda_std_17)


# statistics-benchmarks-stages
add_executable(statistics-benchmarks-stages benchmark_stages.cu)

## C++
target_compile_features(statistics-benchmarks-stages PRIVATE cxx_std_17)
target_compile_options(statistics-benchmarks-stages PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-benchmarks-stages PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-benchmarks-stages PRIVATE apo ApoAppCommon)
target_link_options(statistics-benchmarks-stages PRIVATE ${statistics_link_options})
target_include_directories(statistics-benchmarks-stages SYSTEM PRIVATE ../extern)

## CUDA
target_compile_options(statistics-benchmarks-stages PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-benchmarks-stages PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-benchmarks-stages PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-benchmarks-stages PRIVATE cuda_std_17)


# statistics-benchmarks-contributing-nearest
add_executable(statistics-benchmarks-contributing-nearest benchmark_contributing_nearest.cu)
target_compile_features(statistics-benchmarks-contributing-nearest PRIVATE cxx_std_17)
target_compile_options(statistics-benchmarks-contributing-nearest PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${statistics_compile_flags}>)
target_compile_definitions(statistics-benchmarks-contributing-nearest PRIVATE ${statistics_compile_definition})
target_link_libraries(statistics-benchmarks-contributing-nearest PRIVATE apo ApoAppCommon)
target_link_options(statistics-benchmarks-contributing-nearest PRIVATE ${statistics_link_options})

target_compile_options(statistics-benchmarks-contributing-nearest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${statistics_cuda_flags}>)
set_target_properties(statistics-benchmarks-contributing-nearest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(statistics-benchmarks-contributing-nearest PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(statistics-benchmarks-contributing-nearest PRIVATE cuda_std_17)
