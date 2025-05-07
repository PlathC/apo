define_cxx_flags(topology_compile_flags topology_compile_definition topology_link_options)

if (APO_CUDA_ENABLED)
    define_cuda_flags(topology_cuda_flags)
    enable_language(CUDA)
    find_package(OpenMP REQUIRED) # Voronota relies on OpenMP for multithreading

    # topology-apo
    add_executable(topology-apo topology_apo.cu topology.cpp)

    ## C++
    target_compile_features(topology-apo PRIVATE cxx_std_17)
    target_compile_options(topology-apo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${topology_compile_flags}>)
    target_compile_definitions(topology-apo PRIVATE ${topology_compile_definition})
    target_link_libraries(topology-apo PRIVATE apo ApoAppCommon)
    target_link_options(topology-apo PRIVATE ${topology_link_options})
    target_include_directories(topology-apo SYSTEM PRIVATE ../extern)

    ## CUDA
    target_compile_options(topology-apo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${topology_cuda_flags}>)
    set_target_properties(topology-apo PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    set_target_properties(topology-apo PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_features(topology-apo PRIVATE cuda_std_17)
    add_dependencies(topology-apo DATASET_SAMPLES)


    # topology-apo-fp32
    add_executable(topology-apo-fp32 topology_apo.cu topology.cpp)

    ## C++
    target_compile_features(topology-apo-fp32 PRIVATE cxx_std_17)
    target_compile_options(topology-apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${topology_compile_flags}>)
    target_compile_definitions(topology-apo-fp32 PRIVATE ${topology_compile_definition} APO_REAL_SIZE=4)
    target_link_libraries(topology-apo-fp32 PRIVATE apo-fp32 ApoAppCommon-fp32)
    target_link_options(topology-apo-fp32 PRIVATE ${topology_link_options})
    target_include_directories(topology-apo-fp32 SYSTEM PRIVATE ../extern)

    ## CUDA
    target_compile_options(topology-apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${topology_cuda_flags}>)
    set_target_properties(topology-apo-fp32 PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    set_target_properties(topology-apo-fp32 PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_features(topology-apo-fp32 PRIVATE cuda_std_17)
    add_dependencies(topology-apo-fp32 DATASET_SAMPLES)
endif ()

# topology-voronota
add_executable(topology-voronota topology_voronota.cpp topology.cpp)

## C++
target_compile_features(topology-voronota PRIVATE cxx_std_17)
target_compile_options(topology-voronota PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${topology_compile_flags}>)
target_compile_definitions(topology-voronota PRIVATE ${topology_compile_definition})
target_link_libraries(topology-voronota PRIVATE apo ApoAppCommon OpenMP::OpenMP_CXX)
target_link_options(topology-voronota PRIVATE ${topology_link_options})
target_include_directories(topology-voronota SYSTEM PRIVATE ../extern)

## CUDA
target_compile_options(topology-voronota PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${topology_cuda_flags}>)
set_target_properties(topology-voronota PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(topology-voronota PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(topology-voronota PRIVATE cuda_std_17)
add_dependencies(topology-voronota DATASET_SAMPLES)

# topology-robust
add_executable(topology-robust topology_robust.cpp topology.cpp)

## C++
target_compile_features(topology-robust PRIVATE cxx_std_20)
target_compile_options(topology-robust PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${topology_compile_flags}>)
target_compile_definitions(topology-robust PRIVATE ${topology_compile_definition})
target_link_libraries(topology-robust PRIVATE apo ApoAppCommon
        robust_implicit_networks implicit_functions::implicit_functions)
target_link_options(topology-robust PRIVATE ${topology_link_options})
target_include_directories(topology-robust SYSTEM PRIVATE ../extern)

add_dependencies(topology-robust DATASET_SAMPLES)