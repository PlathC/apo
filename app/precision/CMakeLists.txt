define_cxx_flags(precision_compile_flags precision_compile_definition precision_link_options)

if (APO_CUDA_ENABLED)
    define_cuda_flags(precision_cuda_flags)
    enable_language(CUDA)
    find_package(OpenMP REQUIRED) # Voronota relies on OpenMP for multithreading

    # precision-apo
    add_executable(precision-apo precision_apo.cpp precision.cpp)

    ## C++
    target_compile_features(precision-apo PRIVATE cxx_std_17)
    target_compile_options(precision-apo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${precision_compile_flags}>)
    target_compile_definitions(precision-apo PRIVATE ${precision_compile_definition})
    target_link_libraries(precision-apo PRIVATE apo ApoAppCommon Boost::headers)
    target_link_options(precision-apo PRIVATE ${precision_link_options})
    target_include_directories(precision-apo SYSTEM PRIVATE ../extern)

    ## CUDA
    target_compile_options(precision-apo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${precision_cuda_flags}>)
    set_target_properties(precision-apo PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    set_target_properties(precision-apo PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_features(precision-apo PRIVATE cuda_std_17)
    add_dependencies(precision-apo DATASET_SAMPLES)

    # precision-apo-fp32
    add_executable(precision-apo-fp32 precision_apo.cpp precision.cpp)

    ## C++
    target_compile_features(precision-apo-fp32 PRIVATE cxx_std_17)
    target_compile_options(precision-apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${precision_compile_flags}>)
    target_compile_definitions(precision-apo-fp32 PRIVATE ${precision_compile_definition} APO_REAL_SIZE=4)
    target_link_libraries(precision-apo-fp32 PRIVATE apo-fp32 ApoAppCommon-fp32 Boost::headers)
    target_link_options(precision-apo-fp32 PRIVATE ${precision_link_options})
    target_include_directories(precision-apo-fp32 SYSTEM PRIVATE ../extern)

    ## CUDA
    target_compile_options(precision-apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${precision_cuda_flags}>)
    set_target_properties(precision-apo-fp32 PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    set_target_properties(precision-apo-fp32 PROPERTIES CUDA_ARCHITECTURES native)
    target_compile_features(precision-apo-fp32 PRIVATE cuda_std_17)
    add_dependencies(precision-apo-fp32 DATASET_SAMPLES)
endif ()

find_package(OpenMP REQUIRED) # Voronota relies on OpenMP for multithreading

# precision-voronota
add_executable(precision-voronota precision_voronota.cpp precision.cpp)

## C++
target_compile_features(precision-voronota PRIVATE cxx_std_17)
target_compile_options(precision-voronota PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${precision_compile_flags}>)
target_compile_definitions(precision-voronota PRIVATE ${precision_compile_definition})
target_link_libraries(precision-voronota PRIVATE apo ApoAppCommon Boost::headers OpenMP::OpenMP_CXX)
target_link_options(precision-voronota PRIVATE ${precision_link_options})
target_include_directories(precision-voronota SYSTEM PRIVATE ../extern)
add_dependencies(precision-voronota DATASET_SAMPLES)

# precision-robust
add_executable(precision-robust precision_robust.cpp precision.cpp)

## C++
target_compile_features(precision-robust PRIVATE cxx_std_20)
target_compile_options(precision-robust PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${precision_compile_flags}>)
target_compile_definitions(precision-robust PRIVATE ${precision_compile_definition})
target_link_libraries(precision-robust PRIVATE apo ApoAppCommon
        robust_implicit_networks implicit_functions::implicit_functions Boost::headers)
target_link_options(precision-robust PRIVATE ${precision_link_options})
target_include_directories(precision-robust SYSTEM PRIVATE ../extern)

add_dependencies(precision-robust DATASET_SAMPLES)