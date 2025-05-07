define_cxx_flags(apo_compile_flags apo_compile_definition apo_link_options)
define_cuda_flags(apo_cuda_flags)

add_subdirectory(extern)

set(apo_includes
        include/apo/core/benchmark.hpp
        include/apo/core/logger.hpp
        include/apo/core/math.hpp
        include/apo/core/type.hpp
        include/apo/core/utils.hpp
        include/apo/gpu/algorithm.hpp
)

if(APO_CUDA_ENABLED)
    enable_language(CUDA)

    set(apo_includes ${apo_includes}

            include/apo/gpu/algorithm.cuh
            include/apo/gpu/benchmark.cuh
            include/apo/gpu/heap.hpp
            include/apo/gpu/lbvh.cuh
            include/apo/gpu/math.cuh
            include/apo/gpu/memory.cuh
            include/apo/gpu/setup.hpp
            include/apo/gpu/topology_update.cuh
            include/apo/gpu/utils.cuh
    )
endif()

set(apo_sources
        src/core/benchmark.cpp
        src/core/logger.cpp
        src/core/utils.cpp
)

if(APO_CUDA_ENABLED)
    set(apo_sources ${apo_sources}

            src/gpu/algorithm.cpp
            src/gpu/algorithm.cu
            src/gpu/benchmark.cu
            src/gpu/lbvh.cu
            src/gpu/memory.cu
    )
endif()

add_library(apo STATIC ${apo_includes} ${apo_sources})

# C++
target_include_directories(apo SYSTEM PUBLIC include/ ${apo_extern_includes})
target_compile_features(apo PRIVATE cxx_std_17)
target_compile_options(apo PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${apo_compile_flags}>)
target_compile_definitions(apo PRIVATE ${apo_compile_definition})
target_link_options(apo PRIVATE ${apo_link_options})
target_link_libraries(apo PUBLIC ${apo_extern_libraries})

# CUDA properties
set_target_properties(apo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(apo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${apo_cuda_flags}>)
set_target_properties(apo PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(apo PRIVATE cuda_std_17)

# FP32
add_library(apo-fp32 STATIC ${apo_includes} ${apo_sources})

# C++
target_include_directories(apo-fp32 SYSTEM PUBLIC include/ ${apo_extern_includes})
target_compile_features(apo-fp32 PRIVATE cxx_std_17)
target_compile_options(apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${apo_compile_flags}>)
target_compile_definitions(apo-fp32 PRIVATE ${apo_compile_definition} APO_REAL_SIZE=4)
target_link_options(apo-fp32 PRIVATE ${apo_link_options})
target_link_libraries(apo-fp32 PUBLIC ${apo_extern_libraries})

# CUDA properties
set_target_properties(apo-fp32 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(apo-fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${apo_cuda_flags}>)
set_target_properties(apo-fp32 PROPERTIES CUDA_ARCHITECTURES native)
target_compile_features(apo-fp32 PRIVATE cuda_std_17)