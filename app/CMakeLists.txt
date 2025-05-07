add_subdirectory(extern)

add_subdirectory(common)
if(APO_CUDA_ENABLED)
    add_subdirectory(benchmarks)
    add_subdirectory(cavity)
    add_subdirectory(visualization)
endif()

# For interval arithmetic
find_package(Boost)
if(Boost_FOUND)
    add_subdirectory(precision)
endif()

add_subdirectory(topology)

if(APO_CUDA_ENABLED)
    find_package(CUDAToolkit)
    if (${CUDAToolkit_FOUND})
        add_subdirectory(memory)
    else ()
        message("CUDA Toolkit not found, disable memory benchmark")
    endif()
endif()