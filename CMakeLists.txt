cmake_minimum_required(VERSION 3.24)
project(apollonius)

# Add paths to our CMake code to the module path, so they can be found automatically by CMake.
set(CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/CMake"
        ${CMAKE_MODULE_PATH}
)

set(apo_is_main_project FALSE)
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    if (WIN32)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib")
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
        SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib")
        SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
    endif ()

    set(apo_is_main_project TRUE)
endif ()


include(CheckLanguage)
check_language(CUDA)

set(APO_CUDA_ENABLED TRUE)
if(NOT CMAKE_CUDA_COMPILER)
    set(APO_CUDA_ENABLED FALSE)
endif()
include(CMake/compiler_flags.cmake)
include(CMake/add_dependency_folder.cmake)

add_subdirectory(apo)

if (apo_is_main_project)
    add_subdirectory(app)
endif ()