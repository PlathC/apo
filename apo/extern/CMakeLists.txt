if (apo_is_main_project AND NOT chemfiles)
    add_subdirectory(chemfiles)
endif ()

if (NOT TARGET fmt)
    add_subdirectory(fmt)
endif()

set(apo_extern_libraries
        chemfiles
        fmt
)
set(apo_extern_includes
        ${apo_extern_includes}
        ${CMAKE_CURRENT_SOURCE_DIR}/plf_nanotimer
)

if(APO_CUDA_ENABLED)
    if(NOT TARGET cuda_helper)
        add_subdirectory(cuda_helper)


        set(apo_extern_libraries ${apo_extern_libraries} cuda_helper )
        set(apo_extern_includes ${apo_extern_includes}
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda_helper
        )
    endif()
endif()


set(apo_extern_includes ${apo_extern_includes} PARENT_SCOPE)
set(apo_extern_libraries ${apo_extern_libraries} PARENT_SCOPE)
