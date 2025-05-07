if (NOT TARGET chemfiles)
    add_subdirectory(chemfiles)
endif ()

if (NOT TARGET polyscope)
    add_subdirectory(polyscope)
endif ()

add_subdirectory(Robust-Implicit-Surface-Networks)
