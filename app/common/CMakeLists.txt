add_library(ApoAppCommon samples.cpp)
target_link_libraries(ApoAppCommon PRIVATE apo)
target_include_directories(ApoAppCommon PUBLIC ../)
target_compile_features(ApoAppCommon PRIVATE cxx_std_17)

add_dependency_folder(APO_SAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/samples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/samples)
add_dependencies(ApoAppCommon APO_SAMPLES)

add_dependency_folder(DATASET_SAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/../extern/Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Benchmark-Dataset-for-the-Voronoi-Diagram-of-3D-Spherical-Balls)

# FP32
add_library(ApoAppCommon-fp32 samples.cpp)
target_link_libraries(ApoAppCommon-fp32 PRIVATE apo-fp32)
target_include_directories(ApoAppCommon-fp32 PUBLIC ../)
target_compile_features(ApoAppCommon-fp32 PRIVATE cxx_std_17)
target_compile_definitions(ApoAppCommon-fp32 PRIVATE APO_REAL_SIZE=4)

add_dependencies(ApoAppCommon-fp32 APO_SAMPLES)
