set(ultroids_sources ultroids.cpp)

set(ultroids_thirdparty_libs chai umpire RAJA)

if(CUDA_FOUND)
    list(APPEND ultroids_thirdparty_libs cuda_runtime)
endif()

if(ENABLE_OPENMP)
    list(APPEND ultroids_thirdparty_libs openmp)
endif()

blt_add_executable(
    NAME        ultroids
    SOURCES     ${ultroids_sources}
    DEPENDS_ON  ${ultroids_thirdparty_libs})
