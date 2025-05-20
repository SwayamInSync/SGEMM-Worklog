cmake_minimum_required(VERSION 3.14)
project(sgemm_worklog CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 89 90)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified, defaulting to Release")
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Building in Debug mode - performance will be severely impacted!")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

add_executable(sgemm_runner runner.cu)
target_include_directories(sgemm_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sgemm_runner PRIVATE cublas)

set_target_properties(sgemm_runner
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS sgemm_runner DESTINATION bin)