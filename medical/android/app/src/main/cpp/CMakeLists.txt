cmake_minimum_required(VERSION 3.14)

project("gguf_backend" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# Build llama.cpp as a static sub-library
# -----------------------------------------------------------------------
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

# Disable all extras we don't need for mobile inference
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

if (ANDROID_ABI STREQUAL "armeabi-v7a")
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(
    "${LLAMA_CPP_DIR}"
    "${CMAKE_BINARY_DIR}/llama_build"
    EXCLUDE_FROM_ALL
)

# -----------------------------------------------------------------------
# Our FFI wrapper shared library
# -----------------------------------------------------------------------
add_library(gguf_backend SHARED gguf_backend.cpp)

target_include_directories(gguf_backend PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

find_library(log-lib log)

target_link_libraries(gguf_backend
    llama
    ggml
    ${log-lib}
    m
)

target_compile_options(gguf_backend PRIVATE -O3 -fvisibility=hidden)
