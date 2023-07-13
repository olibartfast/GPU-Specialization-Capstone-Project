cmake_minimum_required(VERSION 3.12)
project(UnderwaterTrashInstanceSegmentation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set the path to the selected framework (modify accordingly)
set(FRAMEWORK "TENSORRT")  # Options: ONNX_RUNTIME, LIBTORCH, TENSORRT

# Set the source files for your project
set(SOURCES main.cpp)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Unset cache compiler definitions for the selected framework
if(FRAMEWORK STREQUAL "ONNX_RUNTIME")
    unset(USE_LIBTORCH CACHE)
    unset(USE_TENSORRT CACHE)
    message(STATUS "Set onnx runtime")
    add_compile_definitions(USE_ONNX_RUNTIME)
    # Set the path to ONNX Runtime (modify accordingly)
    set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-gpu-1.15.1)
    include_directories(${ONNX_RUNTIME_DIR}/include)
    link_directories(${ONNX_RUNTIME_DIR}/lib)

elseif(FRAMEWORK STREQUAL "LIBTORCH")
    unset(USE_ONNX_RUNTIME CACHE)
    unset(USE_TENSORRT CACHE)
    message(STATUS "Set libtorch")
    add_compile_definitions(USE_LIBTORCH)
    # Set the path to LibTorch (modify accordingly)
    set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)
    message(STATUS "${Torch_DIR}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

elseif(FRAMEWORK STREQUAL "TENSORRT")
    unset(USE_ONNX_RUNTIME CACHE)
    unset(USE_LIBTORCH CACHE)
    message(STATUS "Set tensorrt")
    add_compile_definitions(USE_TENSORRT)
    # Set the path to TensorRT (modify accordingly)
    set(TENSORRT_DIR $ENV{HOME}/TensorRT-8.6.1.6/)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-g;-G;-gencode;arch=compute_75;code=sm_75)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(${TENSORRT_DIR}/include)
    link_directories(${TENSORRT_DIR}/lib)
else()
    message(FATAL_ERROR "Invalid framework selected")
endif()

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link against OpenCV and the selected framework libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

# Link against the ONNX Runtime library if selected
if(FRAMEWORK STREQUAL "ONNX_RUNTIME")
    target_link_libraries(${PROJECT_NAME}
        ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )
endif()

# Link against the LibTorch library if selected
if(FRAMEWORK STREQUAL "LIBTORCH")
    target_link_libraries(${PROJECT_NAME}
        ${TORCH_LIBRARIES}
    )
endif()

# Link against the TensorRT library if selected
if(FRAMEWORK STREQUAL "TENSORRT")
    target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart)
endif()

# Set the appropriate compiler flags
if(CMAKE_CUDA_COMPILER AND FRAMEWORK STREQUAL "TENSORRT")
    # If CUDA is available and TensorRT is selected, set the CUDA flags
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
else()
    # If CUDA is not available or a different framework is selected, set the CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
