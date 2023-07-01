cmake_minimum_required(VERSION 3.12)
project(UnderwaterTrashInstanceSegmentation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Set the path to the selected framework (modify accordingly)
set(FRAMEWORK "ONNX_RUNTIME")  # Options: ONNX_RUNTIME, LIBTORCH, TENSORRT

# Set the source files for your project
set(SOURCES main.cpp)

# Include directories and link libraries based on the selected framework
if(FRAMEWORK STREQUAL "ONNX_RUNTIME")
    # Set the path to ONNX Runtime (modify accordingly)
    set(ONNX_RUNTIME_DIR $ENV{HOME}/onnxruntime-linux-x64-1.15.0)

    # Include directories
    include_directories(${ONNX_RUNTIME_DIR}/include)

    # Link directories
    link_directories(${ONNX_RUNTIME_DIR}/lib)

    # Create the executable
    add_executable(${PROJECT_NAME} ${SOURCES})

    # Link against the ONNX Runtime library
    target_link_libraries(${PROJECT_NAME}
        ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )

elseif(FRAMEWORK STREQUAL "LIBTORCH")
    # Set the path to LibTorch (modify accordingly)
    set(LIBTORCH_DIR /path/to/libtorch)

    # Include directories
    include_directories(${LIBTORCH_DIR}/include)

    # Link directories
    link_directories(${LIBTORCH_DIR}/lib)

    # Create the executable
    add_executable(${PROJECT_NAME} ${SOURCES})

    # Link against the LibTorch library
    target_link_libraries(${PROJECT_NAME}
        ${LIBTORCH_DIR}/lib/torch.lib
    )

elseif(FRAMEWORK STREQUAL "TENSORRT")
    # Set the path to TensorRT (modify accordingly)
    set(TENSORRT_DIR /path/to/tensorrt)

    # Include directories
    include_directories(${TENSORRT_DIR}/include)

    # Link directories
    link_directories(${TENSORRT_DIR}/lib)

    # Create the executable
    add_executable(${PROJECT_NAME} ${SOURCES})

    # Link against the TensorRT library
    target_link_libraries(${PROJECT_NAME}
        ${TENSORRT_DIR}/lib/nvinfer.lib
    )

else()
    message(FATAL_ERROR "Invalid framework selected")
endif()

# Set the appropriate compiler flags
if(CMAKE_CUDA_COMPILER AND FRAMEWORK STREQUAL "TENSORRT")
    # If CUDA is available and TensorRT is selected, set the CUDA flags
    set_target_properties(TrashICRAInference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
else()
    # If CUDA is not available or a different framework is selected, set the CPU flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
