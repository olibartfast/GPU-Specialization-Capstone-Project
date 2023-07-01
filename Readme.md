# The Underwater Trash Instance Segmentation YoloV8 Inference

This README provides instructions for performing c++ inference on the [The Underwater Trash Instance Segmentation Dataset](https://conservancy.umn.edu/handle/11299/214865) using various frameworks having both CPU and GPU backends, with the aim of showcasing the differences in inference time between these two devices. The pretrained weights for the dataset with all the training procedure are available at the link [https://learnopencv.com/train-yolov8-instance-segmentation/](https://learnopencv.com/train-yolov8-instance-segmentation/).  

I'm, planning to cover the following frameworks and backends:

1. ONNX Runtime (CPU/GPU Backend)
3. LibTorch (CPU/GPU)
5. TensorRT

## TO DO
## Prerequisites

Before proceeding, ensure that you have the following dependencies installed:

- CMake (version >= 3.12)
- ONNX Runtime
- LibTorch 
- TensorRT 

## Instructions

1. **For each inference framework**

   - Install by following the instructions provided in the official documentation.
   - Load the pretrained weights into the inference session.
   - Write a C++ program using the relative API to perform inference on the weights trainded on the dataset.
   - Measure the inference time by capturing the start and end times around the inference code.
   - Compile the C++ program using the appropriate compiler flags and link against the runtime CPU or GPU library.
   - Run the executable to perform inference on the dataset using the respective backend (CPU or GPU).


## Conclusion

By following the above instructions, you can compare the inference times between different frameworks and backends (CPU and GPU). Measure the time taken by each backend to complete the inference and analyze the results. This will help you understand the performance differences and choose the best option based on your requirements.