# The Underwater Trash Instance Segmentation YoloV8 Inference

This README provides instructions for performing c++ inference on the [The Underwater Trash Instance Segmentation Dataset](https://conservancy.umn.edu/handle/11299/214865) using various frameworks having both CPU and GPU backends, with the aim of showcasing the differences in inference time between these two devices. The dataset with all the training procedure are available at the link [https://learnopencv.com/train-yolov8-instance-segmentation/](https://learnopencv.com/train-yolov8-instance-segmentation/).  

I'm, planning to cover the following frameworks and backends:

1. ONNX Runtime (CPU/GPU Backend)
3. LibTorch (CPU/GPU)
5. TensorRT

## Prerequisites

Before proceeding, ensure that you have the following dependencies installed:

- CMake (version >= 3.12)
- ONNX Runtime 1.15.1
- LibTorch 
- TensorRT 8.6.1.6

## Instructions

### Export the model for the inference

To export the trained weights on the Underwater Trash Dataset, follow these steps:

1. Download the trained weights from the [Dropbox link](https://www.dropbox.com/scl/fo/cjj6w4q3679w1n03211zr/h?dl=1&rlkey=z16lunmbuwsn98we968psulse) available in the [learnopencv GitHub repository](https://github.com/spmallick/learnopencv/tree/master/Train-YOLOv8-Instance-Segmentation-on-Custom-Data).
2. Once you have downloaded the file, unzip it. You will find the weights under the path `Train-YOLOv8-Instance-Segmentation-on-Custom-Data/runs/segment/`.

Then install YoloV8 [following official documentation](https://docs.ultralytics.com/quickstart/) and export the model in different formats, you can use the following commands:

#### Torchscript

To export the model in the TorchScript format:

```
yolo export model=best.pt format=torchscript
```

#### OnnxRuntime

To export the model in the ONNXRuntime format:

```
yolo export model=best.pt format=onnx
```

#### TensorRT

To export the model in the TensorRT format:

```
yolo export model=best.pt format=engine device=0
```

Please note that when using TensorRT, ensure that the version installed under Ultralytics python environment matches the C++ version you plan to use for inference. Another way to export the model is to use `trtexec` with the following command:

```
trtexec --onnx=best.onnx --saveEngine=best.engine
```

By following these steps, you can successfully export the model in the desired formats for further inference.



1. **For each inference framework**

   - Install by following the instructions provided in the official documentation.
   - Load the pretrained weights into the inference session.
   - Write a C++ program using the relative API to perform inference on the weights trainded on the dataset.
   - Measure the inference time by capturing the start and end times around the inference code.
   - Compile the C++ program using the appropriate compiler flags and link against the runtime CPU or GPU library.
   - Run the executable to perform inference on the dataset using the respective backend (CPU or GPU).

