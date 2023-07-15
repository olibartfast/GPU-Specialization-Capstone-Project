# The Underwater Trash Instance Segmentation YoloV8 Inference
I developed this project as final capstone of Coursera GPU Programming Specialization, this README provides instructions for performing c++ inference on the [The Underwater Trash Instance Segmentation Dataset](https://conservancy.umn.edu/handle/11299/214865) using various frameworks having both CPU and GPU backends, with the aim of showcasing the differences in inference time between these two devices. The dataset with all the training procedure is available at the link [https://learnopencv.com/train-yolov8-instance-segmentation/](https://learnopencv.com/train-yolov8-instance-segmentation/).  

I tested the following frameworks and backends:

1. ONNX Runtime (CPU/GPU Backend)
3. LibTorch (CPU/GPU)
5. TensorRT

## Prerequisites

Before proceeding, ensure that you have the following dependencies installed:

- C++ compiler with C++17 support
- CUDA if you want to use GPU, CUDA 12 is supported for LibTorch and TensorRT, I used CUDA 11.8 for onnx-rt
- CMake (used 3.22.1)
- ONNX Runtime 1.15.1 gpu package
- LibTorch 2.0.1-cu118
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
yolo export model=best.pt format=torchscript device=0
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

## Building and Running 
### Step 1: Configure CMake

To build the project with a specific backend, you need to configure CMake with the appropriate parameters. Open a terminal and navigate to the project directory. Then, follow the instructions based on your desired backend:

#### Build for a specific Backend

Run the following commands:

```shell
cmake -D FRAMEWORK=LIBTORCH(or ONNX_RUNTIME or TENSORRT) ..
cmake --build .
```
## Step 2: Run the Program

Once the build process is complete, you can run the program with the specified backend. Use the following command:

```shell
./UnderwaterTrashInstanceSegmentation --weights <path_to_weights> --video <path_to_video> --gpu=<true_or_false> --videowrite=<true_or_false> --imshow=<true_or_false>
```

Replace `<path_to_weights>` with the path to your weights file, `<path_to_video>` with the path to your video source. Set `<true_or_false>` for `--gpu` to `true` if you want to enable GPU acceleration (if available) or `false` to use CPU. Set `<true_or_false>` for `--videowrite` to `true` if you want to enable video writing, or `false` to disable it. Set `<true_or_false>` for `--imshow` to `true` if you want to enable frame display, or `false` to disable it.

### Example Usage

Here's an example command to run the program with LibTorch backend, using a `weights.pt` file, a `video.mp4` file, enabling video writing, and enabling frame display:

```shell
./UnderwaterTrashInstanceSegmentation --weights weights.pt --video video.mp4 --gpu=true --videowrite=true --imshow=true
```

Please adjust the command based on your specific file names and paths, as well as your desired configuration for video writing and frame display.

### Results
Here are the results obtained from the different frameworks and backends:

| Framework      | Backend | Average FPS |
|----------------|---------|-------------|
| LibTorch       | CPU     | 3.98        |
| LibTorch       | GPU     | 35.83       |
| ONNX Runtime   | CPU     | 4.22        |
| ONNX Runtime   | GPU     | 30.26       |
| TensorRT       | GPU     | 43.65       |

These results were obtained using the following hardware configuration:

- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU

### Example Video
You can find an example video demonstrating the inference process by clicking [here](data/processed_tensorrt.mp4).

### Additional Notes

- Make sure to adjust the paths and options in the CMakeLists.txt file according to your installation paths.
- If you don't have a suitable video for testing, you can use a video from the [Trash ICRA 2019 dataset](https://conservancy.umn.edu/handle/11299/214366). Please make sure to download the dataset and specify the path to the dataset video in the command. Tested the video called 'manythings.mp4'
