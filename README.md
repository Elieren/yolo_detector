# yolo-rust

## Pre-launch installation
```
sudo apt update -y
```
```
sudo apt install libopencv-dev pkg-config build-essential cmake libgtk-3-dev libcanberra-gtk3-module llvm-dev libclang-dev clang
```

## Converting the model to onnx
This library uses tipo models.onnx

```
pip install ultralytics
```
Download the model you need.
```
https://huggingface.co/Ultralytics/YOLOv8/tree/main
```
```
yolo export model=yolov8m.pt format=onnx opset=12
```
You also need to download the class file (coco.names)
```
https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

## Sample code

```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m.onnx", "coco.names").unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let result = detector.draw_detections(mat.clone(), detections, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```

## Author

Developed by Elieren https://github.com/Elieren .

When using the library, keep an indication of the author.