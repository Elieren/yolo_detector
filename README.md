# yolo-rust

## Установка перед запуском
```
sudo apt update -y
```
```
sudo apt install libopencv-dev pkg-config build-essential cmake libgtk-3-dev libcanberra-gtk3-module llvm-dev libclang-dev clang
```

## Конвертация модели в onnx
Эта библиотка использует модели типо .onnx

```
pip install ultralytics
```
Скачайте нужную вам модель.
```
https://huggingface.co/Ultralytics/YOLOv8/tree/main
```
```
yolo export model=yolov8m.pt format=onnx opset=12
```
Так же нужно скачать файл классов (coco.names)
```
https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

## Пример кода

```rust
mod yolo_rust;

use opencv::{highgui, imgcodecs};
use yolo_rust::YoloDetector;

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

## Автор

Разработано Elieren https://github.com/Elieren.

При использовании библиотеки — сохраняйте указание на автора.
