# yolo_detector

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
yolo export model=yolov8m.pt format=onnx opset=12 dynamic=True
```
You also need to download the class file (coco.names)
```
https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

## Sample code

### Detection

```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m.onnx", "coco.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let result = detector.draw_detections(mat.clone(), detections, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```
```rust
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m.onnx", "coco.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let detections_with_classes =
        detector.get_detections_with_classes(detections, 0.5, original_size);

    for (class_name, rect) in detections_with_classes {
        println!("Class: {}, Position: {:?}", class_name, rect);
    }

    Ok(())

//returns values
//Class: person, Position: Rect_ { x: 74, y: 875, width: 41, height: 112 }
//Class: car, Position: Rect_ { x: 184, y: 899, width: 499, height: 141 }
}
```

### Weights
```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetectorWeights;

fn main() -> opencv::Result<()> {
    let mut detector =
        YoloDetectorWeights::new("yolov4.weights", "yolov4.cfg", "coco.names").unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (class_ids, confidences, boxes) = detector.detect(&mat.clone(), 0.7, 0.4)?;

    let result = detector.draw_detections(&mut mat.clone(), class_ids, confidences, boxes)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```

### OBB
DOTAv1.names
```
plane
ship
storage tank
baseball diamond
tennis court
basketball court
ground track field
harbor
bridge
large vehicle
small vehicle
helicopter
roundabout
soccer ball field
swimming pool
```

```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-obb.onnx", "DOTAv1.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let result = detector.draw_detections_obb(mat.clone(), detections, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```

```rust
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-obb.onnx", "DOTAv1.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let detections_with_classes =
        detector.get_detections_with_classes_obb(detections, 0.5, original_size);

    for (class_name, rect, rotation_angle) in detections_with_classes {
        println!(
            "Class: {}, Position: {:?}, Rotation Angle: {}°",
            class_name, rect, rotation_angle
        );
    }

    Ok(())

//returns values
// Class: ship, Position: Rect_ { x: 110, y: 738, width: 84, height: 25 }, Rotation Angle: 77.65746°
// Class: ship, Position: Rect_ { x: 576, y: 733, width: 65, height: 23 }, Rotation Angle: 56.169453°
}
```

## Project roadmap

- [x] Detection
- [x] Weights
- [x] OBB
- [ ] Classification
- [ ] Pose
- [ ] Segmentation

## Author

Developed by Elieren https://github.com/Elieren .

When using the library, keep an indication of the author.