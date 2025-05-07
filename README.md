# yolo_detector

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![Yolo](https://img.shields.io/badge/yolo-black.svg?style=for-the-badge&logo=data:image/webp;base64,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)
![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)


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

    let result = detector.draw_detections(mat.clone(), detections, 0.5, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```
```rust
use yolo_detector::YoloDetector;
use opencv::imgcodecs;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m.onnx", "coco.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let detections_with_classes =
        detector.get_detections_with_classes(detections, 0.5, 0.5, original_size);

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

    let result = detector.draw_detections_obb(mat.clone(), detections, 0.5, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```

```rust
use yolo_detector::YoloDetector;
use opencv::imgcodecs;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-obb.onnx", "DOTAv1.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect(&mat.clone())?;

    let detections_with_classes =
        detector.get_detections_with_classes_obb(detections, 0.5, 0.5, original_size);

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

### Classification
ImageNet.names
```
https://github.com/Elieren/yolo_detector/blob/main/ImageNet.names
```

```rust
use yolo_detector::YoloDetector;
use opencv::imgcodecs;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-cls.onnx", "ImageNet.names", 224).unwrap();

    let mat = imgcodecs::imread("zebra.jpg", imgcodecs::IMREAD_COLOR)?;

    let result = detector.classify(&mat.clone(), 0.5)?;

    for (class_name, score) in result {
        println!("Class: {}, Score: {}", class_name, score);
    }
    
    Ok(())

//returns values
// Class: zebra, Score: 0.99995613
}
```

### Pose
pose.names
```
Nose
Left Eye
Right Eye
Left Ear
Right Ear
Left Shoulder
Right Shoulder
Left Elbow
Right Elbow
Left Wrist
Right Wrist
Left Hip
Right Hip
Left Knee
Right Knee
Left Ankle
Right Ankle
```

```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-pose.onnx", "pose.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect_pose(&mat.clone())?;

    let result = detector.draw_detections_pose(mat.clone(), detections, 0.5, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```
```rust
use yolo_detector::YoloDetector;
use opencv::imgcodecs;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-pose.onnx", "pose.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, original_size) = detector.detect_pose(&mat.clone())?;

    let result = detector.get_detections_with_classes_pose(detections, 0.5, 0.5, original_size);

    for (i, keypoints) in result.iter().enumerate() {
        println!("Person {}:", i + 1);
        for (name, point) in keypoints {
            println!("  {}: ({}, {})", name, point.x, point.y);
        }
    }

    Ok(())

//returns values
// Person 1:
//   Nose: (545, 98)
//   Right Shoulder: (524, 115)
//   Left Wrist: (556, 153)
//   Left Hip: (549, 155)
//   Left Shoulder: (552, 115)
//   Left Elbow: (555, 135)
//   Left Eye: (547, 96)
//   Right Eye: (541, 96)
//   Right Wrist: (520, 154)
//   Right Knee: (530, 183)
//   Right Ear: (532, 98)
//   Right Hip: (530, 155)
//   Right Elbow: (516, 136)
// Person 2:
//   Left Hip: (829, 160)
//   Left Knee: (834, 185)
//   Left Shoulder: (820, 122)
//   Right Ear: (851, 102)
//   Left Elbow: (817, 142)
//   Left Ankle: (837, 210)
//   Right Ankle: (847, 210)
//   Left Ear: (824, 103)
//   Right Knee: (849, 185)
//   Right Shoulder: (861, 121)
//   Right Hip: (856, 160)
}
```

### Segmentation
```rust
use opencv::{highgui, imgcodecs};
use yolo_detector::YoloDetector;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-seg.onnx", "coco.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, mask, original_size) = detector.detect_mask(&mat.clone())?;

    let result =
        detector.draw_detections_masked(mat.clone(), detections, mask, 0.5, 0.5, original_size)?;

    highgui::imshow("YOLOv8 Video", &result)?;
    highgui::wait_key(0)?;

    Ok(())
}
```

```rust
use yolo_detector::YoloDetector;
use opencv::imgcodecs;

fn main() -> opencv::Result<()> {
    let detector = YoloDetector::new("yolov8m-seg.onnx", "coco.names", 640).unwrap();

    let mat = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;

    let (detections, mask, original_size) = detector.detect_mask(&mat.clone())?;

    let detections =
        detector.get_detections_with_classes_masks(detections, mask, 0.5, 0.5, original_size);

    // Обработка результатов
    for (class_name, rect, conf, mask) in detections {
        println!(
            "Class: {}, Confidence: {}, BoundingBox: {:?}, Mask: {:?}",
            class_name, conf, rect, mask
        );
    }

    Ok(())
//returns values
// Class: car, Confidence: 0.92939186, BoundingBox: Rect_ { x: 185, y: 900, width: 500, height: 142 }, Mask: Mat { type: "CV_8UC1", flags: "0x42FF4000 (continuous)", channels: 1, depth: "CV_8U", dims: 2, size: Size_ { width: 1680, height: 1116 }, rows: 1116, cols: 1680, elem_size: 1, elem_size1: 1, total: 1874880, is_continuous: true, is_submatrix: false, data: <element count is higher than threshold: 1000> }
// Class: person, Confidence: 0.7530618, BoundingBox: Rect_ { x: 1091, y: 860, width: 78, height: 133 }, Mask: Mat { type: "CV_8UC1", flags: "0x42FF4000 (continuous)", channels: 1, depth: "CV_8U", dims: 2, size: Size_ { width: 1680, height: 1116 }, rows: 1116, cols: 1680, elem_size: 1, elem_size1: 1, total: 1874880, is_continuous: true, is_submatrix: false, data: <element count is higher than threshold: 1000> }
}
```

## Project roadmap

- [x] Detection
- [x] Weights
- [x] OBB
- [x] Classification
- [x] Pose
- [x] Segmentation

__Note: CUDA‑GPU support was added starting from version 0.6.1__

## Author

Developed by Elieren https://github.com/Elieren .

When using the library, keep an indication of the author.