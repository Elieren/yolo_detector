use ndarray::{s, Array2, Array4, ArrayD, ArrayView1, CowArray, IxDyn};

use opencv::{
    core,
    core::{Mat, Point, Point2f, Rect, Scalar, Size, Size2f, Vector},
    dnn::{Backend, DetectionModel},
    imgproc,
    prelude::*,
};
use ordered_float::OrderedFloat;
use ort::{Environment, Session, SessionBuilder, Value};
use rand::distr::{Distribution, Uniform};
use rand::rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

pub struct YoloDetector {
    session: Session,
    classes: Vec<String>,
    colors: Vec<Scalar>,
    input_size: i32,
}

pub struct YoloDetectorWeights {
    model: DetectionModel,
    classes: Vec<String>,
    colors: Vec<Scalar>,
}

impl YoloDetector {
    pub fn new(model_path: &str, class_file: &str, input_size: i32) -> anyhow::Result<Self> {
        if input_size % 32 != 0 {
            anyhow::bail!("Input size must be a multiple of 32");
        }
        let environment = Arc::new(Environment::builder().with_name("YOLO").build()?);
        let session = SessionBuilder::new(&environment)?.with_model_from_file(model_path)?;

        let classes = Self::load_classes(class_file)?;
        let colors = classes
            .iter()
            .map(|_| Self::generate_random_color())
            .collect();

        Ok(Self {
            session,
            classes,
            colors,
            input_size,
        })
    }

    fn load_classes(filename: &str) -> std::io::Result<Vec<String>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().filter_map(Result::ok).collect())
    }

    fn generate_random_color() -> Scalar {
        // получаем новый генератор
        let mut rng = rng();

        // создаём равномерное распределение по [0.0, 255.0]
        let die = Uniform::new_inclusive(0.0, 255.0).expect("допустимый диапазон для Uniform");

        let r: f64 = die.sample(&mut rng);
        let g: f64 = die.sample(&mut rng);
        let b: f64 = die.sample(&mut rng);

        Scalar::new(b, g, r, 0.0)
    }

    fn make_mask_mat(
        mask_weights: ArrayView1<f32>, // длина 32
        mask_protos: &Array2<f32>,     // [32, ph*pw]
        ph: usize,
        original_size: core::Size,
        threshold: f32, // порог для бинаризации
    ) -> opencv::Result<Mat> {
        // 1) линейная комбинация mask_weights·mask_protos → Vec<f32> длины ph*pw
        let mask_lin = mask_weights.dot(mask_protos);

        // 2) бинаризация по порогу, переданному как параметр → Vec<u8>
        let mask_bin: Vec<u8> = mask_lin
            .iter()
            .map(|&v| if v > threshold { 255 } else { 0 })
            .collect();

        // 3) обернуть в Mat (1‑канал)
        let base_mat = Mat::from_slice(&mask_bin)?; // базовый одноканальный
        let mask_mat = base_mat.reshape(1, ph as i32)?; // CV_8U, size ph×pw

        // 4) растянуть до original_size
        let mut mask_resized = Mat::default();
        imgproc::resize(
            &mask_mat,
            &mut mask_resized,
            original_size,
            0.0,
            0.0,
            imgproc::INTER_NEAREST,
        )?;
        Ok(mask_resized)
    }

    fn non_max_suppression(
        boxes: &[(f32, f32, f32, f32)], // x_center, y_center, width, height
        scores: &[f32],
        nms_threshold: f32,
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by_key(|&i| OrderedFloat(scores[i])); // сортируем по score
        indices.reverse();

        let mut keep = Vec::new();

        while let Some(i) = indices.pop() {
            keep.push(i);
            indices.retain(|&j| {
                let (xi, yi, wi, hi) = boxes[i];
                let (xj, yj, wj, hj) = boxes[j];

                let area_i = wi * hi;
                let area_j = wj * hj;

                let xi1 = xi - wi / 2.0;
                let yi1 = yi - hi / 2.0;
                let xi2 = xi + wi / 2.0;
                let yi2 = yi + hi / 2.0;

                let xj1 = xj - wj / 2.0;
                let yj1 = yj - hj / 2.0;
                let xj2 = xj + wj / 2.0;
                let yj2 = yj + hj / 2.0;

                let xx1 = xi1.max(xj1);
                let yy1 = yi1.max(yj1);
                let xx2 = xi2.min(xj2);
                let yy2 = yi2.min(yj2);

                let w = (xx2 - xx1).max(0.0);
                let h = (yy2 - yy1).max(0.0);
                let inter = w * h;
                let union = area_i + area_j - inter;
                let iou = inter / union;

                iou <= nms_threshold
            });
        }

        keep
    }

    pub fn detect(&self, mat: &Mat) -> Result<(Array2<f32>, core::Size), opencv::Error> {
        let original_size = mat.size()?;
        let size = core::Size::new(self.input_size, self.input_size);
        let mut resized = Mat::default();
        imgproc::resize(&mat, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Извлекаем и нормализуем данные
        let data = resized.data_bytes().unwrap();
        let input: Vec<f32> = data
            .chunks(3)
            .flat_map(|bgr| [bgr[2] as f32, bgr[1] as f32, bgr[0] as f32])
            .map(|v| v / 255.0)
            .collect();

        // Создаем Array4 [1, 3, N, N]
        let input_size = self.input_size as usize;

        let input_tensor = Array4::from_shape_fn((1, 3, input_size, input_size), |(_, c, y, x)| {
            input[(y * input_size + x) * 3 + c]
        });

        // Преобразуем в динамический тензор и оборачиваем
        let array_dyn: ArrayD<f32> = input_tensor.into_dyn();
        let cow_input = CowArray::from(array_dyn);
        let input_value = Value::from_array(self.session.allocator(), &cow_input).unwrap();

        // Инференс
        let outputs = self.session.run(vec![input_value]).unwrap();

        // Получаем OrtOwnedTensor<f32, IxDyn>
        let output_tensor: ort::tensor::OrtOwnedTensor<f32, IxDyn> =
            outputs[0].try_extract().unwrap();

        let output_view = output_tensor.view();

        // Удалим batch-ось [1, N, 8400] → [N, 8400]
        let output_array = output_view.clone().index_axis_move(ndarray::Axis(0), 0);

        // Теперь транспонируем: [N, 8400] → [8400, N]
        let transposed_dyn = output_array.t().to_owned();
        let transposed: Array2<f32> = transposed_dyn
            .into_dimensionality()
            .map_err(|e| opencv::Error::new(0, format!("Shape error: {}", e)))?;

        Ok((transposed, original_size))
    }

    pub fn detect_mask(
        &self,
        mat: &Mat,
    ) -> Result<(Array2<f32>, Array2<f32>, core::Size), opencv::Error> {
        let original_size = mat.size()?;
        let size = core::Size::new(self.input_size, self.input_size);
        let mut resized = Mat::default();
        imgproc::resize(&mat, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Извлекаем и нормализуем данные
        let data = resized.data_bytes().unwrap();
        let input: Vec<f32> = data
            .chunks(3)
            .flat_map(|bgr| [bgr[2] as f32, bgr[1] as f32, bgr[0] as f32])
            .map(|v| v / 255.0)
            .collect();

        // Создаем Array4 [1, 3, N, N]
        let input_size = self.input_size as usize;

        let input_tensor = Array4::from_shape_fn((1, 3, input_size, input_size), |(_, c, y, x)| {
            input[(y * input_size + x) * 3 + c]
        });

        // Преобразуем в динамический тензор и оборачиваем
        let array_dyn: ArrayD<f32> = input_tensor.into_dyn();
        let cow_input = CowArray::from(array_dyn);
        let input_value = Value::from_array(self.session.allocator(), &cow_input).unwrap();

        // Инференс
        let outputs = self.session.run(vec![input_value]).unwrap();

        // Получаем OrtOwnedTensor<f32, IxDyn>
        let output_tensor1: ort::tensor::OrtOwnedTensor<f32, IxDyn> =
            outputs[0].try_extract().unwrap();

        let output_tensor2: ort::tensor::OrtOwnedTensor<f32, IxDyn> =
            outputs[1].try_extract().unwrap();

        let temp_output0 = output_tensor1.view(); // [116,8400]
        let output0 = temp_output0.clone().index_axis_move(ndarray::Axis(0), 0);
        let dets = output0.t().to_owned(); // [8400,116]
        let temp_protos = output_tensor2.view(); // [32,160,160]
        let protos = temp_protos
            .clone()
            .index_axis(ndarray::Axis(0), 0)
            .to_owned();
        let (num_proto, ph, pw) = (protos.shape()[0], protos.shape()[1], protos.shape()[2]);
        let proto_flat = protos.into_shape((num_proto, ph * pw)).unwrap();

        let detections: Array2<f32> = dets
            .into_dimensionality()
            .map_err(|e| opencv::Error::new(0, format!("Shape error: {}", e)))?;

        let mask: Array2<f32> = proto_flat
            .into_dimensionality()
            .map_err(|e| opencv::Error::new(0, format!("Shape error: {}", e)))?;

        Ok((detections, mask, original_size))
    }

    pub fn draw_detections(
        &self,
        mut img: Mat,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Mat> {
        let scale_x = original_size.width as f32 / (self.input_size as f32);
        let scale_y = original_size.height as f32 / (self.input_size as f32);

        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();

        for pred in detections.outer_iter() {
            let class_confidences: Vec<f32> = pred.iter().copied().skip(4).collect();
            let (max_class_id, max_confidence) = class_confidences
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &conf)| (i as usize, conf))
                .unwrap_or((0, 0.0));

            if max_confidence > threshold {
                let x_center = pred[0] * scale_x;
                let y_center = pred[1] * scale_y;
                let width = pred[2] * scale_x;
                let height = pred[3] * scale_y;

                boxes.push((x_center, y_center, width, height));
                scores.push(max_confidence);
                class_ids.push(max_class_id);
            }
        }

        let keep_indices = Self::non_max_suppression(&boxes, &scores, nms_threshold); // nms_threshold

        for &i in &keep_indices {
            let (x_center, y_center, width, height) = boxes[i];
            let class_id = class_ids[i];
            let confidence = scores[i];

            let x1 = (x_center - width / 2.0) as i32;
            let y1 = (y_center - height / 2.0) as i32;
            let rect = core::Rect::new(x1, y1, width as i32, height as i32);

            let fallback_color = Scalar::new(255., 255., 255., 0.);
            let color = self.colors.get(class_id).unwrap_or(&fallback_color);
            imgproc::rectangle(&mut img, rect, *color, 2, imgproc::LINE_8, 0)?;

            let class_name = self
                .classes
                .get(class_id)
                .map(String::as_str)
                .unwrap_or("unknown");
            let label = format!("{}: {:.2}", class_name, confidence);
            imgproc::put_text(
                &mut img,
                &label,
                core::Point::new(x1, y1 - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                (0.5 * scale_y.min(scale_x)).into(),
                *color,
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

        Ok(img)
    }

    pub fn draw_detections_obb(
        &self,
        mut img: Mat,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Mat> {
        let scale_x = original_size.width as f32 / self.input_size as f32;
        let scale_y = original_size.height as f32 / self.input_size as f32;

        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();
        let mut angles = Vec::new();

        for pred in detections.outer_iter() {
            let total_len = pred.len();
            let class_confidences: Vec<f32> =
                pred.iter().copied().skip(4).take(total_len - 5).collect();

            let (max_class_id, max_confidence) = class_confidences
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &conf)| (i as usize, conf))
                .unwrap_or((0, 0.0));

            if max_confidence > threshold {
                let x_center = pred[0] * scale_x;
                let y_center = pred[1] * scale_y;
                let width = pred[2] * scale_x;
                let height = pred[3] * scale_y;
                let angle_rad = pred[total_len - 1];

                boxes.push((x_center, y_center, width, height));
                scores.push(max_confidence);
                class_ids.push(max_class_id);
                angles.push(angle_rad);
            }
        }

        let keep_indices = Self::non_max_suppression(&boxes, &scores, nms_threshold); // nms_threshold

        for &i in &keep_indices {
            let (x_center, y_center, width, height) = boxes[i];
            let class_id = class_ids[i];
            let confidence = scores[i];
            let angle_rad = angles[i];
            let angle_deg = angle_rad.to_degrees();

            let rect_center = Point2f::new(x_center, y_center);
            let rect_size = Size2f::new(width, height);
            let rotated_rect = core::RotatedRect::new(rect_center, rect_size, angle_deg)?;

            let mut points: [Point2f; 4] = Default::default();
            rotated_rect.points(&mut points)?;

            let int_points: Vec<Point> = points
                .iter()
                .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
                .collect();

            let mut contour_mat =
                unsafe { Mat::new_rows_cols(int_points.len() as i32, 1, core::CV_32SC2)? };
            for (j, point) in int_points.iter().enumerate() {
                *contour_mat.at_2d_mut::<Point>(j as i32, 0)? = *point;
            }

            imgproc::polylines(
                &mut img,
                &contour_mat,
                true,
                Scalar::new(255., 0., 0., 0.),
                2,
                imgproc::LINE_8,
                0,
            )?;

            let fallback_color = Scalar::new(255., 255., 255., 0.);
            let color = self.colors.get(class_id).unwrap_or(&fallback_color);
            let class_name = self
                .classes
                .get(class_id)
                .map(String::as_str)
                .unwrap_or("unknown");
            let label = format!("{}: {:.2}", class_name, confidence);

            let top_left = points
                .iter()
                .min_by(|a, b| {
                    (a.y as i32 * 10000 + a.x as i32).cmp(&(b.y as i32 * 10000 + b.x as i32))
                })
                .unwrap();

            let label_position = Point::new(top_left.x.round() as i32, (top_left.y - 5.0) as i32);
            imgproc::put_text(
                &mut img,
                &label,
                label_position,
                imgproc::FONT_HERSHEY_SIMPLEX,
                (0.5 * scale_y.min(scale_x)).into(),
                *color,
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

        Ok(img)
    }

    pub fn draw_detections_masked(
        &self,
        img: Mat,
        detections: Array2<f32>,
        mask_protos: Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Mat> {
        let mut output = img.clone();
        let mut overlay = output.clone();

        let ph = (mask_protos.shape()[1] as f32).sqrt() as usize;

        let sx = original_size.width as f32 / self.input_size as f32;
        let sy = original_size.height as f32 / self.input_size as f32;

        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();
        let mut mask_weights_list = Vec::new();

        for pred in detections.outer_iter() {
            let class_scores = pred.slice(s![4..84]);
            let (class_id, &conf) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if conf < threshold {
                continue;
            }

            let bx = pred[0] * sx;
            let by = pred[1] * sy;
            let bw = pred[2] * sx;
            let bh = pred[3] * sy;
            let x1 = bx - bw / 2.0;
            let y1 = by - bh / 2.0;

            boxes.push((x1, y1, bw, bh));
            scores.push(conf);
            class_ids.push(class_id);
            mask_weights_list.push(pred.slice(s![84..116]).to_owned());
        }

        let keep_indices = Self::non_max_suppression(&boxes, &scores, nms_threshold);

        for &i in &keep_indices {
            let (x1, y1, bw, bh) = boxes[i];
            let class_id = class_ids[i];
            let mask_weights = &mask_weights_list[i];

            let bx = x1 + bw / 2.0;
            let by = y1 + bh / 2.0;
            let x1 = (bx - bw / 2.0).round() as i32;
            let y1 = (by - bh / 2.0).round() as i32;
            let x2 = (bx + bw / 2.0).round() as i32;
            let y2 = (by + bh / 2.0).round() as i32;

            let rect = core::Rect::new(
                x1.clamp(0, original_size.width - 1),
                y1.clamp(0, original_size.height - 1),
                (x2 - x1).clamp(1, original_size.width - x1),
                (y2 - y1).clamp(1, original_size.height - y1),
            );

            let fallback_color = Scalar::new(255., 255., 255., 0.);
            let color = *self.colors.get(class_id).unwrap_or(&fallback_color);

            imgproc::rectangle(&mut output, rect, color, 2, imgproc::LINE_8, 0)?;

            let mask_mat = Self::make_mask_mat(
                mask_weights.view(),
                &mask_protos,
                ph,
                original_size,
                threshold,
            )?;

            let mut bbox_mask =
                Mat::zeros(original_size.height, original_size.width, core::CV_8UC1)?.to_mat()?;
            imgproc::rectangle(
                &mut bbox_mask,
                rect,
                Scalar::new(255., 255., 255., 0.),
                -1,
                imgproc::LINE_8,
                0,
            )?;

            let mut mask_clipped = Mat::default();
            core::bitwise_and(&mask_mat, &bbox_mask, &mut mask_clipped, &Mat::default())?;

            overlay.set_to(&color, &mask_clipped)?;
        }

        let mut blended = Mat::default();
        core::add_weighted(&overlay, 0.5, &output, 0.5, 0.0, &mut blended, -1)?;

        Ok(blended)
    }

    pub fn get_detections_with_classes(
        &self,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> Vec<(String, core::Rect)> {
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();

        let scale_x = original_size.width as f32 / (self.input_size as f32);
        let scale_y = original_size.height as f32 / (self.input_size as f32);

        for pred in detections.outer_iter() {
            let class_confidences: Vec<f32> = pred.iter().copied().skip(4).collect();
            let (max_class_id, max_confidence) = class_confidences
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &conf)| (i as usize, conf))
                .unwrap_or((0, 0.0));

            if max_confidence > threshold {
                let x_center = pred[0] * scale_x;
                let y_center = pred[1] * scale_y;
                let width = pred[2] * scale_x;
                let height = pred[3] * scale_y;

                let x1 = x_center - width / 2.0;
                let y1 = y_center - height / 2.0;

                boxes.push((x1, y1, width, height));
                scores.push(max_confidence);
                class_ids.push(max_class_id);
            }
        }

        let keep_indices = Self::non_max_suppression(&boxes, &scores, nms_threshold);

        let mut result = Vec::new();
        for &i in &keep_indices {
            let (x1, y1, w, h) = boxes[i];
            let rect = core::Rect::new(
                x1.round() as i32,
                y1.round() as i32,
                w.round() as i32,
                h.round() as i32,
            );

            let class_name = self
                .classes
                .get(class_ids[i])
                .map(String::as_str)
                .unwrap_or("unknown");

            result.push((class_name.to_string(), rect));
        }

        result
    }

    pub fn get_detections_with_classes_obb(
        &self,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> Vec<(String, core::Rect, f32)> {
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();
        let mut angles = Vec::new();

        let scale_x = original_size.width as f32 / self.input_size as f32;
        let scale_y = original_size.height as f32 / self.input_size as f32;

        for pred in detections.outer_iter() {
            let class_confidences: Vec<f32> = pred.iter().copied().skip(4).take(15).collect();
            let (max_class_id, max_confidence) = class_confidences
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &conf)| (i as usize, conf))
                .unwrap_or((0, 0.0));

            if max_confidence > threshold {
                let x_center = pred[0] * scale_x;
                let y_center = pred[1] * scale_y;
                let width = pred[2] * scale_x;
                let height = pred[3] * scale_y;
                let angle_deg = pred[19].to_degrees();

                let x1 = x_center - width / 2.0;
                let y1 = y_center - height / 2.0;

                boxes.push((x1, y1, width, height));
                scores.push(max_confidence);
                class_ids.push(max_class_id);
                angles.push(angle_deg);
            }
        }

        let keep_indices = Self::non_max_suppression(&boxes, &scores, nms_threshold);

        let mut result = Vec::new();
        for &i in &keep_indices {
            let (x1, y1, w, h) = boxes[i];
            let rect = core::Rect::new(
                x1.round() as i32,
                y1.round() as i32,
                w.round() as i32,
                h.round() as i32,
            );

            let class_name = self
                .classes
                .get(class_ids[i])
                .map(String::as_str)
                .unwrap_or("unknown");

            result.push((class_name.to_string(), rect, angles[i]));
        }

        result
    }

    pub fn get_detections_with_classes_masks(
        &self,
        detections: Array2<f32>,
        mask_protos: Array2<f32>,
        threshold: f32,
        nms_threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Vec<(String, core::Rect, f32, Mat)>> {
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        let mut class_ids = Vec::new();
        let mut masks = Vec::new();

        let ph = (mask_protos.shape()[1] as f32).sqrt() as usize;
        let scale_x = original_size.width as f32 / self.input_size as f32;
        let scale_y = original_size.height as f32 / self.input_size as f32;

        for pred in detections.outer_iter() {
            let class_scores = pred.slice(s![4..84]);
            let (class_id, &conf) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if conf < threshold {
                continue;
            }

            let bx = pred[0] * scale_x;
            let by = pred[1] * scale_y;
            let bw = pred[2] * scale_x;
            let bh = pred[3] * scale_y;
            let x1 = bx - bw / 2.0;
            let y1 = by - bh / 2.0;

            // bbox
            boxes.push((x1, y1, bw, bh));
            scores.push(conf);
            class_ids.push(class_id);

            // маска
            let mask_weights = pred.slice(s![84..116]);
            let mask =
                Self::make_mask_mat(mask_weights, &mask_protos, ph, original_size, threshold)?;
            masks.push(mask);
        }

        // NMS
        let keep = Self::non_max_suppression(&boxes, &scores, nms_threshold);

        // финальные результаты
        let mut results = Vec::new();
        for &i in &keep {
            let (x, y, w, h) = boxes[i];
            let rect = core::Rect::new(
                x.round() as i32,
                y.round() as i32,
                w.round() as i32,
                h.round() as i32,
            );

            let class_name = self
                .classes
                .get(class_ids[i])
                .map(String::as_str)
                .unwrap_or("unknown");

            results.push((class_name.to_string(), rect, scores[i], masks[i].clone()));
        }

        Ok(results)
    }

    pub fn classify(&self, mat: &Mat, threshold: f32) -> Result<Vec<(String, f32)>, opencv::Error> {
        let size = core::Size::new(self.input_size, self.input_size);
        let mut resized = Mat::default();
        imgproc::resize(&mat, &mut resized, size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

        // Извлекаем и нормализуем данные
        let data = resized.data_bytes().unwrap();
        let input: Vec<f32> = data
            .chunks(3)
            .flat_map(|bgr| [bgr[2] as f32, bgr[1] as f32, bgr[0] as f32])
            .map(|v| v / 255.0)
            .collect();

        let input_size = self.input_size as usize;

        let input_tensor = Array4::from_shape_fn((1, 3, input_size, input_size), |(_, c, y, x)| {
            input[(y * input_size + x) * 3 + c]
        });

        let array_dyn: ArrayD<f32> = input_tensor.into_dyn();
        let cow_input = CowArray::from(array_dyn);
        let input_value = Value::from_array(self.session.allocator(), &cow_input).unwrap();

        let outputs = self.session.run(vec![input_value]).unwrap();

        let output_tensor: ort::tensor::OrtOwnedTensor<f32, IxDyn> =
            outputs[0].try_extract().unwrap();

        let output_view = output_tensor.view();
        let output_array = output_view.clone().index_axis_move(ndarray::Axis(0), 0);

        // Собираем все классы с score > threshold
        let mut results = Vec::new();
        for (i, &score) in output_array.iter().enumerate() {
            if score > threshold {
                let class_name = &self.classes[i];
                results.push((class_name.clone(), score));
            }
        }

        Ok(results)
    }
}

impl YoloDetectorWeights {
    pub fn new(weights: &str, cfg: &str, class_file: &str) -> anyhow::Result<Self> {
        let mut model = DetectionModel::new(weights, cfg)?;
        model.set_preferable_backend(Backend::DNN_BACKEND_OPENCV)?;
        model.set_input_params(
            1.0 / 255.0,
            Size::new(640, 640),
            Scalar::default(),
            true,
            false,
        )?;

        let classes = Self::load_classes(class_file)?;
        let colors = classes
            .iter()
            .map(|_| Self::generate_random_color())
            .collect();

        Ok(Self {
            model,
            classes,
            colors,
        })
    }

    fn load_classes(filename: &str) -> std::io::Result<Vec<String>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().filter_map(Result::ok).collect())
    }

    fn generate_random_color() -> Scalar {
        // получаем новый генератор
        let mut rng = rng();

        // создаём равномерное распределение по [0.0, 255.0]
        let die = Uniform::new_inclusive(0.0, 255.0).expect("допустимый диапазон для Uniform");

        let r: f64 = die.sample(&mut rng);
        let g: f64 = die.sample(&mut rng);
        let b: f64 = die.sample(&mut rng);

        Scalar::new(b, g, r, 0.0)
    }

    pub fn detect(
        &mut self,
        mat: &Mat,
        threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vector<i32>, Vector<f32>, Vector<Rect>), opencv::Error> {
        let mut class_ids = Vector::<i32>::new(); // идентификаторы классов
        let mut confidences = Vector::<f32>::new(); // confidence каждого бокса
        let mut boxes = Vector::<Rect>::new(); // прямоугольники

        self.model.detect(
            mat,
            &mut class_ids,
            &mut confidences,
            &mut boxes,
            threshold,
            nms_threshold,
        )?;

        Ok((class_ids, confidences, boxes))
    }

    pub fn draw_detections(
        &self,
        img: &mut Mat,
        class_ids: Vector<i32>,
        confidences: Vector<f32>,
        boxes: Vector<Rect>,
    ) -> opencv::Result<Mat> {
        // Рисуем результаты
        for i in 0..class_ids.len() {
            let rect = boxes.get(i)?;
            let class_id = class_ids.get(i)?;
            let unknown = "Unknown".to_string();
            let class_name = self.classes.get(class_id as usize).unwrap_or(&unknown);
            let conf = confidences.get(i)?;

            // Цвет по умолчанию
            let default_color = Scalar::new(255., 255., 255., 0.);

            // Выбираем случайный цвет для рамки
            let color = self.colors.get(class_id as usize).unwrap_or(&default_color); // Используем переменную

            imgproc::rectangle(img, rect, *color, 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(
                img,
                &format!("{}: {:.2}", class_name, conf), // выводим класс и confidence
                rect.tl(),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(255., 255., 255., 0.),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }
        Ok(img.clone())
    }
}
