use ndarray::{Array2, Array4, ArrayD, CowArray, IxDyn};
use opencv::{
    core,
    core::{Rect, Scalar, Size, Vector},
    dnn::{Backend, DetectionModel},
    imgproc,
    prelude::*,
};
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

    pub fn draw_detections(
        &self,
        mut img: Mat,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Mat> {
        for pred in detections.outer_iter() {
            let scale_x = original_size.width as f32 / (self.input_size as f32);
            let scale_y = original_size.height as f32 / (self.input_size as f32);

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

                let x1 = (x_center - width / 2.0) as i32;
                let y1 = (y_center - height / 2.0) as i32;
                let rect = core::Rect::new(x1, y1, width as i32, height as i32);

                let fallback_color = Scalar::new(255., 255., 255., 0.);
                let color = self.colors.get(max_class_id).unwrap_or(&fallback_color);
                imgproc::rectangle(&mut img, rect, *color, 2, imgproc::LINE_8, 0)?;

                let class_name = self
                    .classes
                    .get(max_class_id)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                let label = format!("{}: {:.2}", class_name, max_confidence);
                imgproc::put_text(
                    &mut img,
                    &label,
                    core::Point::new(x1, y1 - 10),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    (0.5 * scale_y.min(scale_x)).into(), // масштабируем текст
                    *color,
                    1,
                    imgproc::LINE_AA,
                    false,
                )?;
            }
        }

        Ok(img)
    }

    pub fn draw_detections_obb(
        &self,
        mut img: Mat,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        original_size: core::Size,
    ) -> opencv::Result<Mat> {
        for pred in detections.outer_iter() {
            let scale_x = original_size.width as f32 / self.input_size as f32;
            let scale_y = original_size.height as f32 / self.input_size as f32;

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
                let angle_rad = pred[19];
                let angle_deg = angle_rad.to_degrees();

                let rect_center = Point2f::new(x_center, y_center);
                let rect_size = Size2f::new(width, height);
                let rotated_rect = core::RotatedRect::new(rect_center, rect_size, angle_deg)?;

                // Получаем 4 точки прямоугольника
                let mut points: [Point2f; 4] = Default::default();
                rotated_rect.points(&mut points)?;

                // Преобразуем в i32
                let int_points: Vec<Point> = points
                    .iter()
                    .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
                    .collect();

                // Создаём Mat для polylines
                let mut contour_mat =
                    unsafe { Mat::new_rows_cols(int_points.len() as i32, 1, core::CV_32SC2)? };
                for (i, point) in int_points.iter().enumerate() {
                    *contour_mat.at_2d_mut::<Point>(i as i32, 0)? = *point;
                }

                // Рисуем контур
                imgproc::polylines(
                    &mut img,
                    &contour_mat,
                    true,
                    Scalar::new(255., 0., 0., 0.),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;

                // Цвет и подпись класса
                let fallback_color = Scalar::new(255., 255., 255., 0.);
                let color = self.colors.get(max_class_id).unwrap_or(&fallback_color);
                let class_name = self
                    .classes
                    .get(max_class_id)
                    .map(String::as_str)
                    .unwrap_or("unknown");
                let label = format!("{}: {:.2}", class_name, max_confidence);

                // Находим верхнюю левую точку из 4 углов
                let top_left = points
                    .iter()
                    .min_by(|a, b| {
                        (a.y as i32 * 10000 + a.x as i32).cmp(&(b.y as i32 * 10000 + b.x as i32))
                    })
                    .unwrap();

                let label_position =
                    Point::new(top_left.x.round() as i32, (top_left.y - 5.0) as i32);

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
        }

        Ok(img)
    }

    pub fn get_detections_with_classes(
        &self,
        detections: ndarray::Array2<f32>,
        threshold: f32,
        original_size: core::Size,
    ) -> Vec<(String, core::Rect)> {
        let mut result = Vec::new();

        for pred in detections.outer_iter() {
            let scale_x = original_size.width as f32 / (self.input_size as f32);
            let scale_y = original_size.height as f32 / (self.input_size as f32);

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

                let x1 = (x_center - width / 2.0) as i32;
                let y1 = (y_center - height / 2.0) as i32;
                let rect = core::Rect::new(x1, y1, width as i32, height as i32);

                let class_name = self
                    .classes
                    .get(max_class_id)
                    .map(String::as_str)
                    .unwrap_or("unknown");

                // Добавляем вектор с классом и его позицией (прямоугольник)
                result.push((class_name.to_string(), rect));
            }
        }

        result
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
