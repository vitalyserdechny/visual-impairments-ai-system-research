import cv2
import os
import numpy as np
import time

# Вычисление IOU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Площадь пересечения
    intersection = max(0, x2-x1) * max(0, y2-y1)
    # Площадь объединения
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Функция читает и возвращает из файла ограничивающие рамки с ID класса (рамки конвертируются в абсолютные координаты)
def read_boxes_truth(path, img_width, img_height):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, box_width, box_height = map(float, line.split())
            box = convert_to_absolute_coords(img_width, img_height, x_center, y_center, box_width, box_height)
            boxes.append((int(class_id), box))
    return boxes

# Функция выполняет оценку модели YOLO, которая передается в качестве параметра
def evaluate_yolo_model(yolo_model, frames_path, annotations_path, classes_scheme, step = 10, iou_threshold = 0.5):
    frame_files = os.listdir(frames_path)
    all_tp, all_fp, all_fn = 0, 0, 0

    total_time = 0
    processed_frames = 0

    for idx, frame in enumerate(frame_files):
        # Обрабатываем не все фреймы (по-умолчанию - каждый 10-й)
        if idx % step != 0:
            continue
        start_time = time.time()
        print(f'Processing frame {idx}...')
        img = cv2.imread(os.path.join(frames_path, frame))
        img_height, img_width = img.shape[:2]
        # Предсказания модели
        results = yolo_model(img)
        # Получаем результат детекции
        pred_boxes = []
        for result in results:
            if result.boxes is None:
                print(f"No objects found!")
                continue

            # До этого было просто result.boxes.xyxy
            boxes = result.boxes.xyxy.cpu().numpy()  # Используем xyxy для координат ограничивающих рамок
            class_ids = result.boxes.cls.cpu().numpy().astype(int) 
            pred_boxes.extend(zip(class_ids, boxes))

            '''
            # Получение индексов классов для каждого объекта 
            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = box  # Координаты верхнего левого и нижнего правого углов
                print(f'Object: {result.names[class_id]} (ID: {class_id} ), Box: ({x1}, {y1}), ({x2}, {y2})')
        
            '''
        # Чтение правильных рамок
        annotation_path = os.path.join(annotations_path, frame.replace('.jpg', '.txt'))
        true_boxes = read_boxes_truth(annotation_path, img_width, img_height)

        tp, fp, fn = 0, 0, len(true_boxes)
        matched_tr = set()
        for pred_class, pred_box in pred_boxes:
            matched = False
            for true_idx, (true_class, true_box) in enumerate(true_boxes):
                # Преобразование в соответствии со схемой, которая учитывает разницу в индексах классов
                true_class = classes_scheme[true_class]
                if true_idx in matched_tr:
                    continue
                iou = calculate_iou(pred_box, true_box)
                if iou > iou_threshold and pred_class == true_class:
                    tp += 1
                    fn -= 1
                    matched = True
                    matched_tr.add(true_idx)
                    break
            if not matched:
                fp += 1
        
        all_tp += tp
        all_fp += fp
        all_fn += fn

        end_time = time.time()
        frame_time = end_time - start_time  # Время обработки кадра
        total_time += frame_time  # Суммируем общее время
        processed_frames += 1
    
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}')
    if processed_frames > 0:
        average_time = total_time / processed_frames
        print(f'Average frame processing time: {average_time:.4f} s.')
    else:
        print("No processed frames found.")

# Функция рисует ограничивающие рамки на изображениях
# Изображения хранятся по адресу {images_dir}, метки - в отдельных текстовых файлах по адресу {labels_dir}
# Результат (изображения с ограничивающими рамками) сохраняется по адресу {output_dir}
def draw_bound_boxes_to_images(labels_dir, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(labels_dir, label_file)
            image_name = label_file.replace(".txt", ".jpg")  # предполагаем, что изображения имеют расширение .jpg
            image_path = os.path.join(images_dir, image_name)

            if os.path.exists(image_path):
                # Чтение изображения
                image = cv2.imread(image_path)
                height, width, _ = image.shape

                # Открытие текстового файла
                with open(label_path, "r") as file:
                    for line in file:
                        parts = line.strip().split()
                        class_id, x_center, y_center, box_width, box_height = map(float, parts)

                        # Преобразование координат
                        x_min, y_min, x_max, y_max = convert_to_absolute_coords(
                        width, height, x_center, y_center, box_width, box_height
                        )

                        # Рисуем ограничивающую рамку
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Зеленый цвет рамки

                # Сохранение изображения с рамками
                output_image_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_image_path, image)
                print(f"Рамки нанесены на {image_name}, сохранено в {output_image_path}")
            else:
                print(f"Изображение {image_name} не найдено!")

# Конвертация нормализованных координат в абсолютные
def convert_to_absolute_coords(width, height, x_center, y_center, box_width, box_height):
    x_min = int((x_center - box_width / 2) * width)
    y_min = int((y_center - box_height / 2) * height)
    x_max = int((x_center + box_width / 2) * width)
    y_max = int((y_center + box_height / 2) * height)
    return x_min, y_min, x_max, y_max

# Функция извлекает из видео, которое хранится по адресу {video_path} фреймы
# и сохраняет их в папку по адресу {output_dir_path} (папка будет создана, если не существует)
def video_to_frames(video_path, output_dir_path):
    if not os.path.exists(video_path):
        print(f"Video file path {video_path} not found!")
        return
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print("Error opening video file!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = os.path.join(output_dir_path, f'frame_{frame_count:06d}.jpg')
        cv2.imwrite(frame_file, frame)
        frame_count += 1

    print(f"Successfully saved {frame_count} frames in the folder {output_dir_path}.")
