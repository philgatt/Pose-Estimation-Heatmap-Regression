from process_image.process_image import crop_image, heatmaps_to_coordinates_scaled, inverse_transform_coordinates
from visualize.visualize import visualize_coordinates, visualize_coordinates_frame, visualize_coordinates_video
from object_detection.object_detection import detect_people_batch, find_best_box
from PIL import Image
import numpy as np
import cv2
import time

def process_detections(model, frame, detections, threshold):
    coordinates = []
    for detection in detections:
        if detection["class_name"] == "person":
            best_box = find_best_box([detection])
            if best_box is not None:
                padded_image, padding = crop_image(frame, best_box["box_coordinates"])
                img = np.expand_dims(padded_image, axis=0)
                predictions = model.predict(img)
                predictions = np.swapaxes(predictions[0], 0, 2)
                predictions = np.swapaxes(predictions, 1, 2)
                coordinates_scaled = heatmaps_to_coordinates_scaled(predictions, threshold)
                coordinates = inverse_transform_coordinates(coordinates_scaled, best_box["box_coordinates"], padding)
    return coordinates

def process_image(model, yolo_model, image_path, threshold, visualization_type):
    detections = detect_people_batch(yolo_model, [image_path])
    image = np.array(Image.open(image_path))

    coordinates = process_detections(model, image, detections[0], threshold)
    print(np.asarray(coordinates).shape)
    visualize_coordinates(image_path, coordinates, visualization_type)

def process_video(model, yolo_model, video_path, output_path, threshold, visualization_type):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    all_detections = detect_people_batch(yolo_model, frames)
    all_coordinates = []
    for i, detection in enumerate(all_detections):
        coordinates = process_detections(model, frames[i], detection, threshold)
        all_coordinates.append(coordinates)

    print("Now creating video...")
    visualize_coordinates_video(frames, all_coordinates, visualization_type, output_path, 30)

def process_webcam(model, yolo_model, threshold, visualization_type, video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    fps_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            detections = detect_people_batch(yolo_model, [frame])
            coordinates = process_detections(model, frame, detections[0], threshold)

            if coordinates:
                visualized_frame = visualize_coordinates_frame(frame, coordinates, visualization_type)
                cv2.imshow('YOLO and Pose Detection', visualized_frame)

            fps_count += 1
            if fps_count % 10 == 0:
                end_time = time.time()
                fps = fps_count / (end_time - start_time)
                print(f"Current FPS: {fps}")
                start_time = time.time()
                fps_count = 0        

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
