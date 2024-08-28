import random
import numpy as np

def detect_people_batch(yolo_model, images):
    """
    Detect people in a batch of images using a YOLO model.

    Args:
        yolo_model: The YOLO model to use for detection.
        images: A list of images to process.

    Returns:
        A list of filtered boxes for each image.
    """
    all_boxes_filtered = []

    for image in images:
        results = yolo_model(image, save=False)
        boxes_filtered = []

        for result in results:
            # Filter for 'person' class detections (class_index == 0)
            person_boxes = result.boxes[result.boxes.cls == 0]
            if len(person_boxes) > 0:
                xyxy = person_boxes.xyxy.tolist()
                conf = person_boxes.conf.tolist()

                for box, confidence in zip(xyxy, conf):
                    boxes_filtered.append({
                        "class_name": "person",
                        "box_coordinates": box,
                        "confidence_score": confidence
                    })

        all_boxes_filtered.append(boxes_filtered)

    return all_boxes_filtered

def find_best_box(filtered_boxes):
    """
    Find the best bounding box based on the highest confidence score.

    Args:
        filtered_boxes: A list of dictionaries containing bounding box info.

    Returns:
        The dictionary with the best box (highest confidence) or None if empty.
    """
    if not filtered_boxes:
        return None
    
    # Select the box with the highest confidence score
    best_box = max(filtered_boxes, key=lambda x: x["confidence_score"])
    return best_box
