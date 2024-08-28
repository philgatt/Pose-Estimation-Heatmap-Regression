from PIL import Image, ImageOps
import cv2
import numpy as np

def crop_image(image_array, bbox):
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    
    # Crop the image according to the bounding box
    cropped_image = image.crop(bbox)

    # Resize cropped image if it's larger than 224x224
    max_size = (224, 224)
    cropped_image.thumbnail(max_size)

    # Calculate padding
    width, height = cropped_image.size
    left = (224 - width) // 2
    top = (224 - height) // 2
    right = 224 - width - left
    bottom = 224 - height - top

    # Apply padding
    padded_image = ImageOps.expand(cropped_image, (left, top, right, bottom), fill=0)

    # Convert the PIL image back to a NumPy array
    padded_image_array = np.array(padded_image)

    return padded_image_array, (left, top, right, bottom)


def heatmaps_to_coordinates_scaled(heatmaps, threshold, original_size=56, target_size=224):
    scale_factor = target_size / original_size
    coordinates = []
    
    for heatmap in heatmaps:
        max_value = np.max(heatmap)
        if max_value < threshold:
            coordinates.append((None, None))
        else:
            # Find the indices of the maximum value in the heatmap
            max_index = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = max_index  

            # offset in the direction of the second highest response
            if 0 < x < heatmap.shape[1] - 1 and 0 < y < heatmap.shape[0] - 1:
                heatmap_slice = heatmap[y-1:y+2, x-1:x+2]
                heatmap_slice[1, 1] = -np.inf  
                
                # second highest peak
                second_max_index = np.unravel_index(np.argmax(heatmap_slice), heatmap_slice.shape)
                offset_x = (second_max_index[1] - 1) * 0.25  # Calculate quarter offset for x
                offset_y = (second_max_index[0] - 1) * 0.25  # Calculate quarter offset for y

                x += offset_x
                y += offset_y

            x, y = int(x * scale_factor), int(y * scale_factor)
            coordinates.append((x, y))
    
    return coordinates

def inverse_transform_coordinates(coords, bbox, padding, image_size=(224, 224)):
    """
    Transform coordinates from the cropped and padded 224x224 image back to their original positions
    in the full-size image, reversing the earlier adjustments.
    
    :param coords: List of tuples where each tuple contains (x, y) coordinates in the transformed image.
    :param bbox: List or tuple of the bounding box [x1, y1, x2, y2] used for cropping in the original image.
    :param padding: Tuple (left, top, right, bottom) of padding applied to fit the cropped image within 224x224 frame.
    :param image_size: Tuple (width, height) of the final image size, default is (224, 224).
    :return: List of tuples with original (x, y) coordinates in the full-size image.
    """
    original_coords = []
    
    left_pad, top_pad, right_pad, bottom_pad = padding
    padded_width = image_size[0] - (left_pad + right_pad)
    padded_height = image_size[1] - (top_pad + bottom_pad)
    x1, y1, x2, y2 = bbox

    cropped_width = x2 - x1
    cropped_height = y2 - y1
    scale_x = cropped_width / padded_width
    scale_y = cropped_height / padded_height

    for x, y in coords:
        # Adjust for padding
        x -= left_pad
        y -= top_pad

        # Apply inverse scaling
        x *= scale_x
        y *= scale_y

        # Adjust for cropping (reverse operation: add the top-left corner of the original bounding box)
        x += x1
        y += y1

        original_coords.append((x, y))
    
    return original_coords


def inverse_transform_coordinates(coords, bbox, padding, image_size=(224, 224)):

    """
    Transform coordinates from the cropped and padded 224x224 image back to their original positions
    in the full-size image, reversing the earlier adjustments.
    
    :param coords: List of tuples where each tuple contains (x, y) coordinates in the transformed image.
                    If a tuple is (None, None), it will be included as (None, None) in the result.
    :param bbox: List or tuple of the bounding box [x1, y1, x2, y2] used for cropping in the original image.
    :param padding: Tuple (left, top, right, bottom) of padding applied to fit the cropped image within 224x224 frame.
    :param image_size: Tuple (width, height) of the final image size, default is (224, 224).
    :return: List of tuples with original (x, y) coordinates in the full-size image.
    """
    original_coords = []
    
    left_pad, top_pad, right_pad, bottom_pad = padding
    padded_width = image_size[0] - (left_pad + right_pad)
    padded_height = image_size[1] - (top_pad + bottom_pad)
    x1, y1, x2, y2 = bbox

    cropped_width = x2 - x1
    cropped_height = y2 - y1
    scale_x = cropped_width / padded_width
    scale_y = cropped_height / padded_height

    for x, y in coords:
        if x is None and y is None:
            original_coords.append((None, None))
            continue
        
        # Adjust for padding
        x -= left_pad
        y -= top_pad

        # Apply inverse scaling
        x *= scale_x
        y *= scale_y

        # Adjust for cropping (reverse operation: add the top-left corner of the original bounding box)
        x += x1
        y += y1

        original_coords.append((x, y))
    
    return original_coords