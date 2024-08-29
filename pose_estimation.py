from process_input.process_input import process_image, process_video, process_webcam
from utils.loss import arobust_loss
import argparse
import os
from ultralytics import YOLO
from tensorflow.keras import backend as K
import tensorflow as tf

print("Loading pose estimation model...")
pose_model = tf.keras.models.load_model(r'model/pose-estimation-model.h5', custom_objects={'loss': arobust_loss(alpha=1.0, c=1.0, gamma=2.1)})
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

def main(args):
    while True:
        # Validate the input path
        if args.input in ['image', 'video']:
            while not os.path.exists(args.path):
                print(f"The path '{args.path}' does not exist. Please enter a valid path.")
                args.path = input("Enter the correct path to the image or video: ").strip()
        
        # Validate the visualization option
        while args.visualization not in ['skeleton', 'coordinates']:
            print(f"Invalid visualization option '{args.visualization}'. Please enter 'skeleton' or 'coordinates'.")
            args.visualization = input("Enter visualization type (coordinates, skeleton): ").strip().lower()

        # Validate the threshold value
        while not (0 <= args.threshold <= 1):
            print(f"Invalid threshold '{args.threshold}'. The threshold must be a number between 0 and 1.")
            try:
                args.threshold = float(input("Enter the threshold for heatmaps (0-1): ").strip())
            except ValueError:
                print("Please enter a valid number between 0 and 1.")
        
        # Process the input based on the type
        if args.input == 'image':
            process_image(pose_model, yolo_model, args.path, args.threshold, args.visualization)
        elif args.input == 'video':
            process_video(pose_model, yolo_model, args.path, args.output_path, args.threshold, args.visualization)
        elif args.input == 'webcam':
            process_webcam(pose_model, yolo_model, args.threshold, args.visualization)

        # Ask the user if they want to visualize another image or video
        another = input("Would you like to process another image or video? (yes/no): ").strip().lower()
        
        if another != 'yes':
            print("Exiting the program.")
            break
        
        # If the user wants to process another image, prompt for the new parameters
        args.input = input("Enter input type (image, video, webcam): ").strip().lower()
        if args.input == 'image' or args.input == 'video':
            args.path = input("Enter the path to the image or video: ").strip()
            while not os.path.exists(args.path):
                print(f"The path '{args.path}' does not exist. Please enter a valid path.")
                args.path = input("Enter the correct path to the image or video: ").strip()
            if args.input =='video':
                input("Enter the output path for the video: ").strip()
                while not os.path.exists(args.output_path):
                    print(f"The path '{args.output_path}' does not exist. Please enter a valid path.")
                    args.path = input("Enter a valid path where the video should be saved to: ").strip()

        
        args.visualization = input("Enter visualization type (coordinates, skeleton): ").strip().lower()
        while args.visualization not in ['skeleton', 'coordinates']:
            print(f"Invalid visualization option '{args.visualization}'. Please enter 'skeleton' or 'coordinates'.")
            args.visualization = input("Enter visualization type (coordinates, skeleton): ").strip().lower()
        
        try:
            args.threshold = float(input("Enter the threshold for heatmaps (0-1): ").strip())
            while not (0 <= args.threshold <= 1):
                print(f"Invalid threshold '{args.threshold}'. The threshold must be a number between 0 and 1.")
                args.threshold = float(input("Enter the threshold for heatmaps (0-1): ").strip())
        except ValueError:
            print("Please enter a valid number between 0 and 1.")
            args.threshold = float(input("Enter the threshold for heatmaps (0-1): ").strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Estimation CLI Tool')
    parser.add_argument('--input', required=True, choices=['image', 'video', 'webcam'], help='Input type: image, video, or webcam')
    parser.add_argument('--path', type=str, help='Path to the input image or video file')
    parser.add_argument('--output_path', type=str, help='Path where the processed video will be saved to')    
    parser.add_argument('--visualization', required=True, choices=['coordinates', 'skeleton'], help='Type of visualization: coordinates or skeleton')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for heatmaps')
    
    args = parser.parse_args()
    main(args)