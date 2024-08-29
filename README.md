## User Guide

### Running the Program

Before using the program, the user should clone the corresponding [GitHub repository](https://github.com/philgatt/Pose-Estimation-Heatmap-Regression). The pose estimation model itself is made available on [Google Drive](https://github.com/philgatt/Pose-Estimation-Heatmap-Regression) and should be put into the `model` folder. The program allows the usage of the model with the best results described in the Experimental Results section. The user can execute the program from the command line with various options:

- `--input <input_type>`: Specifies the type of input to process.
  - `image`: Process a single image.
  - `video`: Process a video file.
  - `webcam`: Process a live webcam feed.
- `--path <path_to_input>`: Specifies the path to the input image or video file. This option is required if the input type is `image` or `video`.
- `--visualization <visualization_type>`: Specifies the type of visualization to display.
  - `coordinates`: Displays only the joint coordinates.
  - `skeleton`: Displays the joint coordinates with lines connecting them to resemble the human body bone structure.
- `--threshold <threshold_value>`: A float value between 0 and 1 to set a confidence threshold for the heatmaps. If a joint's heatmap value falls below this threshold, its coordinate will not be plotted. This option allows the user to control the confidence level required for the model to display a joint's location.

### Example Commands

Below are some example commands to run the program:

1. Predict on a single image with skeleton visualization and a threshold of 0.5:

    ```bash
    python pose_estimation.py --input image --path path/to/image.jpg --visualization skeleton --threshold 0.1
    ```

2. Predict on a video with coordinates visualization and a threshold of 0.7:

    ```bash
    python pose_estimation.py --input video --path path/to/video.mp4 --visualization coordinates --threshold 0.2
    ```

3. Predict using the live webcam feed with skeleton visualization and a threshold of 0.6:

    ```bash
    python pose_estimation.py --input webcam --visualization skeleton --threshold 0.6
    ```
