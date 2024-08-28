from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tqdm
from io import BytesIO

# Label indices based on the given order
labels = [
    'r ankle', 'r knee', 'r hip', 'l hip', 'l knee', 'l ankle', 
    'pelvis', 'thorax', 'upper neck', 'head top', 'r wrist', 
    'r elbow', 'r shoulder', 'l shoulder', 'l elbow', 'l wrist'
]
label_indices = {label: i for i, label in enumerate(labels)}

# Assign specific colors to each label
label_colors = {
    'r ankle': '#FF0000', 'r knee': '#FF7F00', 'r hip': '#FFFF00',
    'l hip': '#00FF00', 'l knee': '#0000FF', 'l ankle': '#8B00FF',
    'pelvis': '#FF1493', 'thorax': '#00FFFF', 'upper neck': '#FFD700',
    'head top': '#FF00FF', 'r wrist': '#00FF7F', 'r elbow': '#1E90FF',
    'r shoulder': '#ADFF2F', 'l shoulder': '#FF4500', 'l elbow': '#FF69B4',
    'l wrist': '#00CED1'
}

# Connections between keypoints
connections = [
    ('head top', 'upper neck'), ('upper neck', 'thorax'), ('thorax', 'pelvis'),
    ('r shoulder', 'r elbow'), ('r elbow', 'r wrist'), ('l shoulder', 'l elbow'),
    ('l elbow', 'l wrist'), ('r hip', 'r knee'), ('r knee', 'r ankle'),
    ('l hip', 'l knee'), ('l knee', 'l ankle'), ('l shoulder', 'r shoulder'),
    ('l hip', 'r hip')
]

def hex_to_bgr(hex_color):
    """Convert a hex color string to BGR tuple for OpenCV."""
    hex_color = hex_color.strip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

label_colors_bgr = {key: hex_to_bgr(value) for key, value in label_colors.items()}

def plot_coordinates(ax, image, coordinates, visualization_type):
    """Plot keypoints and skeleton on the provided axes."""
    ax.imshow(image)

    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    if visualization_type == 'skeleton':
        for label1, label2 in connections:
            idx1, idx2 = label_indices[label1], label_indices[label2]
            if (x_coords[idx1] != 0 or y_coords[idx1] != 0) and (x_coords[idx2] != 0 or y_coords[idx2] != 0):
                ax.plot([x_coords[idx1], x_coords[idx2]], [y_coords[idx1], y_coords[idx2]], 'r-', linewidth=3, zorder=1)

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if x != 0 or y != 0:
            label = labels[i]
            ax.scatter(x, y, c=label_colors[label], edgecolors='black', marker='o', s=50, linewidths=1, zorder=2)

    ax.axis('off')

def visualize_coordinates(image, coordinates, visualization_type):
    """Visualize coordinates on a single image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    else:
        raise ValueError("Invalid input: image must be a file path or a numpy array.")

    fig, ax = plt.subplots()
    plot_coordinates(ax, image, coordinates, visualization_type)
    plt.show()

def visualize_coordinates_video(frames, all_coordinates, visualization_type, output_path, fps):
    """Visualize coordinates on a video."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fig, ax = plt.subplots()

    for frame, coordinates in tqdm.tqdm(zip(frames, all_coordinates), total=len(frames), desc="Processing frames"):
        ax.clear()
        image = Image.fromarray(frame)
        plot_coordinates(ax, image, coordinates, visualization_type)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        visualized_frame = np.array(Image.open(buf))
        buf.close()

        visualized_frame = cv2.cvtColor(visualized_frame, cv2.COLOR_RGB2BGR)
        out.write(visualized_frame)

    plt.close(fig)
    out.release()
    print(f"Output video saved to {output_path}")

def visualize_coordinates_frame(frame, coordinates, visualization_type):
    """Visualize coordinates on a single video frame."""
    image = frame.copy()

    filtered_coords = [(idx, coord) for idx, coord in enumerate(coordinates) if coord[0] is not None and coord[1] is not None]

    x_coords = [coord[1][0] for coord in filtered_coords]
    y_coords = [coord[1][1] for coord in filtered_coords]
    valid_indices = {coord[0]: idx for idx, coord in enumerate(filtered_coords)}

    if visualization_type == 'skeleton':
        for label1, label2 in connections:
            idx1, idx2 = label_indices[label1], label_indices[label2]
            if idx1 in valid_indices and idx2 in valid_indices:
                point1 = (int(x_coords[valid_indices[idx1]]), int(y_coords[valid_indices[idx1]]))
                point2 = (int(x_coords[valid_indices[idx2]]), int(y_coords[valid_indices[idx2]]))
                cv2.line(image, point1, point2, (0, 0, 255), 7)

    for idx, (x, y) in zip(valid_indices.keys(), zip(x_coords, y_coords)):
        label = labels[idx]
        color = label_colors_bgr[label]
        cv2.circle(image, (int(x), int(y)), 15, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), 15, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return image
