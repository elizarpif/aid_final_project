import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import matplotlib

matplotlib.use("Agg")
# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

# Threshold for 
threshold = .3

# Loads video source (0 is for main webcam)
video_source = 0
# cap = cv2.VideoCapture(video_source)

# Visualization parameters
row_size = 80  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 5
font_thickness = 3
classification_results_to_show = 3
fps_avg_frame_count = 10
keypoint_detection_threshold_for_classifier = 0.1


class ProcessedImage:
    def __init__(self, img, pose_class, angle):
        self.img = img
        self.pose_class = pose_class
        self.angle = angle
    def is_pose(self, pose):
        return self.pose_class == pose

def process_image(img): 
    y, x, _ = img.shape

    tf_img = cv2.resize(img, (256, 256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img, axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf_img, dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    draw_keypoints(img, keypoints, x, y, threshold)
    draw_connections(img, keypoints, KEYPOINT_EDGE_INDS_TO_COLOR, threshold)

    # Show the class
    cl, angle = classify_pose(keypoints)
    #fps_text = 'class = ' + cl + '\n angle =' + angle
    fps_text = f'class: {cl}'
    text_location = (left_margin, row_size)
    cv2.putText(img, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    # Shows image
    return ProcessedImage(img, cl, angle)



def draw_keypoints(frame, keypoints, x, y, confidence_threshold):
    # iterate through keypoints
    for k in keypoints[0, 0, :, :]:
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)

            # Draws a circle on the image for each keypoint
            img = cv2.circle(frame, (xc, yc), 2, (0, 255, 0), 5)


def draw_connections(frame, keypoints, edges, confidence_threshold) -> None:
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def calculate_angle_three(a, b, c):
    """Calculate the angle between three points. Points are (x, y) tuples."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Clip for numerical stability
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_angle(a, b, c, d):
    """Calculate the angle between four points (x, y)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    ba = a - b
    bc = c - b
    bd = d - b
    
    cosine_angle1 = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle2 = np.dot(ba, bd) / (np.linalg.norm(ba) * np.linalg.norm(bd))
    
    # Calculate angles between the vectors
    angle1 = np.arccos(cosine_angle1)
    angle2 = np.arccos(cosine_angle2)
    
    # Calculate the angle between the vectors (considering the fourth point)
    angle = angle1 + angle2
    
    return np.degrees(angle)


# Add global variables to manage pose states
pose_states = {
    'down': False,
    'perp': False,
    'up': False
}

def classify_pose(keypoints):
    global pose_states
    
    # Assuming the same indices for right shoulder, right hip, and right wrist
    SHOULDER_INDEX = 6
    HIP_INDEX = 12
    WRIST_INDEX = 10
    ELBOW_INDEX = 8

    # Extract the specific keypoints (x, y) coordinates
    shoulder = keypoints[0, 0, SHOULDER_INDEX, :2]
    hip = keypoints[0, 0, HIP_INDEX, :2]
    wrist = keypoints[0, 0, WRIST_INDEX, :2]
    elbow = keypoints[0, 0, ELBOW_INDEX, :2]

    # Calculate angle (numpy is used for demonstration, but you should use TensorFlow operations in a real model)
    angle_degrees = calculate_angle(wrist.numpy(), shoulder.numpy(), hip.numpy(), elbow.numpy())
    wes_degrees = calculate_angle_three(wrist.numpy(), elbow.numpy(), shoulder.numpy())
    print(f'elbow angle: {wes_degrees}')
    print(f'gen angle: {angle_degrees}')

    current_movement = None

    if angle_degrees < 30:
        current_movement = 'down'
    elif 100 <= angle_degrees <= 135:
        current_movement = 'perp'
    elif 200 > angle_degrees > 180 and 155 < wes_degrees < 170:  
        current_movement = 'up'

    
    return current_movement, angle_degrees
