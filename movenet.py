# Import tensorflow, opencv, and numpy
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
#import imageio
#from IPython.display import HTML, display

# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

# Threshold for 
threshold = .3

# Loads video source (0 is for main webcam)
video_source = 0
cap = cv2.VideoCapture(video_source)

# Visualization parameters
row_size = 80  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 2
font_thickness = 3
classification_results_to_show = 3
fps_avg_frame_count = 10
keypoint_detection_threshold_for_classifier = 0.1

# Checks errors while opening the Video Capture
if not cap.isOpened():
    print('Error loading video')
    quit()

success, img = cap.read()

if not success:
    print('Error reding frame')
    quit()

y, x, _ = img.shape

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


def calculate_angle(a, b, c):
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

import numpy as np

def calculate_wrist_elbow_shoulder_angle(wrist, elbow, shoulder):
    """Calculate the angle between wrist, elbow, and shoulder points."""
    wrist = np.array(wrist)
    elbow = np.array(elbow)
    shoulder = np.array(shoulder)
    
    # Define vectors from wrist to elbow and wrist to shoulder
    wrist_to_elbow = elbow - wrist
    wrist_to_shoulder = shoulder - wrist
    
    # Calculate angles between the vectors
    cosine_angle = np.dot(wrist_to_elbow, wrist_to_shoulder) / (np.linalg.norm(wrist_to_elbow) * np.linalg.norm(wrist_to_shoulder))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)



# Add global variables to manage pose states
pose_states = {
    'down': False,
    'perp': False,
    'up': False
}

# Function to classify pose
def classify_pose(keypoints):
    global pose_states
    
    # Iindices for right shoulder, right hip, right wrist, and right elbow
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
    wes_degrees = calculate_wrist_elbow_shoulder_angle(wrist.numpy(), elbow.numpy(), shoulder.numpy())
    print(f'Angle: {angle_degrees}')

    current_movement = None

    # Determine pose movement based on angle threshold values
    if angle_degrees < 20:
        current_movement = 'down'
    elif 90 <= angle_degrees <= 105:
        current_movement = 'perp'
    elif 200 > angle_degrees > 180 and 0 < wes_degrees < 8:   
        current_movement = 'up'

    # Print current movement and pose states for debugging
    print("Current Movement:", current_movement)
    print("Pose States:", pose_states)
    print("WES Degrees:", wes_degrees)
   
    # Manage pose states based on sequence logic
    if not pose_states['down']:
        if current_movement == 'down':
            pose_states['down'] = True
    elif pose_states['down'] and not pose_states['perp']:
        if current_movement == 'perp':
            pose_states['perp'] = True
    elif pose_states['down'] and pose_states['perp'] and not pose_states['up']:
        if current_movement == 'up':
            pose_states['up'] = True

    # Check if all poses have been detected in order
    if pose_states['down'] and pose_states['perp'] and pose_states['up']:
        # Reset pose states for the next sequence
        pose_states = {
            'down': False,
            'perp': False,
            'up': False
        }
        return 'Correct Sequence Detected', angle_degrees
    else:
        return f'{pose_states}', angle_degrees


# Visualization parameters
row_size = 450  # pixels - adjusted for larger text space
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1  # reduced font size for better visibility
font_thickness = 2  # slightly reduced font thickness

# Create a larger window for better visibility
window_name = 'Movenet'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # increased window size

correct_sequence_detected = False
show_instructions = False

while success:
    tf_img = cv2.resize(img, (256, 256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img, axis=0)

    image = tf.cast(tf_img, dtype=tf.int32)

    outputs = movenet(image)
    keypoints = outputs['output_0']

    draw_keypoints(img, keypoints, x, y, threshold)
    draw_connections(img, keypoints, KEYPOINT_EDGE_INDS_TO_COLOR, threshold)

    cl, angle = classify_pose(keypoints)

    pose_text = f'Class: {cl}, Angle: {angle:.2f} degrees'  # Pose with angle to display

    # Display the pose and angle text at the bottom of the frame
    cv2.putText(img, pose_text, (left_margin, row_size), cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Check if correct sequence detected before displaying instructions
    if cl.startswith('Correct'):
        show_instructions = True

    # Display instructions if the flag is set
    if show_instructions:
        instruction_text = ["              Instructions:",
                            "      Correct Pose Sequence Detected.",
                            "    Press 'R' to restart or 'Q' to Quit"]
        text_y = 50
        for line in instruction_text:
            cv2.putText(img, line, (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            text_y += 30  # Adjust vertical spacing between lines

    cv2.imshow(window_name, img)
    key = cv2.waitKey(1)

    if key == ord("r"):
        pose_states = {
            'down': False,
            'perp': False,
            'up': False
        }
        show_instructions = False  # Reset the flag when 'R' is pressed
    elif key == ord("q"):
        break

    success, img = cap.read()

cap.release()