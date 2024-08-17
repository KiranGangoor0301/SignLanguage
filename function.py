#import dependency
import cv2
import numpy as np
import os
import mediapipe as mp
#  It provides utilities to draw landmarks, connections, and annotations on images or video frames.
mp_drawing = mp.solutions.drawing_utils
# Contains predefined drawing styles used for rendering landmarks and connections.
mp_drawing_styles = mp.solutions.drawing_styles
# It represents the hand tracking model provided by MediaPipe.
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converts the image from BGR color space to RGB color space.MediaPipe models typically require input images in RGB format.
    image.flags.writeable = False                  #  Sets the image as non-writeable.This might be done to prevent data alteration within the image array and ensure compatibility with the model's processing requirements.
    results = model.process(image)                 # Sends the converted image to the MediaPipe hand tracking model for prediction. Utilizes the provided model to detect and analyze hand landmarks within the image.
    image.flags.writeable = True                   # Resets the image to be writeable again. Restoring the image's writeability after processing for further operations or visualization. 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Converts the image back from RGB to BGR format.Reverts the image to BGR format, which is commonly used in OpenCV for display or further processing.
    return image, results
# Returns the processed image in BGR format along with the results obtained from the model's prediction.
# BGR- Blue-Green-Red

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


# The image on which the landmarks will be drawn.
# The results obtained from hand landmark detection, likely from a MediaPipe model.
def extract_keypoints(results):
    # Checks if there are detected hand landmarks in the results object.Ensures that hand landmarks are available before attempting to draw them on the image.
    if results.multi_hand_landmarks:
      # Iterates through each set of hand landmarks detected in the results object.
      # Allows handling multiple detected hands (if present) individually.
      for hand_landmarks in results.multi_hand_landmarks:
        # Calls a MediaPipe utility function to draw landmarks and connections on the image.
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh]))
      # The code attempts to extract the 3D coordinates (X, Y, Z) of hand landmarks detected by MediaPipe.
# If hand landmarks are detected, it flattens these coordinates into a 1D array. If no hand landmarks are detected, it creates a 1D array filled with zeros to maintain a consistent shape.
      # Renders the detected hand landmarks and their connections with predefined styles onto the provided image.
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

actions = np.array(['0','1','2','3','4','5','6','7','8','9'])

no_sequences = 30

sequence_length = 30
