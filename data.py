from function import *
from time import sleep

# It iterates over a list of actions, which likely represent different hand gestures or poses
for action in actions: 
    for sequence in range(no_sequences):
        # It loops through a specified number of sequences for each action. These sequences might represent repetitions or instances of performing a particular action.
        try: 
            # Creates directories for each action and sequence
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))

        except:
            pass

# cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(
    # Specifies the complexity level of the hand tracking model to be used. Here, it's set to 0, indicating the least complex model, which might perform faster but potentially with less accuracy or fewer features.
    model_complexity=0,
    # Sets the minimum confidence threshold required for a hand detection to be considered valid. This value, 0.5 in this case, is a confidence score between 0 and 1. Higher values indicate higher confidence required for detection.
    min_detection_confidence=0.5,
    # this parameter sets the confidence score required for the continued tracking of hand landmarks.
    min_tracking_confidence=0.5) as hands:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                frame=cv2.imread('Image/{}/{}.png'.format(action,sequence))
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                    # Calls a function (extract_keypoints) to extract keypoints or 3D coordinates of hand landmarks from the results obtained from MediaPipe.
                keypoints = extract_keypoints(results)
                #  Creates the file path where the keypoints will be saved as a numpy array file.
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # Saves the extracted keypoints as a numpy array file at the specified
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    # cap.release()
    cv2.destroyAllWindows()

