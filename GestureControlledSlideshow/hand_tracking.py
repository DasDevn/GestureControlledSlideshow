import cv2
import mediapipe as mp
import keyboard
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Variables to track gestures and time
previous_x = None
last_gesture_time = 0
gesture_cooldown = 2  # Minimum time (in seconds) between gestures

# Start webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect Swipe Gestures
                current_time = time.time()
                current_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                if previous_x is not None and (current_time - last_gesture_time) > gesture_cooldown:
                    if current_x - previous_x < -0.1:
                        print("Swipe Right")
                        keyboard.press_and_release('right')
                        last_gesture_time = current_time  # Update last gesture time
                    elif current_x - previous_x > 0.1:
                        print("Swipe Left")
                        keyboard.press_and_release('left')
                        last_gesture_time = current_time  # Update last gesture time
                previous_x = current_x

                # Detect Pinch Gesture
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )
                if distance < 0.05 and (current_time - last_gesture_time) > gesture_cooldown:
                    print("Pinch Detected")
                    last_gesture_time = current_time  # Update last gesture time

                    # Exit the program when a pinch gesture is detected
                    cv2.destroyAllWindows()  # Close all OpenCV windows
                    exit()  # Exit the program

        # Display the image
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
