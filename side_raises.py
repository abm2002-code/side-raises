import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame for audio feedback
pygame.mixer.init()
shoulder_length_audio = "shoulder_length.mp3"  # Ensure you have a shoulder_length.mp3 file
short_rom_audio = "short_rom.mp3"  # Ensure you have a short_rom.mp3 file
pygame.mixer.music.load(shoulder_length_audio)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., hip)
    b = np.array(b)  # Middle point (e.g., shoulder)
    c = np.array(c)  # Third point (e.g., elbow)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

# Start video capture
cap = cv2.VideoCapture(0)

# Flags for short range of motion detection
left_rom_flag = False
right_rom_flag = False

checkedR=False
checkedL=False



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get keypoints for both arms and thighs
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        
        # Calculate angles between arm and thigh
        left_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        right_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
        
        # Draw angles on the frame
        cv2.putText(image, f'Left Angle: {int(left_angle)}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f'Right Angle: {int(right_angle)}', (400, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detect if arm goes above 90 degrees
        if (left_angle > 95 or right_angle > 95) and not pygame.mixer.music.get_busy():
            cv2.putText(image, "Lower Arms!", (200, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            pygame.mixer.music.load(shoulder_length_audio)
            pygame.mixer.music.play()
        
        # Detect short range of motion
        
        
        
        cv2.putText(image, f'Right bool: {int(right_rom_flag)}', (400, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f'left bool: {int(left_rom_flag)}', (400, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 40 <= left_angle <= 70:
            checkedL=False
        if 40 <= right_angle <= 70:
            checkedR=False
             
        
        
     
        
        if 20 <= left_angle <= 30:
            left_rom_flag = True
        if 20 <= right_angle <= 30:
            right_rom_flag = True
        
        if 70 <= left_angle <= 90:
            
            if not checkedL:
                if not left_rom_flag:
                    cv2.putText(image, "Short ROM!", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load(short_rom_audio)
                        pygame.mixer.music.play()
                else:
                    left_rom_flag = False
                checkedL=True
        
        if 70 <= right_angle <= 90:
            
            if not checkedR:
                if not right_rom_flag:
                    cv2.putText(image, "Short ROM!", (400, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load(short_rom_audio)
                        pygame.mixer.music.play()
                else:
                    right_rom_flag = False
                checkedR=True
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Show the frame
    cv2.imshow("Side Lateral Raises", image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()