import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import time
import os

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename without extension as name
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces("known_faces")  # Folder containing images

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Tracking info
face_trackers = {}
face_data = {}
face_detection_interval = 5  # Detect every 5 frames for efficiency
frame_count = 0

# Start Mediapipe Face Detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_faces = {}

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face detection every few frames to optimize performance
        if frame_count % face_detection_interval == 0:
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Check for existing trackers with overlap to prevent duplicate tracking
                    face_found = False
                    for face_id, tracker in face_trackers.items():
                        success, bbox = tracker.update(frame)
                        if success:
                            tx, ty, tw, th = map(int, bbox)
                            overlap_area = (max(0, min(x + w, tx + tw) - max(x, tx)) * 
                                            max(0, min(y + h, ty + th) - max(y, ty)))
                            if overlap_area > 0:  # Face is already being tracked
                                current_faces[face_id] = (x, y, x + w, y + h)
                                face_found = True
                                break

                    # Create a new tracker if face is not found in existing trackers
                    if not face_found:
                        new_id = max(face_trackers.keys(), default=0) + 1
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        face_trackers[new_id] = tracker
                        face_data[new_id] = {'start_time': time.time(), 'last_seen': time.time()}
                        current_faces[new_id] = (x, y, x + w, y + h)

        # Update and display trackers if detection not running
        for face_id, tracker in list(face_trackers.items()):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                current_faces[face_id] = (x, y, x + w, y + h)
                face_data[face_id]['last_seen'] = time.time()

        # Recognize faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Log recognized face with time
            for face_id, (x1, y1, x2, y2) in current_faces.items():
                if x1 <= left <= x2 and y1 <= top <= y2:
                    time_in_frame = time.time() - face_data[face_id]['start_time']
                    cv2.putText(frame, f"ID:{face_id} {name} Time:{int(time_in_frame)}s", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Remove inactive trackers
        for face_id in list(face_trackers.keys()):
            if face_id not in current_faces and time.time() - face_data[face_id]['last_seen'] > 1.0:
                del face_trackers[face_id]
                del face_data[face_id]

        # Display frame
        cv2.imshow('Face Recognition Tracking', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()