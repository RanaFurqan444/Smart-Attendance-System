"""
Face Recognition Manager using OpenCV and face_recognition library.
Handles face registration, encoding, and verification.
"""

import cv2
import numpy as np
import os
import pickle

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class FaceManager:
    def __init__(self, db_manager, faces_dir="assets/faces"):
        self.db = db_manager
        self.faces_dir = faces_dir
        os.makedirs(faces_dir, exist_ok=True)
        self.known_encodings = []
        self.known_ids = []
        self.tolerance = 0.5
        self._load_known_faces()

    def _load_known_faces(self):
        if not FACE_RECOGNITION_AVAILABLE:
            return
        students = self.db.get_students_with_faces()
        self.known_encodings = []
        self.known_ids = []
        for student in students:
            if student["face_encoding"]:
                encoding = pickle.loads(student["face_encoding"])
                self.known_encodings.append(encoding)
                self.known_ids.append(student["id"])

    def register_face_from_camera(self, student_db_id, student_id_str):
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "face_recognition library not installed"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot access camera"

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False, "Failed to capture image"

        return self._process_face_registration(frame, student_db_id, student_id_str)

    def register_face_from_image(self, image_path, student_db_id, student_id_str):
        if not FACE_RECOGNITION_AVAILABLE:
            return False, "face_recognition library not installed"

        if not os.path.exists(image_path):
            return False, "Image file not found"

        frame = cv2.imread(image_path)
        if frame is None:
            return False, "Failed to read image"

        return self._process_face_registration(frame, student_db_id, student_id_str)

    def _process_face_registration(self, frame, student_db_id, student_id_str):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return False, "No face detected in the image"
        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please ensure only one face is visible"

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if len(encodings) == 0:
            return False, "Could not encode face"

        encoding = encodings[0]

        # Check if face is already registered to another student
        if self.known_encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, encoding, tolerance=self.tolerance
            )
            if any(matches):
                matched_idx = matches.index(True)
                if self.known_ids[matched_idx] != student_db_id:
                    return False, "This face is already registered to another student"

        # Save face image
        face_path = os.path.join(self.faces_dir, f"{student_id_str}.jpg")
        cv2.imwrite(face_path, frame)

        # Save encoding to database
        encoding_bytes = pickle.dumps(encoding)
        self.db.update_face_encoding(student_db_id, encoding_bytes)

        # Update in-memory cache
        self.known_encodings.append(encoding)
        self.known_ids.append(student_db_id)

        return True, "Face registered successfully"

    def verify_face_from_camera(self):
        if not FACE_RECOGNITION_AVAILABLE:
            return None, "face_recognition library not installed"

        if not self.known_encodings:
            self._load_known_faces()
            if not self.known_encodings:
                return None, "No registered faces found"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Cannot access camera"

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Failed to capture image"

        return self._identify_face(frame)

    def verify_face_from_frame(self, frame):
        if not FACE_RECOGNITION_AVAILABLE:
            return None, "face_recognition library not installed"

        if not self.known_encodings:
            self._load_known_faces()
            if not self.known_encodings:
                return None, "No registered faces found"

        return self._identify_face(frame)

    def _identify_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return None, "No face detected"

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if len(encodings) == 0:
            return None, "Could not encode face"

        for encoding in encodings:
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            if len(distances) > 0:
                best_match_idx = np.argmin(distances)
                if distances[best_match_idx] < self.tolerance:
                    student_db_id = self.known_ids[best_match_idx]
                    confidence = round((1 - distances[best_match_idx]) * 100, 1)
                    return student_db_id, f"Face matched with {confidence}% confidence"

        return None, "Face not recognized"

    def get_camera_frame(self, cap):
        if cap is None or not cap.isOpened():
            return None
        ret, frame = cap.read()
        if ret:
            return frame
        return None

    def detect_faces_in_frame(self, frame):
        if not FACE_RECOGNITION_AVAILABLE:
            # Fallback to OpenCV Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            return [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_locations(rgb)

    def reload_faces(self):
        self._load_known_faces()
