"""
Dynamic QR Code Manager.
Generates QR codes that change every 30 seconds using TOTP-like mechanism.
"""

import qrcode
import hashlib
import hmac
import time
import json
import os
import secrets
from io import BytesIO
from PIL import Image

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


class QRManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.refresh_interval = 30  # seconds

    def generate_class_secret(self):
        return secrets.token_hex(32)

    def _get_time_step(self):
        return int(time.time()) // self.refresh_interval

    def _generate_token(self, class_id, secret):
        time_step = self._get_time_step()
        message = f"{class_id}:{time_step}".encode()
        token = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()[:16]
        return token, time_step

    def generate_qr_data(self, class_id, secret):
        token, time_step = self._generate_token(class_id, secret)
        data = {
            "class_id": class_id,
            "token": token,
            "ts": time_step,
            "exp": (time_step + 1) * self.refresh_interval
        }
        return json.dumps(data)

    def generate_qr_image(self, class_id, secret, size=300):
        data = self.generate_qr_data(class_id, secret)
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size), Image.LANCZOS)
        return img

    def get_time_remaining(self):
        current_time = time.time()
        time_step_end = ((int(current_time) // self.refresh_interval) + 1) * self.refresh_interval
        return int(time_step_end - current_time)

    def verify_qr_data(self, qr_data_str, class_id, secret):
        try:
            data = json.loads(qr_data_str)
        except (json.JSONDecodeError, TypeError):
            return False, "Invalid QR code format"

        if data.get("class_id") != class_id:
            return False, "QR code is for a different class"

        # Verify the token
        current_step = self._get_time_step()
        scanned_step = data.get("ts", 0)

        # Allow current and previous time step (grace period)
        if abs(current_step - scanned_step) > 1:
            return False, "QR code has expired. Please scan the latest code"

        expected_token, _ = self._generate_token(class_id, secret)
        # Also check previous step
        prev_message = f"{class_id}:{scanned_step}".encode()
        prev_token = hmac.new(secret.encode(), prev_message, hashlib.sha256).hexdigest()[:16]

        if data.get("token") == expected_token or data.get("token") == prev_token:
            return True, "QR code verified successfully"

        return False, "Invalid QR code token"

    def scan_qr_from_camera(self):
        if not PYZBAR_AVAILABLE:
            return None, "pyzbar library not installed"

        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Cannot access camera"

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Failed to capture frame"

        return self.scan_qr_from_frame(frame)

    def scan_qr_from_frame(self, frame):
        if not PYZBAR_AVAILABLE:
            return None, "pyzbar library not installed"

        decoded = pyzbar_decode(frame)
        for obj in decoded:
            data = obj.data.decode("utf-8")
            try:
                json.loads(data)
                return data, "QR code scanned successfully"
            except json.JSONDecodeError:
                continue

        return None, "No valid QR code found"

    def qr_image_to_bytes(self, img):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
