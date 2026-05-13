"""
SMS Alert Manager.
Sends absent alerts to students/parents via SMS.
Supports Twilio integration (configurable).
"""

import os
from datetime import datetime


class SMSManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.twilio_client = None
        self.twilio_from = None
        self._initialize_twilio()

    def _initialize_twilio(self):
        account_sid = self.db.get_setting("twilio_account_sid")
        auth_token = self.db.get_setting("twilio_auth_token")
        from_number = self.db.get_setting("twilio_from_number")

        if account_sid and auth_token and from_number:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(account_sid, auth_token)
                self.twilio_from = from_number
            except ImportError:
                pass

    def configure_twilio(self, account_sid, auth_token, from_number):
        self.db.set_setting("twilio_account_sid", account_sid)
        self.db.set_setting("twilio_auth_token", auth_token)
        self.db.set_setting("twilio_from_number", from_number)
        self._initialize_twilio()

    def send_absent_alert(self, student_info, class_info):
        phone = student_info.get("phone")
        if not phone:
            return False, "No phone number registered"

        student_name = student_info.get("full_name", "Student")
        subject = class_info.get("subject_name", "Class")
        date = class_info.get("date", datetime.now().strftime("%Y-%m-%d"))

        message = (
            f"Alert: {student_name} was marked absent in {subject} on {date}. "
            f"Please contact the administration for details."
        )

        if self.twilio_client:
            try:
                msg = self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_from,
                    to=phone
                )
                self.db.log_sms(student_info["id"], class_info.get("id"), message, "sent")
                return True, f"SMS sent successfully (SID: {msg.sid})"
            except Exception as e:
                self.db.log_sms(student_info["id"], class_info.get("id"), message, "failed")
                return False, f"SMS failed: {str(e)}"
        else:
            # Log as simulated
            self.db.log_sms(
                student_info["id"], class_info.get("id"), message, "simulated"
            )
            return True, f"SMS simulated (Twilio not configured): {message}"

    def send_bulk_absent_alerts(self, class_id):
        class_info = self.db.get_class_by_id(class_id)
        if not class_info:
            return [], "Class not found"

        absentees = self.db.get_absentees_for_class(class_id)
        results = []

        for student in absentees:
            success, msg = self.send_absent_alert(student, class_info)
            results.append({
                "student": student["full_name"],
                "phone": student.get("phone", "N/A"),
                "success": success,
                "message": msg
            })

        return results, f"Processed {len(results)} alerts"

    def get_sms_history(self, limit=50):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sl.*, u.full_name, s.student_id as roll_number
            FROM sms_logs sl
            JOIN students s ON sl.student_id = s.id
            JOIN users u ON s.user_id = u.id
            ORDER BY sl.sent_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def is_configured(self):
        return self.twilio_client is not None
