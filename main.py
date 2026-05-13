"""
Smart Attendance Management System
Final Year Project

Features:
- Face Recognition attendance
- Dynamic QR Code (refreshes every 30 seconds)
- GPS Verification
- Admin, Teacher, and Student dashboards
- AI Analytics with charts
- Excel report export
- SMS absent alerts
- Proxy attendance prevention
"""

import customtkinter as ctk
from database.db_manager import DatabaseManager
from auth.auth_manager import AuthManager
from face_recognition_module.face_manager import FaceManager
from qr_system.qr_manager import QRManager
from analytics.analytics_manager import AnalyticsManager
from alerts.sms_manager import SMSManager
from ui.login_window import LoginWindow
from ui.admin_dashboard import AdminDashboard
from ui.teacher_dashboard import TeacherDashboard
from ui.student_interface import StudentInterface


class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Smart Attendance Management System")
        self.geometry("1280x780")
        self.minsize(1024, 600)

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize backend services
        self.db = DatabaseManager()
        self.auth = AuthManager(self.db)
        self.face_mgr = FaceManager(self.db)
        self.qr_mgr = QRManager(self.db)
        self.analytics = AnalyticsManager(self.db)
        self.sms_mgr = SMSManager(self.db)

        # Main container
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.current_view = None
        self._show_login()

    def _clear_view(self):
        if self.current_view:
            self.current_view.destroy()
            self.current_view = None

    def _show_login(self):
        self._clear_view()
        self.auth.logout()
        self.current_view = LoginWindow(
            self, self.auth, self._on_login_success
        )
        self.current_view.grid(row=0, column=0, sticky="nsew")

    def _on_login_success(self, user):
        self._clear_view()
        role = user["role"]

        if role == "admin":
            self.current_view = AdminDashboard(
                self, self.db, self.analytics, self.face_mgr,
                self.qr_mgr, self.sms_mgr, user, self._show_login
            )
        elif role == "teacher":
            self.current_view = TeacherDashboard(
                self, self.db, self.analytics, self.face_mgr,
                self.qr_mgr, self.sms_mgr, user, self._show_login
            )
        elif role == "student":
            self.current_view = StudentInterface(
                self, self.db, self.analytics, self.face_mgr,
                self.qr_mgr, user, self._show_login
            )

        if self.current_view:
            self.current_view.grid(row=0, column=0, sticky="nsew")

    def on_closing(self):
        self.db.close()
        self.destroy()


def main():
    app = AttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
