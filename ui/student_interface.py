"""
Student Interface - QR scanning, face detection attendance, and viewing records.
"""

import customtkinter as ctk
from tkinter import messagebox
import json
from datetime import datetime
from ui.components.sidebar import Sidebar
from ui.components.data_table import DataTable
from ui.components.charts import AttendanceChart
from gps.gps_verifier import GPSVerifier


class StudentInterface(ctk.CTkFrame):
    def __init__(self, parent, db, analytics, face_manager, qr_manager,
                 user_info, on_logout, **kwargs):
        super().__init__(parent, **kwargs)
        self.db = db
        self.analytics = analytics
        self.face_mgr = face_manager
        self.qr_mgr = qr_manager
        self.user_info = user_info
        self.on_logout = on_logout
        self.student = self.db.get_student_by_user_id(user_info["id"])

        self.configure(fg_color="transparent")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        menu_items = [
            ("dashboard", "Dashboard"),
            ("scan_qr", "Scan QR Code"),
            ("face_attendance", "Face Attendance"),
            ("my_attendance", "My Attendance"),
            ("my_subjects", "My Subjects"),
        ]
        self.sidebar = Sidebar(self, menu_items, self._navigate, user_info=user_info)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._navigate("dashboard")

    def _clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def _navigate(self, page):
        if page == "logout":
            self.on_logout()
            return
        self._clear_content()
        pages = {
            "dashboard": self._show_dashboard,
            "scan_qr": self._show_scan_qr,
            "face_attendance": self._show_face_attendance,
            "my_attendance": self._show_my_attendance,
            "my_subjects": self._show_my_subjects,
        }
        if page in pages:
            pages[page]()

    def _show_dashboard(self):
        if not self.student:
            ctk.CTkLabel(self.content, text="Student profile not found",
                         text_color="#EF4444").pack(pady=50)
            return

        title = ctk.CTkLabel(self.content, text=f"Welcome, {self.user_info['full_name']}",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Student info card
        info_frame = ctk.CTkFrame(self.content, corner_radius=10)
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        info_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        info_data = [
            ("Roll Number", self.student["student_id"]),
            ("Department", self.student["department"]),
            ("Semester", str(self.student["semester"])),
            ("Face Registered", "Yes" if self.student["face_registered"] else "No"),
        ]
        for idx, (label, value) in enumerate(info_data):
            ctk.CTkLabel(info_frame, text=label, font=ctk.CTkFont(size=11),
                         text_color="gray").grid(row=0, column=idx, padx=15, pady=(10, 0))
            ctk.CTkLabel(info_frame, text=value, font=ctk.CTkFont(size=14, weight="bold")
                         ).grid(row=1, column=idx, padx=15, pady=(0, 10))

        # Subject-wise attendance
        subjects = self.db.get_subjects_by_student(self.student["id"])
        if subjects:
            ctk.CTkLabel(self.content, text="My Attendance Summary",
                         font=ctk.CTkFont(size=16, weight="bold")).grid(
                row=2, column=0, sticky="w", pady=(10, 5))

            att_data = []
            for sub in subjects:
                percentage = self.analytics.get_attendance_percentage(
                    self.student["id"], sub["id"]
                )
                status = "Good" if percentage >= 75 else ("Warning" if percentage >= 50 else "Critical")
                att_data.append({
                    "subject": sub["name"],
                    "code": sub["code"],
                    "percentage": f"{percentage}%",
                    "status": status
                })

            columns = [
                {"key": "subject", "label": "Subject", "width": 200},
                {"key": "code", "label": "Code", "width": 100},
                {"key": "percentage", "label": "Attendance %", "width": 120},
                {"key": "status", "label": "Status", "width": 100},
            ]
            DataTable(self.content, columns, att_data).grid(
                row=3, column=0, columnspan=2, sticky="nsew")
            self.content.grid_rowconfigure(3, weight=1)
        else:
            ctk.CTkLabel(self.content, text="No subjects enrolled",
                         text_color="gray").grid(row=2, column=0, pady=20)

    def _show_scan_qr(self):
        title = ctk.CTkLabel(self.content, text="Scan QR Code",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(pady=(0, 10), anchor="w")

        ctk.CTkLabel(self.content, text="Paste QR Data or scan from camera:",
                     font=ctk.CTkFont(size=13)).pack(anchor="w", pady=(5, 5))

        qr_text = ctk.CTkTextbox(self.content, height=100, width=500)
        qr_text.pack(pady=(0, 10), anchor="w")

        # GPS input
        gps_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        gps_frame.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(gps_frame, text="Your Latitude:").pack(side="left", padx=(0, 5))
        lat_entry = ctk.CTkEntry(gps_frame, width=120)
        lat_entry.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(gps_frame, text="Your Longitude:").pack(side="left", padx=(0, 5))
        lon_entry = ctk.CTkEntry(gps_frame, width=120)
        lon_entry.pack(side="left")

        status_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=13))
        status_label.pack(anchor="w", pady=5)

        detail_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=11),
                                    text_color="gray")
        detail_label.pack(anchor="w", pady=2)

        def verify_and_mark():
            if not self.student:
                status_label.configure(text="Student profile not found", text_color="#EF4444")
                return

            qr_data = qr_text.get("1.0", "end").strip()
            if not qr_data:
                status_label.configure(text="Please enter QR data", text_color="#EF4444")
                return

            try:
                data = json.loads(qr_data)
                class_id = data.get("class_id")
            except (json.JSONDecodeError, TypeError):
                status_label.configure(text="Invalid QR data", text_color="#EF4444")
                return

            # Get class info
            class_info = self.db.get_active_class(class_id)
            if not class_info:
                status_label.configure(text="Class not found or not active", text_color="#EF4444")
                return

            # Verify QR token
            secret = class_info.get("qr_secret")
            qr_valid, qr_msg = self.qr_mgr.verify_qr_data(qr_data, class_id, secret)
            if not qr_valid:
                status_label.configure(text=f"QR Failed: {qr_msg}", text_color="#EF4444")
                return

            # GPS verification
            gps_verified = 0
            gps_msg = "GPS not checked"
            if class_info.get("latitude") and class_info.get("longitude"):
                try:
                    s_lat = float(lat_entry.get())
                    s_lon = float(lon_entry.get())
                    gps_ok, gps_msg = GPSVerifier.verify_location(
                        s_lat, s_lon,
                        class_info["latitude"], class_info["longitude"],
                        class_info.get("radius_meters", 100)
                    )
                    gps_verified = 1 if gps_ok else 0
                    if not gps_ok:
                        status_label.configure(text=f"GPS Failed: {gps_msg}", text_color="#EF4444")
                        return
                except (ValueError, TypeError):
                    gps_msg = "GPS coordinates not provided"

            # Mark attendance
            att_id = self.db.mark_attendance(
                self.student["id"], class_id, "present", "qr",
                qr_verified=1, gps_verified=gps_verified
            )
            if att_id:
                status_label.configure(text="Attendance marked successfully!", text_color="#22C55E")
                detail_label.configure(text=f"QR: Verified | GPS: {gps_msg}")
            else:
                status_label.configure(text="Already marked for this class", text_color="#F59E0B")

        def scan_camera():
            data, msg = self.qr_mgr.scan_qr_from_camera()
            if data:
                qr_text.delete("1.0", "end")
                qr_text.insert("1.0", data)
                status_label.configure(text="QR scanned! Click Verify to mark attendance.",
                                       text_color="#3B82F6")
            else:
                status_label.configure(text=msg, text_color="#EF4444")

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.pack(anchor="w", pady=10)
        ctk.CTkButton(btn_frame, text="Verify & Mark", fg_color="#22C55E",
                      command=verify_and_mark).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Scan from Camera", fg_color="#3B82F6",
                      command=scan_camera).pack(side="left")

    def _show_face_attendance(self):
        title = ctk.CTkLabel(self.content, text="Face Detection Attendance",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(pady=(0, 10), anchor="w")

        if not self.student or not self.student["face_registered"]:
            ctk.CTkLabel(self.content,
                         text="Your face is not registered. Please contact admin.",
                         text_color="#EF4444", font=ctk.CTkFont(size=14)).pack(pady=20)
            return

        ctk.CTkLabel(self.content, text="Class ID:").pack(anchor="w", pady=(10, 0))
        class_entry = ctk.CTkEntry(self.content, width=200, height=35)
        class_entry.pack(anchor="w", pady=5)

        # GPS
        gps_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        gps_frame.pack(anchor="w", pady=5)
        ctk.CTkLabel(gps_frame, text="Latitude:").pack(side="left", padx=(0, 5))
        lat_entry = ctk.CTkEntry(gps_frame, width=120)
        lat_entry.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(gps_frame, text="Longitude:").pack(side="left", padx=(0, 5))
        lon_entry = ctk.CTkEntry(gps_frame, width=120)
        lon_entry.pack(side="left")

        status_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=13))
        status_label.pack(anchor="w", pady=10)

        def verify_face():
            class_id_str = class_entry.get().strip()
            if not class_id_str:
                status_label.configure(text="Enter class ID", text_color="#EF4444")
                return

            try:
                class_id = int(class_id_str)
            except ValueError:
                status_label.configure(text="Invalid class ID", text_color="#EF4444")
                return

            class_info = self.db.get_active_class(class_id)
            if not class_info:
                status_label.configure(text="Class not found or not active", text_color="#EF4444")
                return

            status_label.configure(text="Detecting face... Please look at the camera",
                                   text_color="#3B82F6")
            self.update()

            student_id, msg = self.face_mgr.verify_face_from_camera()

            if student_id is None:
                status_label.configure(text=f"Face verification failed: {msg}", text_color="#EF4444")
                return

            if student_id != self.student["id"]:
                status_label.configure(text="Face does not match your profile! Proxy detected.",
                                       text_color="#EF4444")
                return

            # GPS check
            gps_verified = 0
            if class_info.get("latitude") and class_info.get("longitude"):
                try:
                    s_lat = float(lat_entry.get())
                    s_lon = float(lon_entry.get())
                    gps_ok, gps_msg = GPSVerifier.verify_location(
                        s_lat, s_lon, class_info["latitude"], class_info["longitude"],
                        class_info.get("radius_meters", 100)
                    )
                    gps_verified = 1 if gps_ok else 0
                    if not gps_ok:
                        status_label.configure(text=f"GPS Failed: {gps_msg}", text_color="#EF4444")
                        return
                except (ValueError, TypeError):
                    pass

            att_id = self.db.mark_attendance(
                self.student["id"], class_id, "present", "face",
                face_verified=1, gps_verified=gps_verified
            )
            if att_id:
                status_label.configure(text=f"Attendance marked! {msg}", text_color="#22C55E")
            else:
                status_label.configure(text="Already marked for this class", text_color="#F59E0B")

        ctk.CTkButton(self.content, text="Verify Face & Mark Attendance",
                      fg_color="#8B5CF6", hover_color="#7C3AED", height=42,
                      command=verify_face).pack(anchor="w", pady=5)

    def _show_my_attendance(self):
        if not self.student:
            return

        title = ctk.CTkLabel(self.content, text="My Attendance Records",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Subject filter
        subjects = self.db.get_subjects_by_student(self.student["id"])
        sub_map = {"All Subjects": None}
        sub_map.update({s["name"]: s["id"] for s in subjects})
        sub_var = ctk.StringVar(value="All Subjects")

        ctk.CTkLabel(self.content, text="Subject:").grid(row=1, column=0, sticky="w", pady=5)
        ctk.CTkOptionMenu(self.content, variable=sub_var, values=list(sub_map.keys()),
                          width=250, command=lambda v: load(sub_map.get(v))
                          ).grid(row=1, column=1, sticky="w", pady=5)

        # Chart
        chart_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 5))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        # Table
        table_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        table_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_rowconfigure(3, weight=1)

        def load(subject_id):
            for w in chart_frame.winfo_children():
                w.destroy()
            for w in table_frame.winfo_children():
                w.destroy()

            records = self.db.get_attendance_by_student(self.student["id"], subject_id)

            present = sum(1 for r in records if r["status"] == "present")
            late = sum(1 for r in records if r["status"] == "late")
            absent = sum(1 for r in records if r["status"] == "absent")

            chart = AttendanceChart(chart_frame)
            chart.pack(fill="both", expand=True)
            chart.plot_pie_chart(present, absent, late, "My Attendance")

            columns = [
                {"key": "date", "label": "Date", "width": 110},
                {"key": "start_time", "label": "Time", "width": 80},
                {"key": "subject_name", "label": "Subject", "width": 180},
                {"key": "status", "label": "Status", "width": 80},
            ]
            for r in records:
                r["status"] = r["status"].title() if r["status"] else "Absent"

            DataTable(table_frame, columns, records).pack(fill="both", expand=True)

        load(None)

    def _show_my_subjects(self):
        if not self.student:
            return
        title = ctk.CTkLabel(self.content, text="My Subjects",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        subjects = self.db.get_subjects_by_student(self.student["id"])
        columns = [
            {"key": "code", "label": "Code", "width": 100},
            {"key": "name", "label": "Subject Name", "width": 250},
            {"key": "department", "label": "Department", "width": 150},
            {"key": "semester", "label": "Semester", "width": 80},
        ]
        DataTable(self.content, columns, subjects).grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)
