"""
Teacher Dashboard - Class management, QR generation, and attendance tracking.
"""

import customtkinter as ctk
from tkinter import messagebox
from datetime import datetime
from PIL import Image, ImageTk
from io import BytesIO
from ui.components.sidebar import Sidebar
from ui.components.data_table import DataTable
from ui.components.charts import AttendanceChart


class TeacherDashboard(ctk.CTkFrame):
    def __init__(self, parent, db, analytics, face_manager, qr_manager, sms_manager,
                 user_info, on_logout, **kwargs):
        super().__init__(parent, **kwargs)
        self.db = db
        self.analytics = analytics
        self.face_mgr = face_manager
        self.qr_mgr = qr_manager
        self.sms_mgr = sms_manager
        self.user_info = user_info
        self.on_logout = on_logout
        self.teacher = self.db.get_teacher_by_user_id(user_info["id"])
        self.active_class = None
        self.qr_timer = None

        self.configure(fg_color="transparent")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        menu_items = [
            ("dashboard", "Dashboard"),
            ("my_classes", "My Classes"),
            ("start_class", "Start Class"),
            ("take_attendance", "Take Attendance"),
            ("my_subjects", "My Subjects"),
            ("analytics", "Analytics"),
        ]
        self.sidebar = Sidebar(self, menu_items, self._navigate, user_info=user_info)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._navigate("dashboard")

    def _clear_content(self):
        if self.qr_timer:
            self.after_cancel(self.qr_timer)
            self.qr_timer = None
        for widget in self.content.winfo_children():
            widget.destroy()

    def _navigate(self, page):
        if page == "logout":
            self.on_logout()
            return
        self._clear_content()
        pages = {
            "dashboard": self._show_dashboard,
            "my_classes": self._show_classes,
            "start_class": self._show_start_class,
            "take_attendance": self._show_take_attendance,
            "my_subjects": self._show_subjects,
            "analytics": self._show_analytics,
        }
        if page in pages:
            pages[page]()

    def _show_dashboard(self):
        if not self.teacher:
            ctk.CTkLabel(self.content, text="Teacher profile not found",
                         font=ctk.CTkFont(size=16), text_color="#EF4444").pack(pady=50)
            return

        title = ctk.CTkLabel(self.content, text=f"Welcome, {self.user_info['full_name']}",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        subjects = self.db.get_subjects_by_teacher(self.teacher["id"])
        today = datetime.now().strftime("%Y-%m-%d")
        today_classes = self.db.get_classes_by_teacher(self.teacher["id"], date=today)

        # Stats
        cards = ctk.CTkFrame(self.content, fg_color="transparent")
        cards.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        cards.grid_columnconfigure((0, 1, 2), weight=1)

        for idx, (label, value, color) in enumerate([
            ("My Subjects", str(len(subjects)), "#3B82F6"),
            ("Today's Classes", str(len(today_classes)), "#22C55E"),
            ("Active Class", "Yes" if self.active_class else "No", "#F59E0B"),
        ]):
            card = ctk.CTkFrame(cards, corner_radius=10, height=90)
            card.grid(row=0, column=idx, padx=5, sticky="nsew")
            card.grid_propagate(False)
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=26, weight="bold"),
                         text_color=color).pack(pady=(18, 2))
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=12), text_color="gray").pack()

        # Today's schedule
        ctk.CTkLabel(self.content, text="Today's Schedule",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=2, column=0, sticky="w", pady=(10, 5))

        if today_classes:
            columns = [
                {"key": "subject_name", "label": "Subject", "width": 200},
                {"key": "start_time", "label": "Start", "width": 100},
                {"key": "end_time", "label": "End", "width": 100},
                {"key": "is_active", "label": "Active", "width": 80},
            ]
            for c in today_classes:
                c["is_active"] = "Active" if c["is_active"] else "Ended"
            DataTable(self.content, columns, today_classes).grid(
                row=3, column=0, columnspan=2, sticky="nsew")
            self.content.grid_rowconfigure(3, weight=1)
        else:
            ctk.CTkLabel(self.content, text="No classes scheduled today",
                         text_color="gray").grid(row=3, column=0, sticky="w")

    def _show_classes(self):
        if not self.teacher:
            return
        title = ctk.CTkLabel(self.content, text="My Classes",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        classes = self.db.get_classes_by_teacher(self.teacher["id"])
        columns = [
            {"key": "subject_name", "label": "Subject", "width": 180},
            {"key": "date", "label": "Date", "width": 110},
            {"key": "start_time", "label": "Start", "width": 90},
            {"key": "end_time", "label": "End", "width": 90},
            {"key": "is_active", "label": "Status", "width": 80},
        ]
        for c in classes:
            c["is_active"] = "Active" if c["is_active"] else "Ended"

        DataTable(self.content, columns, classes).grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)

    def _show_start_class(self):
        if not self.teacher:
            return
        title = ctk.CTkLabel(self.content, text="Start a New Class",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        subjects = self.db.get_subjects_by_teacher(self.teacher["id"])
        if not subjects:
            ctk.CTkLabel(self.content, text="No subjects assigned",
                         text_color="gray").grid(row=1, column=0, pady=20)
            return

        sub_map = {s["name"]: s["id"] for s in subjects}

        ctk.CTkLabel(self.content, text="Subject:").grid(row=1, column=0, sticky="w", pady=5)
        sub_var = ctk.StringVar(value=list(sub_map.keys())[0])
        ctk.CTkOptionMenu(self.content, variable=sub_var, values=list(sub_map.keys()),
                          width=300).grid(row=1, column=1, sticky="w", pady=5)

        ctk.CTkLabel(self.content, text="Start Time:").grid(row=2, column=0, sticky="w", pady=5)
        start_entry = ctk.CTkEntry(self.content, width=200)
        start_entry.grid(row=2, column=1, sticky="w", pady=5)
        start_entry.insert(0, datetime.now().strftime("%H:%M"))

        ctk.CTkLabel(self.content, text="End Time:").grid(row=3, column=0, sticky="w", pady=5)
        end_entry = ctk.CTkEntry(self.content, width=200)
        end_entry.grid(row=3, column=1, sticky="w", pady=5)

        ctk.CTkLabel(self.content, text="Class Latitude:").grid(row=4, column=0, sticky="w", pady=5)
        lat_entry = ctk.CTkEntry(self.content, width=200)
        lat_entry.grid(row=4, column=1, sticky="w", pady=5)
        lat_entry.insert(0, self.db.get_setting("default_latitude", ""))

        ctk.CTkLabel(self.content, text="Class Longitude:").grid(row=5, column=0, sticky="w", pady=5)
        lon_entry = ctk.CTkEntry(self.content, width=200)
        lon_entry.grid(row=5, column=1, sticky="w", pady=5)
        lon_entry.insert(0, self.db.get_setting("default_longitude", ""))

        status = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=13))
        status.grid(row=7, column=0, columnspan=2, sticky="w", pady=5)

        # QR display area
        self.qr_frame = ctk.CTkFrame(self.content, corner_radius=10, width=350, height=400)
        self.qr_frame.grid(row=1, column=2, rowspan=7, padx=(30, 0), sticky="n")
        self.qr_frame.grid_propagate(False)
        self.qr_label = ctk.CTkLabel(self.qr_frame, text="QR Code will appear here\nafter starting class",
                                     text_color="gray")
        self.qr_label.pack(expand=True)

        self.timer_label = ctk.CTkLabel(self.qr_frame, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self.timer_label.pack(pady=(0, 10))

        def start():
            sub_id = sub_map.get(sub_var.get())
            lat = float(lat_entry.get()) if lat_entry.get() else None
            lon = float(lon_entry.get()) if lon_entry.get() else None

            secret = self.qr_mgr.generate_class_secret()
            today = datetime.now().strftime("%Y-%m-%d")

            class_id = self.db.create_class(
                sub_id, self.teacher["id"], today,
                start_entry.get(), end_entry.get() or "N/A",
                lat, lon, qr_secret=secret
            )
            self.active_class = {"id": class_id, "secret": secret}
            self.db.update_class_qr_secret(class_id, secret)

            status.configure(text=f"Class started! ID: {class_id}", text_color="#22C55E")
            self._update_qr()

        def end_class():
            if self.active_class:
                self.db.end_class(self.active_class["id"])
                if self.qr_timer:
                    self.after_cancel(self.qr_timer)
                    self.qr_timer = None
                status.configure(text="Class ended", text_color="#F59E0B")
                self.active_class = None
                for w in self.qr_frame.winfo_children():
                    w.destroy()
                ctk.CTkLabel(self.qr_frame, text="Class ended", text_color="gray").pack(expand=True)

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=10)
        ctk.CTkButton(btn_frame, text="Start Class", fg_color="#22C55E",
                      command=start).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="End Class", fg_color="#EF4444",
                      command=end_class).pack(side="left")

    def _update_qr(self):
        if not self.active_class:
            return

        for w in self.qr_frame.winfo_children():
            w.destroy()

        class_id = self.active_class["id"]
        secret = self.active_class["secret"]

        qr_img = self.qr_mgr.generate_qr_image(class_id, secret, size=280)
        qr_bytes = self.qr_mgr.qr_image_to_bytes(qr_img)
        pil_img = Image.open(BytesIO(qr_bytes))
        tk_img = ImageTk.PhotoImage(pil_img)

        img_label = ctk.CTkLabel(self.qr_frame, text="", image=tk_img)
        img_label.image = tk_img
        img_label.pack(pady=(10, 5))

        remaining = self.qr_mgr.get_time_remaining()
        timer = ctk.CTkLabel(self.qr_frame, text=f"Refreshing in {remaining}s",
                             font=ctk.CTkFont(size=13, weight="bold"),
                             text_color="#F59E0B")
        timer.pack()

        ctk.CTkLabel(self.qr_frame, text=f"Class ID: {class_id}",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(5, 0))
        ctk.CTkLabel(self.qr_frame, text="Dynamic QR - changes every 30s",
                     font=ctk.CTkFont(size=10), text_color="gray").pack()

        # Refresh every second to update timer, regenerate QR every 30s
        self.qr_timer = self.after(1000, self._qr_tick)

    def _qr_tick(self):
        if not self.active_class:
            return
        remaining = self.qr_mgr.get_time_remaining()
        if remaining <= 1:
            self._update_qr()
        else:
            # Update timer label only
            for w in self.qr_frame.winfo_children():
                if isinstance(w, ctk.CTkLabel) and "Refreshing" in str(w.cget("text")):
                    w.configure(text=f"Refreshing in {remaining}s")
                    break
            self.qr_timer = self.after(1000, self._qr_tick)

    def _show_take_attendance(self):
        if not self.teacher:
            return
        title = ctk.CTkLabel(self.content, text="Take Attendance",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Select class
        classes = self.db.get_classes_by_teacher(self.teacher["id"])
        active_classes = [c for c in classes if c["is_active"]]

        if not active_classes:
            ctk.CTkLabel(self.content, text="No active classes. Start a class first.",
                         text_color="gray", font=ctk.CTkFont(size=14)).grid(row=1, column=0, pady=20)
            return

        class_map = {f"{c['subject_name']} - {c['date']} {c['start_time']}": c["id"]
                     for c in active_classes}

        ctk.CTkLabel(self.content, text="Select Class:").grid(row=1, column=0, sticky="w", pady=5)
        class_var = ctk.StringVar(value=list(class_map.keys())[0])
        ctk.CTkOptionMenu(self.content, variable=class_var, values=list(class_map.keys()),
                          width=350).grid(row=1, column=1, sticky="w", pady=5)

        # Manual attendance
        ctk.CTkLabel(self.content, text="Roll Number:").grid(row=2, column=0, sticky="w", pady=5)
        roll_entry = ctk.CTkEntry(self.content, width=200)
        roll_entry.grid(row=2, column=1, sticky="w", pady=5)

        status_var = ctk.StringVar(value="present")
        ctk.CTkLabel(self.content, text="Status:").grid(row=3, column=0, sticky="w", pady=5)
        status_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        status_frame.grid(row=3, column=1, sticky="w", pady=5)
        for val in ["present", "late", "absent"]:
            ctk.CTkRadioButton(status_frame, text=val.title(), variable=status_var,
                               value=val).pack(side="left", padx=5)

        result_label = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=12))
        result_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

        def mark_manual():
            class_id = class_map.get(class_var.get())
            roll = roll_entry.get().strip()
            if not roll:
                result_label.configure(text="Enter roll number", text_color="#EF4444")
                return
            student = self.db.get_student_by_id(roll)
            if not student:
                result_label.configure(text="Student not found", text_color="#EF4444")
                return
            att_id = self.db.mark_attendance(student["id"], class_id,
                                             status=status_var.get(), method="manual")
            if att_id:
                result_label.configure(text=f"Attendance marked for {student['full_name']}",
                                       text_color="#22C55E")
                roll_entry.delete(0, "end")
                load_attendance()
            else:
                result_label.configure(text="Already marked", text_color="#F59E0B")

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        ctk.CTkButton(btn_frame, text="Mark Manual", fg_color="#22C55E",
                      command=mark_manual).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Send Absent Alerts", fg_color="#EF4444",
                      command=lambda: self._send_class_alerts(class_map.get(class_var.get()))
                      ).pack(side="left")

        # Attendance list
        columns = [
            {"key": "roll_number", "label": "Roll No", "width": 100},
            {"key": "full_name", "label": "Name", "width": 180},
            {"key": "status", "label": "Status", "width": 80},
            {"key": "method", "label": "Method", "width": 80},
            {"key": "face_verified", "label": "Face", "width": 60},
            {"key": "qr_verified", "label": "QR", "width": 60},
            {"key": "gps_verified", "label": "GPS", "width": 60},
            {"key": "timestamp", "label": "Time", "width": 150},
        ]

        self._att_list_table = DataTable(self.content, columns)
        self._att_list_table.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.content.grid_rowconfigure(6, weight=1)

        def load_attendance():
            class_id = class_map.get(class_var.get())
            if class_id:
                records = self.db.get_attendance_by_class(class_id)
                for r in records:
                    r["face_verified"] = "Yes" if r["face_verified"] else "No"
                    r["qr_verified"] = "Yes" if r["qr_verified"] else "No"
                    r["gps_verified"] = "Yes" if r["gps_verified"] else "No"
                self._att_list_table.refresh(records)

        load_attendance()

    def _send_class_alerts(self, class_id):
        if not class_id:
            return
        results, msg = self.sms_mgr.send_bulk_absent_alerts(class_id)
        summary = f"{msg}\n" + "\n".join(
            f"{'OK' if r['success'] else 'FAIL'}: {r['student']}" for r in results
        )
        messagebox.showinfo("SMS Alerts", summary)

    def _show_subjects(self):
        if not self.teacher:
            return
        title = ctk.CTkLabel(self.content, text="My Subjects",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        subjects = self.db.get_subjects_by_teacher(self.teacher["id"])
        columns = [
            {"key": "code", "label": "Code", "width": 100},
            {"key": "name", "label": "Subject", "width": 200},
            {"key": "department", "label": "Department", "width": 150},
            {"key": "semester", "label": "Semester", "width": 80},
        ]
        DataTable(self.content, columns, subjects).grid(row=1, column=0, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)

    def _show_analytics(self):
        if not self.teacher:
            return
        title = ctk.CTkLabel(self.content, text="Subject Analytics",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        subjects = self.db.get_subjects_by_teacher(self.teacher["id"])
        if not subjects:
            ctk.CTkLabel(self.content, text="No subjects assigned",
                         text_color="gray").grid(row=1, column=0)
            return

        sub_map = {s["name"]: s["id"] for s in subjects}
        sub_var = ctk.StringVar(value=list(sub_map.keys())[0])
        ctk.CTkOptionMenu(self.content, variable=sub_var, values=list(sub_map.keys()),
                          width=300, command=lambda v: load(sub_map[v])
                          ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        chart_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        chart_frame.grid_columnconfigure((0, 1), weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        table_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        table_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_rowconfigure(3, weight=1)

        def load(subject_id):
            for w in chart_frame.winfo_children():
                w.destroy()
            for w in table_frame.winfo_children():
                w.destroy()

            data = self.analytics.get_subject_analytics(subject_id)
            if not data:
                ctk.CTkLabel(chart_frame, text="No data", text_color="gray").grid(row=0, column=0)
                return

            tp = sum(a["present"] for a in data)
            tl = sum(a["late"] for a in data)
            ta = sum(a["absent"] for a in data)

            pie = AttendanceChart(chart_frame)
            pie.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
            pie.plot_pie_chart(tp, ta, tl)

            bar = AttendanceChart(chart_frame)
            bar.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
            bar.plot_bar_chart([a["name"][:12] for a in data],
                               [a["percentage"] for a in data], "Student %")

            columns = [
                {"key": "roll_number", "label": "Roll No", "width": 100},
                {"key": "name", "label": "Name", "width": 150},
                {"key": "present", "label": "Present", "width": 70},
                {"key": "late", "label": "Late", "width": 60},
                {"key": "absent", "label": "Absent", "width": 70},
                {"key": "percentage", "label": "%", "width": 60},
                {"key": "status", "label": "Status", "width": 80},
            ]
            DataTable(table_frame, columns, data).pack(fill="both", expand=True)

        load(list(sub_map.values())[0])
