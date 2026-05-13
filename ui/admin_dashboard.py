"""
Admin Dashboard - Full management interface for administrators.
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
from ui.components.sidebar import Sidebar
from ui.components.data_table import DataTable
from ui.components.charts import AttendanceChart


class AdminDashboard(ctk.CTkFrame):
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

        self.configure(fg_color="transparent")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        menu_items = [
            ("dashboard", "Dashboard"),
            ("students", "Students"),
            ("teachers", "Teachers"),
            ("subjects", "Subjects"),
            ("attendance", "Attendance"),
            ("reports", "Reports"),
            ("analytics", "Analytics"),
            ("sms", "SMS Alerts"),
            ("settings", "Settings"),
        ]
        self.sidebar = Sidebar(self, menu_items, self._navigate, user_info=user_info)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.current_page = None
        self._navigate("dashboard")

    def _clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def _navigate(self, page):
        if page == "logout":
            self.on_logout()
            return
        self._clear_content()
        self.current_page = page
        pages = {
            "dashboard": self._show_dashboard,
            "students": self._show_students,
            "teachers": self._show_teachers,
            "subjects": self._show_subjects,
            "attendance": self._show_attendance,
            "reports": self._show_reports,
            "analytics": self._show_analytics,
            "sms": self._show_sms,
            "settings": self._show_settings,
        }
        if page in pages:
            pages[page]()

    # ---- Dashboard ----
    def _show_dashboard(self):
        stats = self.analytics.get_overall_analytics()

        title = ctk.CTkLabel(self.content, text="Admin Dashboard",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 15))

        # Stats cards
        cards_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        cards_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        card_data = [
            ("Total Students", str(stats["total_students"]), "#3B82F6"),
            ("Total Teachers", str(stats["total_teachers"]), "#22C55E"),
            ("Total Subjects", str(stats["total_subjects"]), "#F59E0B"),
            ("Avg Attendance", f"{stats['avg_attendance']}%", "#8B5CF6"),
        ]

        for idx, (label, value, color) in enumerate(card_data):
            card = ctk.CTkFrame(cards_frame, corner_radius=10, height=100)
            card.grid(row=0, column=idx, padx=5, sticky="nsew")
            card.grid_propagate(False)
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=28, weight="bold"),
                         text_color=color).pack(pady=(20, 2))
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=12),
                         text_color="gray").pack()

        # Charts row
        chart_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        chart_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        chart_frame.grid_columnconfigure((0, 1), weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        # Attendance trend
        trend_data = self.analytics.get_attendance_trend(30)
        trend_chart = AttendanceChart(chart_frame)
        trend_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        if trend_data:
            dates = [d["date"][-5:] for d in trend_data]
            values = [d["percentage"] for d in trend_data]
            trend_chart.plot_line_chart(dates, values, "30-Day Attendance Trend")
        else:
            trend_chart.plot_line_chart([], [], "30-Day Attendance Trend")

        # Low attendance students
        low_frame = ctk.CTkFrame(chart_frame, corner_radius=10)
        low_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ctk.CTkLabel(low_frame, text="Low Attendance Alerts",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#EF4444").pack(pady=(10, 5))

        low_students = self.analytics.get_low_attendance_students(75)
        if low_students:
            for s in low_students[:8]:
                text = f"{s['full_name']} - {s['subject_name']}: {s['percentage']}%"
                ctk.CTkLabel(low_frame, text=text, font=ctk.CTkFont(size=11),
                             text_color="#F59E0B").pack(padx=10, anchor="w")
        else:
            ctk.CTkLabel(low_frame, text="No alerts", font=ctk.CTkFont(size=12),
                         text_color="gray").pack(pady=20)

    # ---- Students ----
    def _show_students(self):
        title = ctk.CTkLabel(self.content, text="Student Management",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Action buttons
        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkButton(btn_frame, text="+ Add Student", fg_color="#22C55E",
                      hover_color="#16A34A", command=self._add_student_dialog
                      ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Register Face", fg_color="#3B82F6",
                      hover_color="#2563EB", command=self._register_face_dialog
                      ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Refresh", fg_color="gray40",
                      command=lambda: self._navigate("students")
                      ).pack(side="left")

        # Table
        columns = [
            {"key": "student_id", "label": "Roll Number", "width": 120},
            {"key": "full_name", "label": "Full Name", "width": 180},
            {"key": "department", "label": "Department", "width": 120},
            {"key": "semester", "label": "Semester", "width": 80},
            {"key": "email", "label": "Email", "width": 180},
            {"key": "phone", "label": "Phone", "width": 130},
            {"key": "face_registered", "label": "Face", "width": 70},
        ]

        students = self.db.get_all_students()
        for s in students:
            s["face_registered"] = "Yes" if s.get("face_registered") else "No"

        self.students_table = DataTable(self.content, columns, students)
        self.students_table.grid(row=2, column=0, sticky="nsew")
        self.content.grid_rowconfigure(2, weight=1)

    def _add_student_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add New Student")
        dialog.geometry("450x550")
        dialog.transient(self)
        dialog.grab_set()
        dialog.after(150, lambda: dialog.focus_force())

        container = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        fields = {}
        field_list = [
            ("Full Name", "full_name"),
            ("Username", "username"),
            ("Password", "password"),
            ("Roll Number", "student_id"),
            ("Department", "department"),
            ("Semester", "semester"),
            ("Email", "email"),
            ("Phone", "phone"),
        ]

        for label, key in field_list:
            ctk.CTkLabel(container, text=label, font=ctk.CTkFont(size=13)).pack(padx=10, pady=(8, 0), anchor="w")
            entry = ctk.CTkEntry(container, height=35, width=350)
            if key == "password":
                entry.configure(show="*")
            entry.pack(padx=10, pady=(2, 0))
            fields[key] = entry

        def save():
            data = {k: e.get().strip() for k, e in fields.items()}
            if not all(data.get(k) for k in ["full_name", "username", "password", "student_id", "department"]):
                messagebox.showerror("Error", "Please fill all required fields", parent=dialog)
                return

            user_id = self.db.create_user(
                data["username"], data["password"], data["full_name"],
                "student", data["email"], data["phone"]
            )
            if not user_id:
                messagebox.showerror("Error", "Username already exists", parent=dialog)
                return

            semester = int(data.get("semester") or 1)
            result = self.db.add_student(user_id, data["student_id"], data["department"], semester)
            if result:
                messagebox.showinfo("Success", "Student added successfully", parent=dialog)
                dialog.destroy()
                self._navigate("students")
            else:
                messagebox.showerror("Error", "Roll number already exists", parent=dialog)

        ctk.CTkButton(container, text="Save Student", fg_color="#22C55E",
                      hover_color="#16A34A", height=40, command=save
                      ).pack(padx=10, pady=15)

    def _register_face_dialog(self):
        selected = self.students_table.get_selected() if hasattr(self, 'students_table') else None
        dialog = ctk.CTkToplevel(self)
        dialog.title("Register Face")
        dialog.geometry("450x300")
        dialog.transient(self)
        dialog.grab_set()
        dialog.after(150, lambda: dialog.focus_force())

        container = ctk.CTkFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(container, text="Register Student Face",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 10))

        ctk.CTkLabel(container, text="Roll Number:").pack(pady=(10, 0))
        roll_entry = ctk.CTkEntry(container, height=35, width=250)
        roll_entry.pack(pady=(5, 10))

        if selected:
            roll_entry.insert(0, selected.get("student_id", ""))

        status_label = ctk.CTkLabel(container, text="", font=ctk.CTkFont(size=12))
        status_label.pack(pady=5)

        def from_image():
            roll = roll_entry.get().strip()
            if not roll:
                status_label.configure(text="Enter roll number", text_color="#EF4444")
                return
            student = self.db.get_student_by_id(roll)
            if not student:
                status_label.configure(text="Student not found", text_color="#EF4444")
                return
            filepath = filedialog.askopenfilename(
                title="Select Face Image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
                parent=dialog
            )
            if filepath:
                success, msg = self.face_mgr.register_face_from_image(
                    filepath, student["id"], student["student_id"]
                )
                color = "#22C55E" if success else "#EF4444"
                status_label.configure(text=msg, text_color=color)
                if success:
                    self.face_mgr.reload_faces()

        def from_camera():
            roll = roll_entry.get().strip()
            if not roll:
                status_label.configure(text="Enter roll number", text_color="#EF4444")
                return
            student = self.db.get_student_by_id(roll)
            if not student:
                status_label.configure(text="Student not found", text_color="#EF4444")
                return
            success, msg = self.face_mgr.register_face_from_camera(
                student["id"], student["student_id"]
            )
            color = "#22C55E" if success else "#EF4444"
            status_label.configure(text=msg, text_color=color)
            if success:
                self.face_mgr.reload_faces()

        btn_frame = ctk.CTkFrame(container, fg_color="transparent")
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="From Image", fg_color="#3B82F6",
                      command=from_image).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="From Camera", fg_color="#8B5CF6",
                      command=from_camera).pack(side="left", padx=10)

    # ---- Teachers ----
    def _show_teachers(self):
        title = ctk.CTkLabel(self.content, text="Teacher Management",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkButton(btn_frame, text="+ Add Teacher", fg_color="#22C55E",
                      hover_color="#16A34A", command=self._add_teacher_dialog
                      ).pack(side="left", padx=(0, 10))

        columns = [
            {"key": "employee_id", "label": "Employee ID", "width": 130},
            {"key": "full_name", "label": "Full Name", "width": 200},
            {"key": "department", "label": "Department", "width": 150},
            {"key": "email", "label": "Email", "width": 200},
            {"key": "phone", "label": "Phone", "width": 140},
        ]
        teachers = self.db.get_all_teachers()
        DataTable(self.content, columns, teachers).grid(row=2, column=0, sticky="nsew")
        self.content.grid_rowconfigure(2, weight=1)

    def _add_teacher_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add New Teacher")
        dialog.geometry("450x480")
        dialog.transient(self)
        dialog.grab_set()
        dialog.after(150, lambda: dialog.focus_force())

        container = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        fields = {}
        field_list = [
            ("Full Name", "full_name"),
            ("Username", "username"),
            ("Password", "password"),
            ("Employee ID", "employee_id"),
            ("Department", "department"),
            ("Email", "email"),
            ("Phone", "phone"),
        ]

        for label, key in field_list:
            ctk.CTkLabel(container, text=label).pack(padx=10, pady=(8, 0), anchor="w")
            entry = ctk.CTkEntry(container, height=35, width=350)
            if key == "password":
                entry.configure(show="*")
            entry.pack(padx=10, pady=(2, 0))
            fields[key] = entry

        def save():
            data = {k: e.get().strip() for k, e in fields.items()}
            if not all(data.get(k) for k in ["full_name", "username", "password", "employee_id", "department"]):
                messagebox.showerror("Error", "Fill all required fields", parent=dialog)
                return
            user_id = self.db.create_user(
                data["username"], data["password"], data["full_name"],
                "teacher", data["email"], data["phone"]
            )
            if not user_id:
                messagebox.showerror("Error", "Username already exists", parent=dialog)
                return
            result = self.db.add_teacher(user_id, data["employee_id"], data["department"])
            if result:
                messagebox.showinfo("Success", "Teacher added successfully", parent=dialog)
                dialog.destroy()
                self._navigate("teachers")
            else:
                messagebox.showerror("Error", "Employee ID already exists", parent=dialog)

        ctk.CTkButton(container, text="Save Teacher", fg_color="#22C55E",
                      height=40, command=save
                      ).pack(padx=10, pady=15)

    # ---- Subjects ----
    def _show_subjects(self):
        title = ctk.CTkLabel(self.content, text="Subject Management",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkButton(btn_frame, text="+ Add Subject", fg_color="#22C55E",
                      command=self._add_subject_dialog).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Enroll Students", fg_color="#3B82F6",
                      command=self._enroll_dialog).pack(side="left")

        columns = [
            {"key": "code", "label": "Code", "width": 100},
            {"key": "name", "label": "Subject Name", "width": 200},
            {"key": "teacher_name", "label": "Teacher", "width": 180},
            {"key": "department", "label": "Department", "width": 130},
            {"key": "semester", "label": "Semester", "width": 80},
        ]
        subjects = self.db.get_all_subjects()
        DataTable(self.content, columns, subjects).grid(row=2, column=0, sticky="nsew")
        self.content.grid_rowconfigure(2, weight=1)

    def _add_subject_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add Subject")
        dialog.geometry("450x420")
        dialog.transient(self)
        dialog.grab_set()

        dialog.after(150, lambda: dialog.focus_force())

        container = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        fields = {}
        for label, key in [
            ("Subject Name", "name"), ("Subject Code", "code"),
            ("Department", "department"), ("Semester", "semester")
        ]:
            ctk.CTkLabel(container, text=label).pack(padx=10, pady=(8, 0), anchor="w")
            entry = ctk.CTkEntry(container, height=35, width=350)
            entry.pack(padx=10, pady=(2, 0))
            fields[key] = entry

        ctk.CTkLabel(container, text="Assign Teacher").pack(padx=10, pady=(8, 0), anchor="w")
        teachers = self.db.get_all_teachers()
        teacher_names = {f"{t['full_name']} ({t['employee_id']})": t["id"] for t in teachers}
        teacher_var = ctk.StringVar(value="Select Teacher")
        ctk.CTkOptionMenu(container, variable=teacher_var, values=list(teacher_names.keys()) or ["No teachers"],
                          width=350).pack(padx=10, pady=(2, 0))

        def save():
            data = {k: e.get().strip() for k, e in fields.items()}
            if not all(data.get(k) for k in ["name", "code", "department"]):
                messagebox.showerror("Error", "Fill required fields", parent=dialog)
                return
            teacher_id = teacher_names.get(teacher_var.get())
            semester = int(data.get("semester") or 1)
            result = self.db.add_subject(data["name"], data["code"], teacher_id, data["department"], semester)
            if result:
                messagebox.showinfo("Success", "Subject added", parent=dialog)
                dialog.destroy()
                self._navigate("subjects")
            else:
                messagebox.showerror("Error", "Subject code already exists", parent=dialog)

        ctk.CTkButton(container, text="Save Subject", fg_color="#22C55E", height=40,
                      command=save).pack(padx=10, pady=15)

    def _enroll_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Enroll Students")
        dialog.geometry("450x350")
        dialog.transient(self)
        dialog.grab_set()
        dialog.after(150, lambda: dialog.focus_force())

        container = ctk.CTkFrame(dialog, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(container, text="Enroll Student in Subject",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 10))

        ctk.CTkLabel(container, text="Roll Number:").pack(pady=(10, 0))
        roll_entry = ctk.CTkEntry(container, height=35, width=300)
        roll_entry.pack(pady=5)

        ctk.CTkLabel(container, text="Subject:").pack(pady=(10, 0))
        subjects = self.db.get_all_subjects()
        sub_map = {f"{s['name']} ({s['code']})": s["id"] for s in subjects}
        sub_var = ctk.StringVar(value="Select Subject")
        ctk.CTkOptionMenu(container, variable=sub_var, values=list(sub_map.keys()) or ["No subjects"],
                          width=300).pack(pady=5)

        status = ctk.CTkLabel(container, text="", font=ctk.CTkFont(size=12))
        status.pack(pady=5)

        def enroll():
            roll = roll_entry.get().strip()
            sub_id = sub_map.get(sub_var.get())
            if not roll or not sub_id:
                status.configure(text="Fill all fields", text_color="#EF4444")
                return
            student = self.db.get_student_by_id(roll)
            if not student:
                status.configure(text="Student not found", text_color="#EF4444")
                return
            if self.db.enroll_student_in_subject(student["id"], sub_id):
                status.configure(text="Enrolled successfully!", text_color="#22C55E")
            else:
                status.configure(text="Already enrolled", text_color="#F59E0B")

        ctk.CTkButton(container, text="Enroll", fg_color="#3B82F6", height=40,
                      command=enroll).pack(pady=15)

    # ---- Attendance ----
    def _show_attendance(self):
        title = ctk.CTkLabel(self.content, text="Attendance Records",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Filter
        filter_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(filter_frame, text="Subject:").pack(side="left", padx=(0, 5))
        subjects = self.db.get_all_subjects()
        sub_map = {"All Subjects": None}
        sub_map.update({f"{s['name']} ({s['code']})": s["id"] for s in subjects})
        self._att_sub_var = ctk.StringVar(value="All Subjects")
        ctk.CTkOptionMenu(filter_frame, variable=self._att_sub_var,
                          values=list(sub_map.keys()), width=250,
                          command=lambda v: self._load_attendance(sub_map.get(v))
                          ).pack(side="left", padx=(0, 15))

        columns = [
            {"key": "roll_number", "label": "Roll No", "width": 100},
            {"key": "full_name", "label": "Student Name", "width": 180},
            {"key": "subject_name", "label": "Subject", "width": 150},
            {"key": "subject_code", "label": "Code", "width": 80},
            {"key": "present_count", "label": "Present", "width": 80},
            {"key": "late_count", "label": "Late", "width": 60},
            {"key": "absent_count", "label": "Absent", "width": 80},
            {"key": "percentage", "label": "Attendance %", "width": 100},
        ]

        self._att_table = DataTable(self.content, columns)
        self._att_table.grid(row=2, column=0, sticky="nsew")
        self.content.grid_rowconfigure(2, weight=1)
        self._load_attendance(None)

    def _load_attendance(self, subject_id):
        summary = self.db.get_attendance_summary(subject_id=subject_id)
        for r in summary:
            present = r["present_count"] or 0
            late = r["late_count"] or 0
            total = present + (r["absent_count"] or 0) + late
            r["percentage"] = f"{round(((present + late) / total) * 100, 1)}%" if total > 0 else "0%"
        self._att_table.refresh(summary)

    # ---- Reports ----
    def _show_reports(self):
        title = ctk.CTkLabel(self.content, text="Export Reports",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 15))

        # Subject filter
        ctk.CTkLabel(self.content, text="Select Subject (optional):").grid(
            row=1, column=0, sticky="w", pady=(0, 5))
        subjects = self.db.get_all_subjects()
        sub_map = {"All Subjects": None}
        sub_map.update({f"{s['name']} ({s['code']})": s["id"] for s in subjects})
        self._rpt_sub_var = ctk.StringVar(value="All Subjects")
        ctk.CTkOptionMenu(self.content, variable=self._rpt_sub_var,
                          values=list(sub_map.keys()), width=300
                          ).grid(row=2, column=0, sticky="w", pady=(0, 15))

        status = ctk.CTkLabel(self.content, text="", font=ctk.CTkFont(size=13))
        status.grid(row=4, column=0, sticky="w", pady=(10, 0))

        def export_summary():
            sub_id = sub_map.get(self._rpt_sub_var.get())
            path, msg = self.analytics.export_attendance_report(subject_id=sub_id)
            color = "#22C55E" if path else "#EF4444"
            status.configure(text=msg, text_color=color)

        def export_detailed():
            sub_id = sub_map.get(self._rpt_sub_var.get())
            if not sub_id:
                status.configure(text="Select a specific subject for detailed report", text_color="#EF4444")
                return
            path, msg = self.analytics.export_detailed_report(sub_id)
            color = "#22C55E" if path else "#EF4444"
            status.configure(text=msg, text_color=color)

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="w", pady=(0, 10))
        ctk.CTkButton(btn_frame, text="Export Summary (Excel)", fg_color="#22C55E",
                      command=export_summary).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Export Detailed (Excel)", fg_color="#3B82F6",
                      command=export_detailed).pack(side="left")

    # ---- Analytics ----
    def _show_analytics(self):
        title = ctk.CTkLabel(self.content, text="AI Analytics",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Subject selector
        filter_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        subjects = self.db.get_all_subjects()
        sub_map = {f"{s['name']} ({s['code']})": s["id"] for s in subjects}

        if sub_map:
            sub_var = ctk.StringVar(value=list(sub_map.keys())[0])
            ctk.CTkLabel(filter_frame, text="Subject:").pack(side="left", padx=(0, 5))
            ctk.CTkOptionMenu(filter_frame, variable=sub_var,
                              values=list(sub_map.keys()), width=300,
                              command=lambda v: load_analytics(sub_map.get(v))
                              ).pack(side="left")

        chart_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        chart_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        chart_frame.grid_columnconfigure((0, 1), weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)
        self.content.grid_rowconfigure(2, weight=1)

        self._analytics_chart_frame = chart_frame

        table_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        table_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.content.grid_rowconfigure(3, weight=1)
        self._analytics_table_frame = table_frame

        def load_analytics(subject_id):
            for w in chart_frame.winfo_children():
                w.destroy()
            for w in table_frame.winfo_children():
                w.destroy()

            analytics = self.analytics.get_subject_analytics(subject_id)
            if not analytics:
                ctk.CTkLabel(chart_frame, text="No data available",
                             font=ctk.CTkFont(size=14), text_color="gray").grid(row=0, column=0)
                return

            # Pie chart
            total_present = sum(a["present"] for a in analytics)
            total_late = sum(a["late"] for a in analytics)
            total_absent = sum(a["absent"] for a in analytics)

            pie = AttendanceChart(chart_frame)
            pie.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
            pie.plot_pie_chart(total_present, total_absent, total_late)

            # Bar chart
            bar = AttendanceChart(chart_frame)
            bar.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
            names = [a["name"][:15] for a in analytics]
            percentages = [a["percentage"] for a in analytics]
            bar.plot_bar_chart(names, percentages, "Student Attendance %")

            # Table
            columns = [
                {"key": "roll_number", "label": "Roll No", "width": 100},
                {"key": "name", "label": "Name", "width": 160},
                {"key": "present", "label": "Present", "width": 80},
                {"key": "late", "label": "Late", "width": 60},
                {"key": "absent", "label": "Absent", "width": 80},
                {"key": "percentage", "label": "%", "width": 70},
                {"key": "status", "label": "Status", "width": 90},
            ]
            DataTable(table_frame, columns, analytics).pack(fill="both", expand=True)

        if sub_map:
            load_analytics(list(sub_map.values())[0])

    # ---- SMS ----
    def _show_sms(self):
        title = ctk.CTkLabel(self.content, text="SMS Alerts",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Config status
        configured = self.sms_mgr.is_configured()
        status_text = "Twilio: Connected" if configured else "Twilio: Not configured (simulated mode)"
        status_color = "#22C55E" if configured else "#F59E0B"
        ctk.CTkLabel(self.content, text=status_text, font=ctk.CTkFont(size=12),
                     text_color=status_color).grid(row=1, column=0, sticky="w", pady=(0, 10))

        btn_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkButton(btn_frame, text="Configure Twilio", fg_color="#3B82F6",
                      command=self._configure_twilio_dialog).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Send Absent Alerts", fg_color="#EF4444",
                      command=self._send_alerts_dialog).pack(side="left")

        # SMS History
        ctk.CTkLabel(self.content, text="SMS History",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=3, column=0, sticky="w", pady=(10, 5))

        columns = [
            {"key": "full_name", "label": "Student", "width": 160},
            {"key": "roll_number", "label": "Roll No", "width": 100},
            {"key": "message", "label": "Message", "width": 350},
            {"key": "status", "label": "Status", "width": 90},
            {"key": "sent_at", "label": "Sent At", "width": 150},
        ]
        history = self.sms_mgr.get_sms_history()
        DataTable(self.content, columns, history).grid(row=4, column=0, sticky="nsew")
        self.content.grid_rowconfigure(4, weight=1)

    def _configure_twilio_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Configure Twilio")
        dialog.geometry("450x350")
        dialog.transient(self)
        dialog.grab_set()

        fields = {}
        for idx, (label, key) in enumerate([
            ("Account SID", "sid"), ("Auth Token", "token"), ("From Number", "from")
        ]):
            ctk.CTkLabel(dialog, text=label).grid(row=idx, column=0, padx=20, pady=(15, 0), sticky="w")
            entry = ctk.CTkEntry(dialog, height=35, width=300)
            if key == "token":
                entry.configure(show="*")
            entry.grid(row=idx, column=1, padx=20, pady=(15, 0))
            fields[key] = entry

        def save():
            self.sms_mgr.configure_twilio(
                fields["sid"].get().strip(),
                fields["token"].get().strip(),
                fields["from"].get().strip()
            )
            messagebox.showinfo("Success", "Twilio configured", parent=dialog)
            dialog.destroy()
            self._navigate("sms")

        ctk.CTkButton(dialog, text="Save Configuration", fg_color="#22C55E",
                      height=40, command=save).grid(row=3, column=0, columnspan=2, padx=20, pady=20)

    def _send_alerts_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Send Absent Alerts")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Send SMS to Absentees",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))

        # Select class
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.date, c.start_time, sub.name
            FROM classes c JOIN subjects sub ON c.subject_id = sub.id
            ORDER BY c.date DESC LIMIT 20
        """)
        classes = [dict(r) for r in cursor.fetchall()]
        class_map = {f"{c['name']} - {c['date']} {c['start_time']}": c["id"] for c in classes}

        if not class_map:
            ctk.CTkLabel(dialog, text="No classes found", text_color="gray").pack(pady=20)
            return

        class_var = ctk.StringVar(value=list(class_map.keys())[0])
        ctk.CTkOptionMenu(dialog, variable=class_var, values=list(class_map.keys()),
                          width=400).pack(pady=10)

        result_text = ctk.CTkTextbox(dialog, width=450, height=200)
        result_text.pack(pady=10)

        def send():
            class_id = class_map.get(class_var.get())
            if not class_id:
                return
            results, msg = self.sms_mgr.send_bulk_absent_alerts(class_id)
            result_text.delete("1.0", "end")
            result_text.insert("1.0", f"{msg}\n\n")
            for r in results:
                line = f"{'[OK]' if r['success'] else '[FAIL]'} {r['student']} ({r['phone']}): {r['message']}\n"
                result_text.insert("end", line)

        ctk.CTkButton(dialog, text="Send Alerts", fg_color="#EF4444", command=send).pack(pady=5)

    # ---- Settings ----
    def _show_settings(self):
        title = ctk.CTkLabel(self.content, text="System Settings",
                             font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 15))

        settings_frame = ctk.CTkFrame(self.content, corner_radius=10)
        settings_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))

        # GPS settings
        ctk.CTkLabel(settings_frame, text="GPS Settings",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=20, pady=(15, 5), sticky="w")

        ctk.CTkLabel(settings_frame, text="Default Class Latitude:").grid(
            row=1, column=0, padx=20, pady=5, sticky="w")
        lat_entry = ctk.CTkEntry(settings_frame, width=200)
        lat_entry.grid(row=1, column=1, padx=20, pady=5)
        lat_entry.insert(0, self.db.get_setting("default_latitude", "0.0"))

        ctk.CTkLabel(settings_frame, text="Default Class Longitude:").grid(
            row=2, column=0, padx=20, pady=5, sticky="w")
        lon_entry = ctk.CTkEntry(settings_frame, width=200)
        lon_entry.grid(row=2, column=1, padx=20, pady=5)
        lon_entry.insert(0, self.db.get_setting("default_longitude", "0.0"))

        ctk.CTkLabel(settings_frame, text="GPS Radius (meters):").grid(
            row=3, column=0, padx=20, pady=5, sticky="w")
        radius_entry = ctk.CTkEntry(settings_frame, width=200)
        radius_entry.grid(row=3, column=1, padx=20, pady=5)
        radius_entry.insert(0, self.db.get_setting("gps_radius", "100"))

        # QR settings
        ctk.CTkLabel(settings_frame, text="QR Settings",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=4, column=0, columnspan=2, padx=20, pady=(15, 5), sticky="w")

        ctk.CTkLabel(settings_frame, text="QR Refresh Interval (sec):").grid(
            row=5, column=0, padx=20, pady=5, sticky="w")
        qr_entry = ctk.CTkEntry(settings_frame, width=200)
        qr_entry.grid(row=5, column=1, padx=20, pady=5)
        qr_entry.insert(0, self.db.get_setting("qr_refresh_interval", "30"))

        # Attendance threshold
        ctk.CTkLabel(settings_frame, text="Attendance Threshold (%):").grid(
            row=6, column=0, padx=20, pady=(5, 15), sticky="w")
        thresh_entry = ctk.CTkEntry(settings_frame, width=200)
        thresh_entry.grid(row=6, column=1, padx=20, pady=(5, 15))
        thresh_entry.insert(0, self.db.get_setting("attendance_threshold", "75"))

        def save_settings():
            self.db.set_setting("default_latitude", lat_entry.get())
            self.db.set_setting("default_longitude", lon_entry.get())
            self.db.set_setting("gps_radius", radius_entry.get())
            self.db.set_setting("qr_refresh_interval", qr_entry.get())
            self.db.set_setting("attendance_threshold", thresh_entry.get())
            messagebox.showinfo("Success", "Settings saved successfully")

        ctk.CTkButton(self.content, text="Save Settings", fg_color="#22C55E",
                      height=40, command=save_settings
                      ).grid(row=2, column=0, sticky="w", pady=10)
