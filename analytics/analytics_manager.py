"""
AI Analytics and Report Generation Module.
Provides attendance analytics, trend analysis, and Excel report export.
"""

import pandas as pd
from datetime import datetime, timedelta
import os


class AnalyticsManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)

    def get_attendance_percentage(self, student_db_id, subject_id=None):
        records = self.db.get_attendance_by_student(student_db_id, subject_id)
        if not records:
            return 0.0
        present = sum(1 for r in records if r["status"] in ("present", "late"))
        return round((present / len(records)) * 100, 1)

    def get_subject_analytics(self, subject_id):
        summary = self.db.get_attendance_summary(subject_id=subject_id)
        total_classes = self.db.get_total_classes_for_subject(subject_id)

        analytics = []
        for record in summary:
            present = record["present_count"] or 0
            late = record["late_count"] or 0
            total = total_classes if total_classes > 0 else 1
            percentage = round(((present + late) / total) * 100, 1)

            status = "Good"
            if percentage < 50:
                status = "Critical"
            elif percentage < 75:
                status = "Warning"

            analytics.append({
                "roll_number": record["roll_number"],
                "name": record["full_name"],
                "present": present,
                "late": late,
                "absent": total - present - late,
                "total": total,
                "percentage": percentage,
                "status": status
            })

        return analytics

    def get_overall_analytics(self):
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Total students
        cursor.execute("SELECT COUNT(*) as count FROM students s JOIN users u ON s.user_id=u.id WHERE u.is_active=1")
        total_students = cursor.fetchone()["count"]

        # Total teachers
        cursor.execute("SELECT COUNT(*) as count FROM teachers t JOIN users u ON t.user_id=u.id WHERE u.is_active=1")
        total_teachers = cursor.fetchone()["count"]

        # Total subjects
        cursor.execute("SELECT COUNT(*) as count FROM subjects")
        total_subjects = cursor.fetchone()["count"]

        # Total classes conducted
        cursor.execute("SELECT COUNT(*) as count FROM classes")
        total_classes = cursor.fetchone()["count"]

        # Average attendance rate
        cursor.execute("""
            SELECT
                COUNT(CASE WHEN status='present' OR status='late' THEN 1 END) as present,
                COUNT(*) as total
            FROM attendance
        """)
        att = cursor.fetchone()
        avg_attendance = round((att["present"] / att["total"]) * 100, 1) if att["total"] > 0 else 0

        # Today's classes
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) as count FROM classes WHERE date=?", (today,))
        today_classes = cursor.fetchone()["count"]

        return {
            "total_students": total_students,
            "total_teachers": total_teachers,
            "total_subjects": total_subjects,
            "total_classes": total_classes,
            "avg_attendance": avg_attendance,
            "today_classes": today_classes
        }

    def get_attendance_trend(self, days=30):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        cursor.execute("""
            SELECT c.date,
                   COUNT(CASE WHEN a.status='present' OR a.status='late' THEN 1 END) as present,
                   COUNT(a.id) as total
            FROM classes c
            LEFT JOIN attendance a ON c.id = a.class_id
            WHERE c.date >= ? AND c.date <= ?
            GROUP BY c.date
            ORDER BY c.date
        """, (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

        trend = []
        for row in cursor.fetchall():
            row = dict(row)
            percentage = round((row["present"] / row["total"]) * 100, 1) if row["total"] > 0 else 0
            trend.append({
                "date": row["date"],
                "percentage": percentage,
                "present": row["present"],
                "total": row["total"]
            })
        return trend

    def get_low_attendance_students(self, threshold=75):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.student_id as roll_number, u.full_name, u.phone,
                   sub.name as subject_name, sub.code,
                   COUNT(CASE WHEN a.status='present' OR a.status='late' THEN 1 END) as present,
                   COUNT(a.id) as total
            FROM students s
            JOIN users u ON s.user_id = u.id
            JOIN student_subjects ss ON s.id = ss.student_id
            JOIN subjects sub ON ss.subject_id = sub.id
            LEFT JOIN classes c ON c.subject_id = sub.id
            LEFT JOIN attendance a ON a.student_id = s.id AND a.class_id = c.id
            WHERE u.is_active = 1
            GROUP BY s.id, sub.id
            HAVING total > 0
        """)

        low_students = []
        for row in cursor.fetchall():
            row = dict(row)
            percentage = round((row["present"] / row["total"]) * 100, 1) if row["total"] > 0 else 0
            if percentage < threshold:
                row["percentage"] = percentage
                low_students.append(row)
        return low_students

    def export_attendance_report(self, subject_id=None, date_from=None, date_to=None,
                                 filename=None):
        summary = self.db.get_attendance_summary(
            subject_id=subject_id, date_from=date_from, date_to=date_to
        )

        if not summary:
            return None, "No data to export"

        data = []
        for record in summary:
            present = record["present_count"] or 0
            absent = record["absent_count"] or 0
            late = record["late_count"] or 0
            total = present + absent + late
            percentage = round(((present + late) / total) * 100, 1) if total > 0 else 0

            data.append({
                "Roll Number": record["roll_number"],
                "Student Name": record["full_name"],
                "Subject": record["subject_name"],
                "Subject Code": record["subject_code"],
                "Present": present,
                "Late": late,
                "Absent": absent,
                "Total Classes": total,
                "Attendance %": percentage,
                "Status": "Good" if percentage >= 75 else ("Warning" if percentage >= 50 else "Critical")
            })

        df = pd.DataFrame(data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"attendance_report_{timestamp}.xlsx"

        filepath = os.path.join(self.reports_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Attendance Report", index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets["Attendance Report"]
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = max_length

        return filepath, f"Report exported successfully to {filepath}"

    def export_detailed_report(self, subject_id, filename=None):
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.date, c.start_time, s.student_id as roll_number,
                   u.full_name, a.status, a.method,
                   a.face_verified, a.qr_verified, a.gps_verified
            FROM classes c
            JOIN subjects sub ON c.subject_id = sub.id
            JOIN student_subjects ss ON ss.subject_id = sub.id
            JOIN students s ON ss.student_id = s.id
            JOIN users u ON s.user_id = u.id
            LEFT JOIN attendance a ON a.class_id = c.id AND a.student_id = s.id
            WHERE sub.id = ? AND u.is_active = 1
            ORDER BY c.date, c.start_time, u.full_name
        """, (subject_id,))

        rows = [dict(r) for r in cursor.fetchall()]
        if not rows:
            return None, "No data to export"

        data = []
        for row in rows:
            data.append({
                "Date": row["date"],
                "Time": row["start_time"],
                "Roll Number": row["roll_number"],
                "Student Name": row["full_name"],
                "Status": row["status"] or "absent",
                "Method": row["method"] or "N/A",
                "Face Verified": "Yes" if row["face_verified"] else "No",
                "QR Verified": "Yes" if row["qr_verified"] else "No",
                "GPS Verified": "Yes" if row["gps_verified"] else "No",
            })

        df = pd.DataFrame(data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_report_{timestamp}.xlsx"

        filepath = os.path.join(self.reports_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Detailed Report", index=False)
            worksheet = writer.sheets["Detailed Report"]
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                col_letter = chr(65 + idx) if idx < 26 else chr(64 + idx // 26) + chr(65 + idx % 26)
                worksheet.column_dimensions[col_letter].width = max_length

        return filepath, f"Detailed report exported to {filepath}"
