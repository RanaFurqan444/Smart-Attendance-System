"""
Database Manager - SQLite database operations for Smart Attendance System.
"""

import sqlite3
import os
import bcrypt
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="attendance.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()

    def get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn

    def initialize_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                role TEXT NOT NULL CHECK(role IN ('admin', 'teacher', 'student')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                student_id TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                semester INTEGER NOT NULL,
                face_encoding BLOB,
                face_registered INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                employee_id TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT UNIQUE NOT NULL,
                teacher_id INTEGER,
                department TEXT NOT NULL,
                semester INTEGER NOT NULL,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                teacher_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                radius_meters REAL DEFAULT 100.0,
                qr_secret TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
                FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('present', 'absent', 'late')),
                method TEXT CHECK(method IN ('qr', 'face', 'manual', 'qr+face')),
                face_verified INTEGER DEFAULT 0,
                qr_verified INTEGER DEFAULT 0,
                gps_verified INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE,
                UNIQUE(student_id, class_id)
            );

            CREATE TABLE IF NOT EXISTS student_subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
                UNIQUE(student_id, subject_id)
            );

            CREATE TABLE IF NOT EXISTS sms_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                class_id INTEGER,
                message TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'sent',
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Create default admin if not exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        if cursor.fetchone()[0] == 0:
            password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, role, email) VALUES (?, ?, ?, ?, ?)",
                ("admin", password_hash, "System Administrator", "admin", "admin@system.com")
            )

        conn.commit()

    # ---- User Management ----
    def authenticate_user(self, username, password):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,))
        user = cursor.fetchone()
        if user and bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            return dict(user)
        return None

    def create_user(self, username, password, full_name, role, email="", phone=""):
        conn = self.get_connection()
        cursor = conn.cursor()
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, full_name, role, email, phone) VALUES (?,?,?,?,?,?)",
                (username, password_hash, full_name, role, email, phone)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_all_users(self, role=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        if role:
            cursor.execute("SELECT * FROM users WHERE role=? AND is_active=1", (role,))
        else:
            cursor.execute("SELECT * FROM users WHERE is_active=1")
        return [dict(row) for row in cursor.fetchall()]

    def delete_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
        conn.commit()

    def update_user(self, user_id, **kwargs):
        conn = self.get_connection()
        cursor = conn.cursor()
        valid_fields = ["full_name", "email", "phone", "username"]
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [user_id]
        cursor.execute(f"UPDATE users SET {set_clause} WHERE id=?", values)
        conn.commit()

    # ---- Student Management ----
    def add_student(self, user_id, student_id, department, semester):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO students (user_id, student_id, department, semester) VALUES (?,?,?,?)",
                (user_id, student_id, department, semester)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_all_students(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.username, u.full_name, u.email, u.phone
            FROM students s
            JOIN users u ON s.user_id = u.id
            WHERE u.is_active=1
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_student_by_id(self, student_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.username, u.full_name, u.email, u.phone
            FROM students s JOIN users u ON s.user_id = u.id
            WHERE s.student_id=? AND u.is_active=1
        """, (student_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_student_by_user_id(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.username, u.full_name, u.email, u.phone
            FROM students s JOIN users u ON s.user_id = u.id
            WHERE s.user_id=? AND u.is_active=1
        """, (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_face_encoding(self, student_db_id, encoding_bytes):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET face_encoding=?, face_registered=1 WHERE id=?",
            (encoding_bytes, student_db_id)
        )
        conn.commit()

    def get_students_with_faces(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.full_name FROM students s
            JOIN users u ON s.user_id = u.id
            WHERE s.face_registered=1 AND u.is_active=1
        """)
        return [dict(row) for row in cursor.fetchall()]

    # ---- Teacher Management ----
    def add_teacher(self, user_id, employee_id, department):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO teachers (user_id, employee_id, department) VALUES (?,?,?)",
                (user_id, employee_id, department)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_all_teachers(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.*, u.username, u.full_name, u.email, u.phone
            FROM teachers t JOIN users u ON t.user_id = u.id
            WHERE u.is_active=1
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_teacher_by_user_id(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.*, u.username, u.full_name, u.email, u.phone
            FROM teachers t JOIN users u ON t.user_id = u.id
            WHERE t.user_id=? AND u.is_active=1
        """, (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ---- Subject Management ----
    def add_subject(self, name, code, teacher_id, department, semester):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO subjects (name, code, teacher_id, department, semester) VALUES (?,?,?,?,?)",
                (name, code, teacher_id, department, semester)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_all_subjects(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, t.employee_id, u.full_name as teacher_name
            FROM subjects s
            LEFT JOIN teachers t ON s.teacher_id = t.id
            LEFT JOIN users u ON t.user_id = u.id
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_subjects_by_teacher(self, teacher_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM subjects WHERE teacher_id=?", (teacher_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_subjects_by_student(self, student_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sub.* FROM subjects sub
            JOIN student_subjects ss ON sub.id = ss.subject_id
            WHERE ss.student_id=?
        """, (student_id,))
        return [dict(row) for row in cursor.fetchall()]

    def enroll_student_in_subject(self, student_id, subject_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO student_subjects (student_id, subject_id) VALUES (?,?)",
                (student_id, subject_id)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ---- Class Management ----
    def create_class(self, subject_id, teacher_id, date, start_time, end_time,
                     latitude=None, longitude=None, radius=100.0, qr_secret=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO classes (subject_id, teacher_id, date, start_time, end_time,
               latitude, longitude, radius_meters, qr_secret)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (subject_id, teacher_id, date, start_time, end_time,
             latitude, longitude, radius, qr_secret)
        )
        conn.commit()
        return cursor.lastrowid

    def get_classes_by_teacher(self, teacher_id, date=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        if date:
            cursor.execute("""
                SELECT c.*, sub.name as subject_name, sub.code as subject_code
                FROM classes c JOIN subjects sub ON c.subject_id = sub.id
                WHERE c.teacher_id=? AND c.date=?
                ORDER BY c.start_time
            """, (teacher_id, date))
        else:
            cursor.execute("""
                SELECT c.*, sub.name as subject_name, sub.code as subject_code
                FROM classes c JOIN subjects sub ON c.subject_id = sub.id
                WHERE c.teacher_id=?
                ORDER BY c.date DESC, c.start_time
            """, (teacher_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_active_class(self, class_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, sub.name as subject_name
            FROM classes c JOIN subjects sub ON c.subject_id = sub.id
            WHERE c.id=? AND c.is_active=1
        """, (class_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_class_by_id(self, class_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, sub.name as subject_name
            FROM classes c JOIN subjects sub ON c.subject_id = sub.id
            WHERE c.id=?
        """, (class_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_class_qr_secret(self, class_id, qr_secret):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE classes SET qr_secret=? WHERE id=?", (qr_secret, class_id))
        conn.commit()

    def end_class(self, class_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE classes SET is_active=0 WHERE id=?", (class_id,))
        conn.commit()

    # ---- Attendance Management ----
    def mark_attendance(self, student_id, class_id, status="present", method="manual",
                        face_verified=0, qr_verified=0, gps_verified=0):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO attendance (student_id, class_id, status, method,
                   face_verified, qr_verified, gps_verified)
                   VALUES (?,?,?,?,?,?,?)""",
                (student_id, class_id, status, method, face_verified, qr_verified, gps_verified)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Already marked
            return None

    def get_attendance_by_class(self, class_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.*, s.student_id as roll_number, u.full_name
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            JOIN users u ON s.user_id = u.id
            WHERE a.class_id=?
            ORDER BY u.full_name
        """, (class_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_by_student(self, student_db_id, subject_id=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        if subject_id:
            cursor.execute("""
                SELECT a.*, c.date, c.start_time, sub.name as subject_name, sub.code
                FROM attendance a
                JOIN classes c ON a.class_id = c.id
                JOIN subjects sub ON c.subject_id = sub.id
                WHERE a.student_id=? AND c.subject_id=?
                ORDER BY c.date DESC
            """, (student_db_id, subject_id))
        else:
            cursor.execute("""
                SELECT a.*, c.date, c.start_time, sub.name as subject_name, sub.code
                FROM attendance a
                JOIN classes c ON a.class_id = c.id
                JOIN subjects sub ON c.subject_id = sub.id
                WHERE a.student_id=?
                ORDER BY c.date DESC
            """, (student_db_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_summary(self, subject_id=None, date_from=None, date_to=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        query = """
            SELECT s.student_id as roll_number, u.full_name,
                   sub.name as subject_name, sub.code as subject_code,
                   COUNT(CASE WHEN a.status='present' THEN 1 END) as present_count,
                   COUNT(CASE WHEN a.status='absent' THEN 1 END) as absent_count,
                   COUNT(CASE WHEN a.status='late' THEN 1 END) as late_count,
                   COUNT(a.id) as total_classes
            FROM students s
            JOIN users u ON s.user_id = u.id
            JOIN student_subjects ss ON s.id = ss.student_id
            JOIN subjects sub ON ss.subject_id = sub.id
            LEFT JOIN classes c ON c.subject_id = sub.id
            LEFT JOIN attendance a ON a.student_id = s.id AND a.class_id = c.id
            WHERE u.is_active=1
        """
        params = []
        if subject_id:
            query += " AND sub.id=?"
            params.append(subject_id)
        if date_from:
            query += " AND c.date>=?"
            params.append(date_from)
        if date_to:
            query += " AND c.date<=?"
            params.append(date_to)
        query += " GROUP BY s.id, sub.id ORDER BY u.full_name, sub.name"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_total_classes_for_subject(self, subject_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM classes WHERE subject_id=?", (subject_id,))
        return cursor.fetchone()["count"]

    def get_absentees_for_class(self, class_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, u.full_name, u.phone
            FROM students s
            JOIN users u ON s.user_id = u.id
            JOIN student_subjects ss ON s.id = ss.student_id
            JOIN classes c ON c.subject_id = ss.subject_id AND c.id = ?
            WHERE s.id NOT IN (
                SELECT student_id FROM attendance WHERE class_id = ?
            ) AND u.is_active = 1
        """, (class_id, class_id))
        return [dict(row) for row in cursor.fetchall()]

    # ---- SMS Logs ----
    def log_sms(self, student_id, class_id, message, status="sent"):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sms_logs (student_id, class_id, message, status) VALUES (?,?,?,?)",
            (student_id, class_id, message, status)
        )
        conn.commit()

    # ---- Settings ----
    def get_setting(self, key, default=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)",
            (key, str(value))
        )
        conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
