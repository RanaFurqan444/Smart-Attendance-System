# Smart Attendance Management System

A comprehensive **Final Year Project** for automated attendance management using **Face Recognition**, **Dynamic QR Codes**, **GPS Verification**, and **AI Analytics**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-green.svg)
![SQLite](https://img.shields.io/badge/Database-SQLite-orange.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red.svg)

---

## Features

### Admin Panel
- **Student Management** – Add, view, and manage student records
- **Teacher Management** – Add and manage teachers
- **Subject Management** – Create subjects and assign teachers
- **Student Enrollment** – Enroll students in subjects
- **Face Registration** – Register student faces via image or camera
- **Attendance Overview** – View attendance across all subjects
- **Report Export** – Export attendance reports to Excel
- **AI Analytics** – Visual charts with pie charts, bar graphs, and trend lines
- **SMS Alerts** – Send absent notifications (Twilio integration)
- **System Settings** – Configure GPS, QR refresh rate, attendance thresholds

### Teacher Panel
- **Dashboard** – View today's classes and active sessions
- **Start Class** – Launch a class session with auto-generated Dynamic QR
- **Dynamic QR Code** – QR refreshes every 30 seconds to prevent sharing
- **Take Attendance** – Manual attendance marking with verification status
- **Subject Analytics** – Per-subject attendance charts and statistics
- **SMS Alerts** – Send absent alerts directly from class view

### Student Panel
- **Dashboard** – View personal attendance summary across subjects
- **QR Scan Attendance** – Scan Dynamic QR code to mark attendance
- **Face Detection Attendance** – Use face recognition to verify identity
- **Attendance Records** – View personal attendance history with charts
- **Subject List** – View enrolled subjects

### Anti-Proxy Attendance System
- **Dynamic QR** – QR codes change every 30 seconds using HMAC-SHA256
- **Face Verification** – Biometric identity confirmation
- **GPS Verification** – Ensures student is physically in the classroom
- **Multi-factor** – Combines QR + Face + GPS for maximum security

---

## Tech Stack

| Module | Technology |
|--------|-----------|
| Frontend/UI | CustomTkinter |
| Backend | Python 3.8+ |
| Database | SQLite |
| Face Recognition | OpenCV + face_recognition |
| QR System | qrcode + pyzbar |
| Reports | Pandas + openpyxl (Excel) |
| Charts | Matplotlib |
| SMS Alerts | Twilio |
| GPS | Haversine formula |
| Security | bcrypt + HMAC-SHA256 |

---

## Project Structure

```
Smart-Attendance-System/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── database/
│   ├── __init__.py
│   └── db_manager.py               # SQLite database operations
├── auth/
│   ├── __init__.py
│   └── auth_manager.py             # Authentication logic
├── face_recognition_module/
│   ├── __init__.py
│   └── face_manager.py             # Face encoding & verification
├── qr_system/
│   ├── __init__.py
│   └── qr_manager.py               # Dynamic QR generation & verification
├── gps/
│   ├── __init__.py
│   └── gps_verifier.py             # GPS location verification
├── analytics/
│   ├── __init__.py
│   └── analytics_manager.py        # AI analytics & report export
├── alerts/
│   ├── __init__.py
│   └── sms_manager.py              # SMS notification system
├── ui/
│   ├── __init__.py
│   ├── login_window.py             # Login screen
│   ├── admin_dashboard.py          # Admin dashboard
│   ├── teacher_dashboard.py        # Teacher dashboard
│   ├── student_interface.py        # Student interface
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py              # Sidebar navigation
│       ├── data_table.py           # Reusable data table
│       └── charts.py               # Chart components
├── assets/
│   ├── faces/                      # Stored face images
│   ├── icons/
│   └── images/
└── reports/                         # Generated Excel reports
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Camera (optional, for face recognition and QR scanning)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/RanaFurqan444/Smart-Attendance-System.git
   cd Smart-Attendance-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies for face_recognition (Linux)**
   ```bash
   sudo apt-get install cmake libopenblas-dev liblapack-dev
   sudo apt-get install libdlib-dev
   ```

5. **Install pyzbar system dependency (Linux)**
   ```bash
   sudo apt-get install libzbar0
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

---

## Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |

> Create teacher and student accounts from the Admin panel.

---

## How It Works

### Dynamic QR Code System
1. Teacher starts a class → system generates a unique HMAC-SHA256 secret
2. QR code encodes: `class_id + token + timestamp`
3. Token regenerates every **30 seconds**
4. Student scans QR → system verifies token validity (allows ±1 time step grace)
5. Old/shared QR codes are automatically rejected

### Face Recognition Flow
1. Admin registers student face (from image or camera capture)
2. System generates 128-dimensional face encoding
3. Student uses camera → system matches face against all registered encodings
4. Match threshold: 50% (configurable)
5. Cross-checks: face must match the logged-in student (prevents proxy)

### GPS Verification
1. Admin/Teacher sets class coordinates (latitude, longitude)
2. Student provides their GPS coordinates
3. System calculates Haversine distance
4. Attendance only marked if within configured radius (default: 100m)

### SMS Alerts
1. Configure Twilio credentials in Settings or SMS panel
2. System identifies absent students after a class
3. Sends automated SMS alerts to registered phone numbers
4. All SMS activity logged for audit trail

---

## Screenshots

The application features a modern dark-themed UI built with CustomTkinter:

- **Login Screen** – Role-based authentication (Admin/Teacher/Student)
- **Admin Dashboard** – Statistics cards, attendance trends, low-attendance alerts
- **Teacher QR Panel** – Live Dynamic QR code with countdown timer
- **Student Attendance** – Pie charts and attendance history
- **Analytics** – Bar charts, line graphs, and detailed tables

---

## SMS Configuration (Optional)

To enable real SMS alerts, configure Twilio:

1. Create a [Twilio account](https://www.twilio.com/)
2. Get your Account SID, Auth Token, and a phone number
3. Enter credentials in **Admin → SMS Alerts → Configure Twilio**

Without Twilio configuration, SMS alerts run in **simulated mode** (logged but not actually sent).

---

## License

This project is developed as a Final Year Project for academic purposes.

---

## Author

**Rana Furqan** – [GitHub](https://github.com/RanaFurqan444)
