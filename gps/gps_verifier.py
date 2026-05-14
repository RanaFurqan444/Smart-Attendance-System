"""
GPS Verification Module.
Verifies if a student is within the allowed radius of the classroom.
"""

import math


class GPSVerifier:
    DEFAULT_RADIUS = 100  # meters

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth's radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @classmethod
    def verify_location(cls, student_lat, student_lon, class_lat, class_lon,
                        allowed_radius=None):
        if class_lat is None or class_lon is None:
            return True, "GPS verification not required for this class"

        if student_lat is None or student_lon is None:
            return False, "Student location not available"

        if allowed_radius is None:
            allowed_radius = cls.DEFAULT_RADIUS

        distance = cls.haversine_distance(student_lat, student_lon, class_lat, class_lon)

        if distance <= allowed_radius:
            return True, f"Location verified (distance: {distance:.0f}m)"
        else:
            return False, f"You are {distance:.0f}m away from class (max: {allowed_radius}m)"

    @staticmethod
    def get_current_location():
        """
        Get current GPS coordinates.
        In a desktop app, this would use system location services.
        For demo purposes, returns a configurable location.
        """
        try:
            from geopy.geocoders import Nominatim
            return None, None, "Use manual GPS input for desktop application"
        except ImportError:
            return None, None, "geopy not installed"
