"""
Authentication Manager for Smart Attendance System.
"""


class AuthManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.current_user = None

    def login(self, username, password):
        user = self.db.authenticate_user(username, password)
        if user:
            self.current_user = user
            return user
        return None

    def logout(self):
        self.current_user = None

    def get_current_user(self):
        return self.current_user

    def is_logged_in(self):
        return self.current_user is not None

    def get_role(self):
        if self.current_user:
            return self.current_user["role"]
        return None

    def change_password(self, user_id, old_password, new_password):
        import bcrypt
        user = self.db.get_connection().execute(
            "SELECT password_hash FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if user and bcrypt.checkpw(old_password.encode(), user["password_hash"].encode()):
            new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            self.db.get_connection().execute(
                "UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id)
            )
            self.db.get_connection().commit()
            return True
        return False
