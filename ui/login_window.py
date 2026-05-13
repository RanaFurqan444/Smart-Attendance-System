"""
Login Window - Entry point UI for authentication.
"""

import customtkinter as ctk
from tkinter import messagebox


class LoginWindow(ctk.CTkFrame):
    def __init__(self, parent, auth_manager, on_login_success, **kwargs):
        super().__init__(parent, **kwargs)
        self.auth = auth_manager
        self.on_login_success = on_login_success

        self.configure(fg_color="transparent")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Center card
        card = ctk.CTkFrame(self, width=420, height=520, corner_radius=15)
        card.grid(row=0, column=0)
        card.grid_propagate(False)
        card.grid_columnconfigure(0, weight=1)

        # Logo / Title
        title = ctk.CTkLabel(
            card, text="Smart Attendance\nManagement System",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#3B82F6"
        )
        title.grid(row=0, column=0, padx=30, pady=(40, 5))

        subtitle = ctk.CTkLabel(
            card, text="Sign in to continue",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        subtitle.grid(row=1, column=0, padx=30, pady=(0, 30))

        # Username
        user_label = ctk.CTkLabel(card, text="Username", font=ctk.CTkFont(size=13))
        user_label.grid(row=2, column=0, padx=40, pady=(0, 5), sticky="w")

        self.username_entry = ctk.CTkEntry(
            card, placeholder_text="Enter username",
            height=42, corner_radius=8, font=ctk.CTkFont(size=13)
        )
        self.username_entry.grid(row=3, column=0, padx=40, pady=(0, 15), sticky="ew")

        # Password
        pass_label = ctk.CTkLabel(card, text="Password", font=ctk.CTkFont(size=13))
        pass_label.grid(row=4, column=0, padx=40, pady=(0, 5), sticky="w")

        self.password_entry = ctk.CTkEntry(
            card, placeholder_text="Enter password", show="*",
            height=42, corner_radius=8, font=ctk.CTkFont(size=13)
        )
        self.password_entry.grid(row=5, column=0, padx=40, pady=(0, 10), sticky="ew")

        # Role selector
        role_label = ctk.CTkLabel(card, text="Login As", font=ctk.CTkFont(size=13))
        role_label.grid(row=6, column=0, padx=40, pady=(0, 5), sticky="w")

        self.role_var = ctk.StringVar(value="admin")
        role_frame = ctk.CTkFrame(card, fg_color="transparent")
        role_frame.grid(row=7, column=0, padx=40, pady=(0, 20), sticky="ew")
        role_frame.grid_columnconfigure((0, 1, 2), weight=1)

        for idx, (val, text) in enumerate([("admin", "Admin"), ("teacher", "Teacher"), ("student", "Student")]):
            rb = ctk.CTkRadioButton(
                role_frame, text=text, variable=self.role_var, value=val,
                font=ctk.CTkFont(size=12), radiobutton_height=18, radiobutton_width=18
            )
            rb.grid(row=0, column=idx, padx=5)

        # Login Button
        login_btn = ctk.CTkButton(
            card, text="Sign In", height=42, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#3B82F6", hover_color="#2563EB",
            command=self._login
        )
        login_btn.grid(row=8, column=0, padx=40, pady=(0, 15), sticky="ew")

        # Status label
        self.status_label = ctk.CTkLabel(
            card, text="", font=ctk.CTkFont(size=12),
            text_color="#EF4444"
        )
        self.status_label.grid(row=9, column=0, padx=40, pady=(0, 10))

        # Default credentials info
        info = ctk.CTkLabel(
            card, text="Default Admin: admin / admin123",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        info.grid(row=10, column=0, padx=40, pady=(0, 20))

        # Bind Enter key
        self.password_entry.bind("<Return>", lambda e: self._login())
        self.username_entry.bind("<Return>", lambda e: self.password_entry.focus())

    def _login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        role = self.role_var.get()

        if not username or not password:
            self.status_label.configure(text="Please enter username and password")
            return

        user = self.auth.login(username, password)

        if user:
            if user["role"] != role:
                self.status_label.configure(text=f"This account is not a {role} account")
                self.auth.logout()
                return
            self.on_login_success(user)
        else:
            self.status_label.configure(text="Invalid username or password")

    def clear(self):
        self.username_entry.delete(0, "end")
        self.password_entry.delete(0, "end")
        self.status_label.configure(text="")
