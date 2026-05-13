"""
Sidebar navigation component for the dashboard.
"""

import customtkinter as ctk


class Sidebar(ctk.CTkFrame):
    def __init__(self, parent, menu_items, on_select, user_info=None, **kwargs):
        super().__init__(parent, width=220, corner_radius=0, **kwargs)
        self.on_select = on_select
        self.buttons = {}
        self.active_button = None

        self.grid_rowconfigure(len(menu_items) + 3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # App title
        title_label = ctk.CTkLabel(
            self, text="Smart\nAttendance",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#3B82F6"
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        # User info
        if user_info:
            role_label = ctk.CTkLabel(
                self, text=f"{user_info.get('role', '').title()} Panel",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            role_label.grid(row=1, column=0, padx=20, pady=(0, 5))

            name_label = ctk.CTkLabel(
                self, text=user_info.get("full_name", ""),
                font=ctk.CTkFont(size=13, weight="bold"),
            )
            name_label.grid(row=2, column=0, padx=20, pady=(0, 20))

        separator = ctk.CTkFrame(self, height=2, fg_color="gray30")
        separator.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Menu buttons
        for idx, (key, label) in enumerate(menu_items):
            btn = ctk.CTkButton(
                self, text=f"  {label}", anchor="w",
                font=ctk.CTkFont(size=14),
                height=40, corner_radius=8,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray75", "gray30"),
                command=lambda k=key: self._on_click(k)
            )
            btn.grid(row=idx + 4, column=0, padx=10, pady=2, sticky="ew")
            self.buttons[key] = btn

        # Logout button at bottom
        logout_btn = ctk.CTkButton(
            self, text="  Logout", anchor="w",
            font=ctk.CTkFont(size=14),
            height=40, corner_radius=8,
            fg_color="transparent",
            text_color="#EF4444",
            hover_color=("gray75", "gray30"),
            command=lambda: self._on_click("logout")
        )
        logout_btn.grid(row=len(menu_items) + 5, column=0, padx=10, pady=(5, 20), sticky="sew")

        # Set first item as active
        if menu_items:
            self._set_active(menu_items[0][0])

    def _on_click(self, key):
        self._set_active(key)
        self.on_select(key)

    def _set_active(self, key):
        if key == "logout":
            return
        for k, btn in self.buttons.items():
            if k == key:
                btn.configure(fg_color=("gray75", "gray30"), text_color="#3B82F6")
            else:
                btn.configure(fg_color="transparent", text_color=("gray10", "gray90"))
        self.active_button = key
