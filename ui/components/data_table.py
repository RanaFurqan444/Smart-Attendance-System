"""
Reusable data table component using CustomTkinter.
"""

import customtkinter as ctk
from tkinter import ttk
import tkinter as tk


class DataTable(ctk.CTkFrame):
    def __init__(self, parent, columns, data=None, on_select=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.columns = columns
        self.on_select = on_select
        self.data = data or []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Treeview",
            background="#2B2B2B",
            foreground="white",
            fieldbackground="#2B2B2B",
            borderwidth=0,
            rowheight=35,
            font=("Segoe UI", 11)
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#1F1F1F",
            foreground="#3B82F6",
            font=("Segoe UI", 12, "bold"),
            borderwidth=0
        )
        style.map(
            "Custom.Treeview",
            background=[("selected", "#3B82F6")],
            foreground=[("selected", "white")]
        )

        # Treeview
        col_ids = [c["key"] for c in columns]
        self.tree = ttk.Treeview(
            self, columns=col_ids, show="headings",
            style="Custom.Treeview", selectmode="browse"
        )

        for col in columns:
            self.tree.heading(col["key"], text=col["label"])
            self.tree.column(
                col["key"],
                width=col.get("width", 120),
                minwidth=col.get("min_width", 80),
                anchor=col.get("anchor", "w")
            )

        # Scrollbar
        scrollbar = ctk.CTkScrollbar(self, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        if on_select:
            self.tree.bind("<<TreeviewSelect>>", self._on_select)

        if data:
            self.load_data(data)

    def load_data(self, data):
        self.data = data
        self.tree.delete(*self.tree.get_children())
        for row in data:
            values = [row.get(col["key"], "") for col in self.columns]
            self.tree.insert("", "end", values=values)

    def _on_select(self, event):
        selected = self.tree.selection()
        if selected and self.on_select:
            item = self.tree.item(selected[0])
            idx = self.tree.index(selected[0])
            if idx < len(self.data):
                self.on_select(self.data[idx])

    def get_selected(self):
        selected = self.tree.selection()
        if selected:
            idx = self.tree.index(selected[0])
            if idx < len(self.data):
                return self.data[idx]
        return None

    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.data = []

    def refresh(self, data):
        self.load_data(data)
