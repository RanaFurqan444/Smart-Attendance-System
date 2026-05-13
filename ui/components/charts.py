"""
Chart components for analytics dashboard using matplotlib.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk


class AttendanceChart(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = None
        self.figure = None

    def _clear(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.figure:
            plt.close(self.figure)
            self.figure = None

    def plot_pie_chart(self, present, absent, late, title="Attendance Distribution"):
        self._clear()
        self.figure, ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")

        labels = []
        sizes = []
        colors = []

        if present > 0:
            labels.append(f"Present ({present})")
            sizes.append(present)
            colors.append("#22C55E")
        if late > 0:
            labels.append(f"Late ({late})")
            sizes.append(late)
            colors.append("#F59E0B")
        if absent > 0:
            labels.append(f"Absent ({absent})")
            sizes.append(absent)
            colors.append("#EF4444")

        if not sizes:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    color="white", fontsize=14)
        else:
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"color": "white", "fontsize": 9}
            )
            for text in autotexts:
                text.set_fontsize(10)
                text.set_fontweight("bold")

        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_bar_chart(self, labels, values, title="Attendance Trend", color="#3B82F6"):
        self._clear()
        self.figure, ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")

        if labels and values:
            bars = ax.bar(range(len(labels)), values, color=color, alpha=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=8)
            ax.set_ylabel("Attendance %", color="white", fontsize=10)
            ax.tick_params(axis="y", colors="white")
            ax.spines["bottom"].set_color("gray")
            ax.spines["left"].set_color("gray")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    color="white", fontsize=14)

        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_line_chart(self, dates, values, title="Attendance Trend"):
        self._clear()
        self.figure, ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.figure.patch.set_facecolor("#2B2B2B")
        ax.set_facecolor("#2B2B2B")

        if dates and values:
            ax.plot(range(len(dates)), values, color="#3B82F6", linewidth=2, marker="o",
                    markersize=4, markerfacecolor="#60A5FA")
            ax.fill_between(range(len(dates)), values, alpha=0.1, color="#3B82F6")
            ax.set_xticks(range(0, len(dates), max(1, len(dates) // 7)))
            display_dates = [dates[i] for i in range(0, len(dates), max(1, len(dates) // 7))]
            ax.set_xticklabels(display_dates, rotation=45, ha="right", color="white", fontsize=8)
            ax.set_ylabel("Attendance %", color="white", fontsize=10)
            ax.set_ylim(0, 105)
            ax.tick_params(axis="y", colors="white")
            ax.spines["bottom"].set_color("gray")
            ax.spines["left"].set_color("gray")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axhline(y=75, color="#F59E0B", linestyle="--", alpha=0.5, label="75% threshold")
            ax.legend(facecolor="#2B2B2B", edgecolor="gray", labelcolor="white")
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    color="white", fontsize=14)

        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
