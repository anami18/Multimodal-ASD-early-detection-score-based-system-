"""
ASD Screening GUI Application
==============================
Tkinter-based STAT digitization tool.
Integrates: Gaze Tracking + Speech Analysis + MobileNetV2 Vision

Screens:
1. Welcome / Patient Info
2. Gaze Tracking Test (dot-following)
3. Speech Screening (4 prompts)
4. Results / Risk Report
"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.gaze_tracker import GazeTracker, DotAnimator
from modules.speech_analyzer import SpeechAnalyzer
from models.mobilenet_asd import VisualProcessor, ASDFusionClassifier


# ─── Color Palette ───────────────────────────────────────────────────────────
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
BG_PANEL = "#334155"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
ACCENT_YELLOW = "#f59e0b"
ACCENT_PURPLE = "#a855f7"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"
TEXT_MUTED = "#475569"
BORDER_COLOR = "#1e3a5f"

# Risk level colors
RISK_LOW = "#22c55e"
RISK_MODERATE = "#f59e0b"
RISK_ELEVATED = "#f97316"
RISK_HIGH = "#ef4444"


class ASDScreeningApp:
    """Main application window."""

    WINDOW_TITLE = "ASD Early Screening Tool — STAT Digitization"
    WINDOW_SIZE = "1280x800"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # Initialize components
        self.gaze_tracker = GazeTracker(dot_follow_threshold=0.30)
        self.speech_analyzer = SpeechAnalyzer()
        self.fusion_classifier = ASDFusionClassifier()

        # Try to initialize visual processor (may fail if torch not installed)
        try:
            self.visual_processor = VisualProcessor()
            self.visual_available = True
        except Exception as e:
            print(f"Visual processor init failed: {e}")
            self.visual_processor = None
            self.visual_available = False

        # Session state
        self.session = {
            "patient_name": "",
            "patient_age": "",
            "examiner": "",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gaze_result": None,
            "speech_result": None,
            "final_score": None,
            "risk_category": None,
        }

        # Webcam
        self.cap = None
        self._cam_thread = None
        self._cam_running = False

        # Screen manager
        self.current_screen = None
        self._screens = {}

        # Setup UI
        self._setup_fonts()
        self._setup_styles()
        self._build_layout()
        self._show_screen("welcome")

        # Cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_fonts(self):
        self.fonts = {
            "title": tkfont.Font(family="Helvetica", size=22, weight="bold"),
            "heading": tkfont.Font(family="Helvetica", size=16, weight="bold"),
            "subheading": tkfont.Font(family="Helvetica", size=13, weight="bold"),
            "body": tkfont.Font(family="Helvetica", size=11),
            "small": tkfont.Font(family="Helvetica", size=9),
            "mono": tkfont.Font(family="Courier", size=10),
            "score": tkfont.Font(family="Helvetica", size=48, weight="bold"),
            "button": tkfont.Font(family="Helvetica", size=12, weight="bold"),
        }

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

    def _build_layout(self):
        """Build main layout: header + content area."""
        # Header
        self.header = tk.Frame(self.root, bg=BG_CARD, height=60)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        tk.Label(
            self.header,
            text="⚕  ASD Early Screening Tool",
            font=self.fonts["heading"],
            bg=BG_CARD,
            fg=TEXT_PRIMARY
        ).pack(side="left", padx=20, pady=15)

        self.header_status = tk.Label(
            self.header,
            text="STAT Digitization v1.0",
            font=self.fonts["small"],
            bg=BG_CARD,
            fg=TEXT_SECONDARY
        )
        self.header_status.pack(side="right", padx=20)

        # Progress bar
        self.progress_frame = tk.Frame(self.root, bg=ACCENT_BLUE, height=3)
        self.progress_frame.pack(fill="x")

        # Main content
        self.content = tk.Frame(self.root, bg=BG_DARK)
        self.content.pack(fill="both", expand=True)

    def _show_screen(self, screen_name: str):
        """Switch to a different screen."""
        # Clear content
        for widget in self.content.winfo_children():
            widget.destroy()

        self.current_screen = screen_name

        if screen_name == "welcome":
            WelcomeScreen(self.content, self).pack(fill="both", expand=True)
        elif screen_name == "gaze_test":
            GazeTestScreen(self.content, self).pack(fill="both", expand=True)
        elif screen_name == "speech_test":
            SpeechTestScreen(self.content, self).pack(fill="both", expand=True)
        elif screen_name == "results":
            ResultsScreen(self.content, self).pack(fill="both", expand=True)

    def start_webcam(self):
        """Start webcam capture."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error",
                    "Could not open webcam.\nPlease check camera connection.")
                return False
        return True

    def stop_webcam(self):
        """Release webcam."""
        self._cam_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def _on_close(self):
        """Cleanup on window close."""
        self.stop_webcam()
        if hasattr(self, 'gaze_tracker') and self.gaze_tracker.face_mesh:
            self.gaze_tracker.face_mesh.close()
        self.root.destroy()


def make_button(parent, text, command, color=ACCENT_BLUE, fg=TEXT_PRIMARY,
                width=None, font=None):
    """Create a styled button."""
    kwargs = dict(
        text=text,
        command=command,
        bg=color,
        fg=fg,
        activebackground=color,
        activeforeground=fg,
        relief="flat",
        cursor="hand2",
        font=font or tkfont.Font(family="Helvetica", size=11, weight="bold"),
        padx=20,
        pady=10,
        bd=0
    )
    if width:
        kwargs["width"] = width
    btn = tk.Button(parent, **kwargs)
    # Hover effect
    def on_enter(e): btn.config(bg=_lighten(color))
    def on_leave(e): btn.config(bg=color)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def _lighten(hex_color: str, amount: float = 0.15) -> str:
    """Lighten a hex color."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


class WelcomeScreen(tk.Frame):
    """Welcome screen with patient info form."""

    def __init__(self, parent, app: ASDScreeningApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._build()

    def _build(self):
        # Center container
        center = tk.Frame(self, bg=BG_DARK)
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Logo/title
        tk.Label(center, text="🧠", font=tkfont.Font(size=50),
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(pady=(0, 10))
        tk.Label(center, text="ASD Early Screening Tool",
                 font=tkfont.Font(family="Helvetica", size=26, weight="bold"),
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack()
        tk.Label(center,
                 text="Digitized STAT — Screening Tool for Autism in Toddlers",
                 font=tkfont.Font(family="Helvetica", size=13),
                 bg=BG_DARK, fg=TEXT_SECONDARY).pack(pady=(4, 30))

        # Info card
        card = tk.Frame(center, bg=BG_CARD, padx=40, pady=30)
        card.pack(fill="x")

        tk.Label(card, text="Patient Information",
                 font=tkfont.Font(family="Helvetica", size=14, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 20))

        # Form fields
        self._fields = {}
        fields = [
            ("Child's Name:", "name"),
            ("Age (months):", "age"),
            ("Examiner Name:", "examiner"),
        ]

        for label_text, key in fields:
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=label_text, width=18, anchor="w",
                     font=tkfont.Font(family="Helvetica", size=11),
                     bg=BG_CARD, fg=TEXT_SECONDARY).pack(side="left")
            entry = tk.Entry(
                row,
                font=tkfont.Font(family="Helvetica", size=11),
                bg=BG_PANEL, fg=TEXT_PRIMARY,
                insertbackground=TEXT_PRIMARY,
                relief="flat", bd=5, width=30
            )
            entry.pack(side="left", padx=(10, 0))
            self._fields[key] = entry

        # Disclaimer
        disclaimer = tk.Frame(center, bg="#1a1a2e", padx=15, pady=10)
        disclaimer.pack(fill="x", pady=(20, 0))
        tk.Label(
            disclaimer,
            text="⚠️  CLINICAL DISCLAIMER: This tool is for SCREENING purposes only and does "
                 "not constitute a clinical diagnosis. Results should be reviewed by a qualified "
                 "clinician. Consult a pediatric specialist for formal ASD evaluation.",
            font=tkfont.Font(family="Helvetica", size=9),
            bg="#1a1a2e", fg=ACCENT_YELLOW,
            wraplength=500, justify="left"
        ).pack()

        # Start button
        make_button(center, "▶  Begin Screening", self._start,
                    color=ACCENT_GREEN, width=25).pack(pady=(25, 0))

    def _start(self):
        self.app.session["patient_name"] = self._fields["name"].get()
        self.app.session["patient_age"] = self._fields["age"].get()
        self.app.session["examiner"] = self._fields["examiner"].get()

        if not self.app.session["patient_name"]:
            messagebox.showwarning("Missing Info", "Please enter the child's name.")
            return

        self.app._show_screen("gaze_test")


class GazeTestScreen(tk.Frame):
    """
    Gaze tracking test screen.
    Shows webcam feed with animated dot for the child to follow.
    """

    TEST_DURATION = 45  # seconds
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    def __init__(self, parent, app: ASDScreeningApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._running = False
        self._test_started = False
        self._start_time = None
        self._dot_animator = None
        self._frames_collected = 0

        if not app.start_webcam():
            tk.Label(self, text="⚠ Camera not available",
                     fg=ACCENT_RED, bg=BG_DARK,
                     font=tkfont.Font(size=16)).pack(expand=True)
            return

        self.app.gaze_tracker.reset()
        self._build()
        self._dot_animator = DotAnimator(self.CAMERA_WIDTH, self.CAMERA_HEIGHT)

    def _build(self):
        # Left: camera feed
        left = tk.Frame(self, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        tk.Label(left, text="Gaze Tracking Test",
                 font=tkfont.Font(family="Helvetica", size=16, weight="bold"),
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        self.camera_label = tk.Label(left, bg="black")
        self.camera_label.pack()

        self.gaze_status = tk.Label(left, text="", bg=BG_DARK,
                                     fg=ACCENT_GREEN,
                                     font=tkfont.Font(family="Helvetica", size=12))
        self.gaze_status.pack(pady=5)

        # Timer
        self.timer_label = tk.Label(left, text="",
                                     font=tkfont.Font(family="Helvetica", size=14),
                                     bg=BG_DARK, fg=TEXT_SECONDARY)
        self.timer_label.pack()

        # Right: instructions + live stats
        right = tk.Frame(self, bg=BG_CARD, width=320, padx=20, pady=20)
        right.pack(side="right", fill="y", padx=(0, 20), pady=20)
        right.pack_propagate(False)

        tk.Label(right, text="📋  Instructions",
                 font=tkfont.Font(family="Helvetica", size=14, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 15))

        instructions = (
            "1. Position the child ~50cm from the screen\n\n"
            "2. Ensure the child's face is clearly visible\n\n"
            "3. Click START TEST to begin\n\n"
            "4. Guide the child to watch the moving blue dot\n\n"
            "5. Test runs for 45 seconds\n\n"
            "6. Results are computed automatically"
        )
        tk.Label(right, text=instructions,
                 font=tkfont.Font(family="Helvetica", size=10),
                 bg=BG_CARD, fg=TEXT_SECONDARY,
                 justify="left", wraplength=280).pack(anchor="w")

        # Live metrics
        tk.Label(right, text="━" * 30, bg=BG_CARD, fg=BG_PANEL).pack(pady=10)
        tk.Label(right, text="Live Metrics",
                 font=tkfont.Font(family="Helvetica", size=12, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w")

        self.metric_vars = {}
        metrics = [
            ("Following Ratio", "following"),
            ("Frames Tracked", "frames"),
            ("Gaze Status", "status"),
        ]
        for label, key in metrics:
            row = tk.Frame(right, bg=BG_CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{label}:", width=18, anchor="w",
                     font=tkfont.Font(size=10), bg=BG_CARD,
                     fg=TEXT_SECONDARY).pack(side="left")
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            tk.Label(row, textvariable=var,
                     font=tkfont.Font(size=10, weight="bold"),
                     bg=BG_CARD, fg=ACCENT_BLUE).pack(side="left")

        # Buttons
        self.start_btn = make_button(right, "▶  Start Test", self._start_test,
                                      color=ACCENT_GREEN)
        self.start_btn.pack(pady=(20, 8), fill="x")

        self.skip_btn = make_button(right, "⏭  Skip (No Camera)", self._skip,
                                     color=BG_PANEL)
        self.skip_btn.pack(fill="x")

        # Start camera preview
        self._running = True
        self._update_frame()

    def _start_test(self):
        if not self._test_started:
            self._test_started = True
            self._start_time = time.time()
            self.start_btn.config(state="disabled")
            self._dot_animator.reset()

    def _update_frame(self):
        if not self._running:
            return

        if self.app.cap and self.app.cap.isOpened():
            ret, frame = self.app.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror
                frame_resized = cv2.resize(frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))

                # Draw dot
                if self._dot_animator:
                    dot_pos = self._dot_animator.get_dot_position()
                    frame_with_dot = self._dot_animator.draw_dot(frame_resized.copy())
                else:
                    frame_with_dot = frame_resized.copy()
                    dot_pos = (0.5, 0.5)

                # Process gaze if test running
                if self._test_started and self._start_time:
                    elapsed = time.time() - self._start_time

                    if elapsed < self.TEST_DURATION:
                        frame_with_dot, gaze_frame = self.app.gaze_tracker.process_frame(
                            frame_with_dot, dot_pos, timestamp=elapsed
                        )
                        self._frames_collected += 1

                        # Update live metrics
                        total = len(self.app.gaze_tracker.frames)
                        following = sum(1 for f in self.app.gaze_tracker.frames if f.following)
                        ratio = following / total if total > 0 else 0

                        self.metric_vars["following"].set(f"{ratio:.1%}")
                        self.metric_vars["frames"].set(str(total))
                        self.metric_vars["status"].set(
                            "✓ Face Detected" if gaze_frame else "✗ No Face"
                        )

                        # Timer
                        remaining = int(self.TEST_DURATION - elapsed)
                        self.timer_label.config(
                            text=f"⏱  {remaining}s remaining",
                            fg=ACCENT_YELLOW if remaining < 10 else TEXT_SECONDARY
                        )

                    else:
                        # Test complete
                        self._finish_test()

                # Display frame
                rgb = cv2.cvtColor(frame_with_dot, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

        if self._running:
            self.camera_label.after(33, self._update_frame)  # ~30fps

    def _finish_test(self):
        """Test complete - compute results."""
        self._test_started = False
        self.timer_label.config(text="✓  Test Complete!", fg=ACCENT_GREEN)

        result = self.app.gaze_tracker.compute_session_results()
        self.app.session["gaze_result"] = result

        self._running = False
        self.app.stop_webcam()

        # Show brief summary then advance
        self.after(1500, lambda: self.app._show_screen("speech_test"))

    def _skip(self):
        """Skip gaze test."""
        self._running = False
        self.app.stop_webcam()
        self.app._show_screen("speech_test")

    def destroy(self):
        self._running = False
        super().destroy()


class SpeechTestScreen(tk.Frame):
    """
    Speech analysis test screen.
    Presents STAT-inspired speech prompts and records responses.
    """

    RECORD_DURATION = 8  # seconds per prompt

    def __init__(self, parent, app: ASDScreeningApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self.app.speech_analyzer = SpeechAnalyzer()  # Fresh instance
        self._current_prompt = 0
        self._recording = False
        self._recording_thread = None
        self._timer_id = None
        self._build()

    def _build(self):
        # Title
        tk.Label(self, text="Speech Screening Test",
                 font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(pady=(20, 5))
        tk.Label(self,
                 text="The child will be prompted to speak. Record each response.",
                 font=tkfont.Font(family="Helvetica", size=11),
                 bg=BG_DARK, fg=TEXT_SECONDARY).pack(pady=(0, 20))

        # Progress indicators
        prog_frame = tk.Frame(self, bg=BG_DARK)
        prog_frame.pack()
        self._prompt_indicators = []
        for i in range(len(SpeechAnalyzer.SCREENING_PROMPTS)):
            ind = tk.Label(prog_frame, text=f"  {i+1}  ",
                           font=tkfont.Font(size=11, weight="bold"),
                           bg=TEXT_MUTED, fg=TEXT_SECONDARY, padx=5)
            ind.pack(side="left", padx=3)
            self._prompt_indicators.append(ind)

        # Main card
        card = tk.Frame(self, bg=BG_CARD, padx=30, pady=25)
        card.pack(fill="x", padx=100, pady=20)

        self.prompt_num_label = tk.Label(
            card, text="Prompt 1 of 4",
            font=tkfont.Font(family="Helvetica", size=11),
            bg=BG_CARD, fg=TEXT_SECONDARY
        )
        self.prompt_num_label.pack(anchor="w")

        self.prompt_label = tk.Label(
            card,
            text=SpeechAnalyzer.SCREENING_PROMPTS[0]["instruction"],
            font=tkfont.Font(family="Helvetica", size=14),
            bg=BG_CARD, fg=TEXT_PRIMARY,
            justify="center", wraplength=600
        )
        self.prompt_label.pack(pady=20)

        # Recording status
        self.rec_status = tk.Label(card, text="",
                                    font=tkfont.Font(size=18),
                                    bg=BG_CARD, fg=ACCENT_RED)
        self.rec_status.pack(pady=5)

        self.rec_timer = tk.Label(card, text="",
                                   font=tkfont.Font(size=12),
                                   bg=BG_CARD, fg=TEXT_SECONDARY)
        self.rec_timer.pack()

        # Level meter (visual RMS indicator)
        self.level_canvas = tk.Canvas(card, width=400, height=20,
                                       bg=BG_PANEL, highlightthickness=0)
        self.level_canvas.pack(pady=10)

        # Buttons row
        btn_row = tk.Frame(card, bg=BG_CARD)
        btn_row.pack(pady=(15, 0))

        self.rec_btn = make_button(btn_row, "🎙  START Recording",
                                    self._toggle_recording, color=ACCENT_RED)
        self.rec_btn.pack(side="left", padx=10)

        self.skip_prompt_btn = make_button(btn_row, "Skip Prompt",
                                            self._next_prompt, color=BG_PANEL)
        self.skip_prompt_btn.pack(side="left", padx=10)

        # Results summary (will be filled after all prompts)
        self.summary_frame = tk.Frame(self, bg=BG_DARK)
        self.summary_frame.pack(fill="x", padx=100)
        self.completed_labels = []

        # Navigation
        nav = tk.Frame(self, bg=BG_DARK)
        nav.pack(side="bottom", pady=20)
        make_button(nav, "← Back", lambda: self.app._show_screen("gaze_test"),
                    color=BG_PANEL).pack(side="left", padx=10)
        self.finish_btn = make_button(nav, "Analyze & See Results →",
                                       self._finish, color=ACCENT_BLUE)
        self.finish_btn.pack(side="left", padx=10)
        self.finish_btn.config(state="disabled")

        self._update_prompt_display()

    def _update_prompt_display(self):
        """Update UI for current prompt."""
        prompts = SpeechAnalyzer.SCREENING_PROMPTS
        idx = self._current_prompt

        if idx < len(prompts):
            p = prompts[idx]
            self.prompt_num_label.config(text=f"Prompt {idx+1} of {len(prompts)}")
            self.prompt_label.config(text=p["instruction"])

            # Update indicators
            for i, ind in enumerate(self._prompt_indicators):
                if i < idx:
                    ind.config(bg=ACCENT_GREEN, fg="white")
                elif i == idx:
                    ind.config(bg=ACCENT_BLUE, fg="white")
                else:
                    ind.config(bg=TEXT_MUTED, fg=TEXT_SECONDARY)

    def _toggle_recording(self):
        if self._recording:
            return  # Already recording

        self._recording = True
        self.rec_btn.config(state="disabled", text="⏺  Recording...")
        self.rec_status.config(text="🔴  RECORDING")
        self._rec_start = time.time()

        duration = SpeechAnalyzer.SCREENING_PROMPTS[self._current_prompt]["expected_duration"]

        def on_complete(audio):
            self._recording = False
            self.after(0, self._on_recording_complete, audio)

        self._recording_thread = self.app.speech_analyzer.record_with_callback(
            duration=duration,
            on_complete=on_complete
        )

        self._update_rec_timer()

    def _update_rec_timer(self):
        if self._recording:
            elapsed = time.time() - self._rec_start
            duration = SpeechAnalyzer.SCREENING_PROMPTS[self._current_prompt]["expected_duration"]
            remaining = max(0, duration - elapsed)

            self.rec_timer.config(text=f"⏱  {remaining:.1f}s remaining")

            # Level meter (fake but gives visual feedback)
            level_width = int((1 - remaining/duration) * 400)
            self.level_canvas.delete("all")
            color = ACCENT_GREEN if remaining > 2 else ACCENT_RED
            self.level_canvas.create_rectangle(0, 0, level_width, 20, fill=color)

            self._timer_id = self.after(100, self._update_rec_timer)

    def _on_recording_complete(self, audio):
        """Called when recording finishes."""
        self.rec_status.config(text="✓  Recorded", fg=ACCENT_GREEN)
        self.rec_timer.config(text="")
        self.level_canvas.delete("all")
        self.level_canvas.create_rectangle(0, 0, 400, 20, fill=ACCENT_GREEN)

        # Show completion for this prompt
        label = tk.Label(
            self.summary_frame,
            text=f"✓ Prompt {self._current_prompt+1} recorded ({len(audio)/22050:.1f}s)",
            font=tkfont.Font(size=10), bg=BG_DARK, fg=ACCENT_GREEN
        )
        label.pack(anchor="w")
        self.completed_labels.append(label)

        self.after(1000, self._next_prompt)

    def _next_prompt(self):
        """Move to next prompt."""
        self._current_prompt += 1
        prompts = SpeechAnalyzer.SCREENING_PROMPTS

        if self._current_prompt >= len(prompts):
            # All done
            self.finish_btn.config(state="normal")
            self.rec_btn.config(state="disabled", text="All Done!")
            self.prompt_label.config(
                text="✓  All speech prompts completed.\nClick 'Analyze & See Results' to continue.",
                fg=ACCENT_GREEN
            )
            return

        self.rec_btn.config(state="normal", text="🎙  START Recording")
        self.rec_status.config(text="", fg=ACCENT_RED)
        self._update_prompt_display()

    def _finish(self):
        """Analyze speech and go to results."""
        speech_result = self.app.speech_analyzer.analyze_all_recordings()
        self.app.session["speech_result"] = speech_result
        self.app._show_screen("results")


class ResultsScreen(tk.Frame):
    """
    Final results screen showing ASD risk assessment.
    """

    def __init__(self, parent, app: ASDScreeningApp):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self._compute_results()
        self._build()

    def _compute_results(self):
        """Fuse all modalities into final score."""
        gaze_result = self.app.session.get("gaze_result")
        speech_result = self.app.session.get("speech_result")

        gaze_risk = gaze_result.asd_risk_score if gaze_result else 50.0
        speech_risk = speech_result.asd_risk_score if speech_result else 50.0

        score, category, details = self.app.fusion_classifier.predict(
            gaze_risk=gaze_risk,
            speech_risk=speech_risk
        )

        self.app.session["final_score"] = score
        self.app.session["risk_category"] = category
        self.app.session["details"] = details

    def _build(self):
        # Scrollable canvas
        canvas = tk.Canvas(self, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG_DARK)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        score = self.app.session["final_score"]
        category = self.app.session["risk_category"]
        details = self.app.session.get("details", {})

        # Risk color
        if score < 30:
            risk_color = RISK_LOW
        elif score < 55:
            risk_color = RISK_MODERATE
        elif score < 75:
            risk_color = RISK_ELEVATED
        else:
            risk_color = RISK_HIGH

        # ─── Header ───────────────────────────────────────────────────
        header = tk.Frame(inner, bg=BG_CARD, padx=30, pady=20)
        header.pack(fill="x", padx=20, pady=(20, 0))

        tk.Label(header, text="Screening Results",
                 font=tkfont.Font(family="Helvetica", size=20, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(side="left")

        tk.Label(header,
                 text=f"Patient: {self.app.session['patient_name']} | "
                      f"Age: {self.app.session['patient_age']} months | "
                      f"Date: {self.app.session['date']}",
                 font=tkfont.Font(family="Helvetica", size=10),
                 bg=BG_CARD, fg=TEXT_SECONDARY).pack(side="right")

        # ─── Score Card ────────────────────────────────────────────────
        score_row = tk.Frame(inner, bg=BG_DARK)
        score_row.pack(fill="x", padx=20, pady=10)

        # Big score
        score_card = tk.Frame(score_row, bg=BG_CARD, padx=30, pady=20)
        score_card.pack(side="left", fill="y")

        tk.Label(score_card, text="Overall Risk Score",
                 font=tkfont.Font(size=11), bg=BG_CARD,
                 fg=TEXT_SECONDARY).pack()

        tk.Label(score_card, text=f"{score:.0f}",
                 font=tkfont.Font(family="Helvetica", size=64, weight="bold"),
                 bg=BG_CARD, fg=risk_color).pack()

        tk.Label(score_card, text="/ 100",
                 font=tkfont.Font(size=14), bg=BG_CARD,
                 fg=TEXT_MUTED).pack()

        # Risk band
        band = tk.Frame(score_card, bg=risk_color, padx=15, pady=6)
        band.pack(pady=(10, 0))
        tk.Label(band, text=f"  {category}  ",
                 font=tkfont.Font(family="Helvetica", size=13, weight="bold"),
                 bg=risk_color, fg="white").pack()

        # Modality breakdown
        breakdown = tk.Frame(score_row, bg=BG_CARD, padx=25, pady=20)
        breakdown.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(breakdown, text="Modality Breakdown",
                 font=tkfont.Font(size=12, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 15))

        modalities = [
            ("👁  Gaze Tracking", details.get("gaze_risk", 50.0), ACCENT_BLUE),
            ("🎙  Speech Analysis", details.get("speech_risk", 50.0), ACCENT_PURPLE),
            ("📷  Visual Features", details.get("visual_risk", 50.0), ACCENT_YELLOW),
        ]

        for name, val, color in modalities:
            row = tk.Frame(breakdown, bg=BG_CARD)
            row.pack(fill="x", pady=5)
            tk.Label(row, text=name, width=22, anchor="w",
                     font=tkfont.Font(size=10), bg=BG_CARD,
                     fg=TEXT_SECONDARY).pack(side="left")

            # Progress bar
            bar_bg = tk.Frame(row, bg=BG_PANEL, height=12, width=200)
            bar_bg.pack(side="left", padx=5)
            bar_bg.pack_propagate(False)

            fill_w = max(4, int(val / 100 * 200))
            bar_fill = tk.Frame(bar_bg, bg=color, height=12, width=fill_w)
            bar_fill.place(x=0, y=0)

            tk.Label(row, text=f"{val:.0f}",
                     font=tkfont.Font(size=10, weight="bold"),
                     bg=BG_CARD, fg=color).pack(side="left", padx=5)

        # ─── Gaze Details ──────────────────────────────────────────────
        gaze_result = self.app.session.get("gaze_result")
        if gaze_result:
            gaze_section = self._make_detail_section(
                inner, "👁  Gaze Tracking Details"
            )
            items = [
                ("Following Ratio", f"{gaze_result.following_ratio:.1%}",
                 "How often gaze followed the dot"),
                ("Avg. Deviation", f"{gaze_result.avg_gaze_deviation:.2f}",
                 "Mean gaze-dot distance (normalized)"),
                ("Saccade Smoothness", f"{gaze_result.saccade_smoothness:.2f}",
                 "Smooth tracking quality (0-1)"),
                ("Fixations Detected", str(gaze_result.fixation_count),
                 "Number of stable gaze fixations"),
                ("Mutual Gaze Ratio", f"{gaze_result.mutual_gaze_ratio:.1%}",
                 "Time looking toward camera"),
                ("Blink Rate", f"{gaze_result.blink_rate:.0f}/min",
                 "Blinks per minute"),
                ("Frames Tracked", str(gaze_result.frames_tracked),
                 "Total gaze frames analyzed"),
            ]
            self._add_detail_items(gaze_section, items)

        # ─── Speech Details ────────────────────────────────────────────
        speech_result = self.app.session.get("speech_result")
        if speech_result and speech_result.features.valid:
            f = speech_result.features
            speech_section = self._make_detail_section(
                inner, "🎙  Speech Analysis Details"
            )
            items = [
                ("Pitch Std Dev", f"{f.pitch_std:.1f} Hz",
                 "F0 variation (monotone=low)"),
                ("Pitch Range", f"{f.pitch_range:.1f} Hz",
                 "Max-min F0 range"),
                ("Pause Ratio", f"{f.pause_ratio:.1%}",
                 "Fraction of time silent"),
                ("Speech Rate", f"{f.speech_rate:.1f} syl/s",
                 "Estimated syllables/second"),
                ("Rhythm Regularity", f"{f.rhythm_regularity:.2f}",
                 "Speech rhythm consistency"),
                ("Repetition Score", f"{f.repetition_score:.2f}",
                 "Echolalia proxy (1=highly repetitive)"),
                ("Jitter", f"{f.jitter:.3f}",
                 "F0 cycle-to-cycle variation"),
                ("Prompts Completed", str(speech_result.prompts_completed),
                 "Number of speech samples recorded"),
            ]
            self._add_detail_items(speech_section, items)

            # Risk factors
            if speech_result.risk_factors:
                tk.Label(speech_section.master if hasattr(speech_section, 'master')
                         else inner,
                         text="Speech Risk Factors Identified:",
                         font=tkfont.Font(size=10, weight="bold"),
                         bg=BG_DARK, fg=ACCENT_YELLOW).pack(
                             anchor="w", padx=20)
                for rf in speech_result.risk_factors:
                    tk.Label(inner, text=f"  ⚠  {rf}",
                             font=tkfont.Font(size=10),
                             bg=BG_DARK, fg=ACCENT_YELLOW).pack(
                                 anchor="w", padx=20)

        # ─── Recommendations ──────────────────────────────────────────
        rec_section = tk.Frame(inner, bg=BG_CARD, padx=25, pady=20)
        rec_section.pack(fill="x", padx=20, pady=(10, 0))

        tk.Label(rec_section, text="📋  Clinical Recommendations",
                 font=tkfont.Font(size=13, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        recommendations = self._get_recommendations(score)
        for rec in recommendations:
            tk.Label(rec_section, text=f"  •  {rec}",
                     font=tkfont.Font(size=10),
                     bg=BG_CARD, fg=TEXT_SECONDARY,
                     justify="left", wraplength=800).pack(anchor="w", pady=2)

        # ─── Disclaimer ────────────────────────────────────────────────
        disc = tk.Frame(inner, bg="#1a0a0a", padx=15, pady=10)
        disc.pack(fill="x", padx=20, pady=10)
        tk.Label(
            disc,
            text="⚠️  IMPORTANT: This screening score is NOT a clinical diagnosis. "
                 "It is an automated risk indicator to supplement—not replace—clinical judgment. "
                 "A score indicating elevated or high risk requires referral to a qualified "
                 "developmental pediatrician or autism specialist for comprehensive evaluation.",
            font=tkfont.Font(size=9),
            bg="#1a0a0a", fg=ACCENT_RED,
            wraplength=900, justify="left"
        ).pack()

        # ─── Actions ───────────────────────────────────────────────────
        actions = tk.Frame(inner, bg=BG_DARK)
        actions.pack(pady=15)

        make_button(actions, "🔄  New Screening",
                    lambda: self.app._show_screen("welcome"),
                    color=ACCENT_BLUE).pack(side="left", padx=10)

        make_button(actions, "💾  Save Report",
                    self._save_report, color=ACCENT_GREEN).pack(side="left", padx=10)

    def _make_detail_section(self, parent, title: str) -> tk.Frame:
        section = tk.Frame(parent, bg=BG_CARD, padx=25, pady=15)
        section.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(section, text=title,
                 font=tkfont.Font(size=12, weight="bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))
        return section

    def _add_detail_items(self, parent, items):
        grid = tk.Frame(parent, bg=BG_CARD)
        grid.pack(fill="x")
        for i, (label, value, tooltip) in enumerate(items):
            row = tk.Frame(grid, bg=BG_CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, width=22, anchor="w",
                     font=tkfont.Font(size=10), bg=BG_CARD,
                     fg=TEXT_SECONDARY).pack(side="left")
            tk.Label(row, text=value, width=14,
                     font=tkfont.Font(size=10, weight="bold"),
                     bg=BG_CARD, fg=TEXT_PRIMARY).pack(side="left")
            tk.Label(row, text=tooltip,
                     font=tkfont.Font(size=9),
                     bg=BG_CARD, fg=TEXT_MUTED).pack(side="left", padx=10)

    def _get_recommendations(self, score: float) -> list:
        if score < 30:
            return [
                "Screening results suggest LOW RISK of ASD based on current indicators.",
                "Continue routine developmental monitoring at scheduled well-child visits.",
                "Encourage continued social engagement and language-rich environments.",
                "Repeat screening in 6 months or if developmental concerns arise.",
            ]
        elif score < 55:
            return [
                "Screening results suggest MODERATE RISK — some atypical indicators were detected.",
                "Consider referral to a developmental pediatrician for detailed evaluation.",
                "Parent/caregiver education about ASD signs and early intervention benefits.",
                "Schedule follow-up screening in 3 months.",
                "Review speech and language development milestones with primary care physician.",
            ]
        elif score < 75:
            return [
                "Screening results suggest ELEVATED RISK — multiple atypical indicators detected.",
                "RECOMMENDED: Prompt referral to a developmental specialist or autism clinic.",
                "Begin early intervention services if available (speech therapy, OT evaluation).",
                "Complete formal diagnostic evaluation using ADOS-2 or ADI-R instruments.",
                "Inform parents/caregivers of findings and discuss next steps.",
            ]
        else:
            return [
                "Screening results suggest HIGH RISK — significant atypical indicators across modalities.",
                "URGENT: Refer to autism specialist or developmental pediatrician immediately.",
                "Do NOT delay intervention pending formal diagnosis — begin speech/behavioral therapy.",
                "Administer comprehensive evaluation: ADOS-2, ADI-R, developmental testing.",
                "Coordinate with school system for early intervention programs (age-appropriate).",
                "Provide family support resources and psychoeducation.",
            ]

    def _save_report(self):
        """Save results to JSON file."""
        report = {
            "patient": self.app.session["patient_name"],
            "age_months": self.app.session["patient_age"],
            "examiner": self.app.session["examiner"],
            "date": self.app.session["date"],
            "final_score": self.app.session["final_score"],
            "risk_category": self.app.session["risk_category"],
            "details": self.app.session.get("details", {}),
        }

        os.makedirs("./reports", exist_ok=True)
        filename = f"./reports/screening_{self.app.session['patient_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        messagebox.showinfo("Saved", f"Report saved to:\n{filename}")
