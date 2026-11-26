import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import cv2
from PIL import Image, ImageTk

from trimmer_engine import (
    DetectionResult,
    TrainableTrimModel,
    detect_trim_points,
    get_video_info,
    trim_video_ffmpeg,
)


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


class VRAutoTrimmerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SnipVR.IQ")
        self.geometry("980x680")
        self.minsize(880, 620)
        self.configure(bg="#111827")

        self.app_dir = (
            os.path.dirname(sys.executable)
            if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__))
        )
        bundle_dir = getattr(sys, "_MEIPASS", self.app_dir)
        self.default_model_path = os.path.join(self.app_dir, "default_model.npz")
        bundled_model = os.path.join(bundle_dir, "default_model.npz")
        load_path = self.default_model_path if os.path.isfile(self.default_model_path) else bundled_model

        self.model = TrainableTrimModel()
        if os.path.isfile(load_path):
            try:
                self.model = TrainableTrimModel.load(load_path)
            except Exception:
                self.model = TrainableTrimModel()

        # Train tab state
        self.train_video_path: Optional[str] = None
        self.train_cap: Optional[cv2.VideoCapture] = None
        self.train_fps: float = 30.0
        self.train_duration: float = 0.0
        self.train_total_frames: int = 0
        self.train_current_sec: float = 0.0
        self.keep_start_sec: float = 0.0
        self.keep_end_sec: float = 0.0
        self._train_photo: Optional[ImageTk.PhotoImage] = None

        # Trim & Export tab state
        self.queue_paths: List[str] = []
        self.detections: Dict[str, DetectionResult] = {}
        self.output_dir: str = os.path.join(os.path.expanduser("~"), "Videos", "SnipVR.IQ")
        self._start_thumb: Optional[ImageTk.PhotoImage] = None
        self._end_thumb: Optional[ImageTk.PhotoImage] = None

        self._setup_styles()
        self._build_ui()
        self._refresh_model_status()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background="#111827", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1f2937", foreground="#e5e7eb", padding=(18, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#2563eb")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview", background="#1f2937", foreground="#f3f4f6", fieldbackground="#1f2937", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#374151", foreground="#f9fafb", font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#1f2937", height=52)
        header.pack(fill="x", side="top")
        tk.Label(
            header,
            text="SnipVR.IQ",
            bg="#1f2937",
            fg="#f9fafb",
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left", padx=16, pady=10)

        self.model_status_lbl = tk.Label(
            header,
            text="",
            bg="#1f2937",
            fg="#93c5fd",
            font=("Segoe UI", 9, "bold"),
        )
        self.model_status_lbl.pack(side="right", padx=16)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=10)

        self.trim_tab = tk.Frame(self.notebook, bg="#111827")
        self.train_tab = tk.Frame(self.notebook, bg="#111827")
        self.notebook.add(self.trim_tab, text="  Trim & Export  ")
        self.notebook.add(self.train_tab, text="  Train Model  ")

        self._build_trim_tab()
        self._build_train_tab()

        self.status_var = tk.StringVar(value="Add video(s) and click 'Auto-Trim & Export'.")
        tk.Label(
            self,
            textvariable=self.status_var,
            bg="#0f172a",
            fg="#cbd5e1",
            anchor="w",
            padx=12,
            pady=6,
            font=("Segoe UI", 9),
        ).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # TAB 1: TRIM & EXPORT (SIMPLIFIED)
    # ------------------------------------------------------------------
    def _build_trim_tab(self) -> None:
        top_bar = tk.Frame(self.trim_tab, bg="#111827")
        top_bar.pack(fill="x", pady=(6, 10))

        btn_blue = {"bg": "#2563eb", "fg": "#ffffff", "relief": "flat", "padx": 14, "pady": 7, "font": ("Segoe UI", 9, "bold"), "cursor": "hand2"}
        btn_gray = {"bg": "#374151", "fg": "#f9fafb", "relief": "flat", "padx": 12, "pady": 7, "font": ("Segoe UI", 9), "cursor": "hand2"}
        btn_green = {"bg": "#059669", "fg": "#ffffff", "relief": "flat", "padx": 16, "pady": 7, "font": ("Segoe UI", 10, "bold"), "cursor": "hand2"}

        tk.Button(top_bar, text="+ Add Video(s)", command=self._add_videos, **btn_blue).pack(side="left", padx=(0, 6))
        tk.Button(top_bar, text="+ Add Folder", command=self._add_folder, **btn_gray).pack(side="left", padx=(0, 6))
        tk.Button(top_bar, text="Clear", command=self._clear_queue, **btn_gray).pack(side="left", padx=(0, 14))

        tk.Button(top_bar, text="Auto-Trim & Export All", command=self._trim_all_async, **btn_green).pack(side="right")
        tk.Button(top_bar, text="Preview Cuts", command=self._analyze_all_async, bg="#d97706", fg="#ffffff", relief="flat", padx=12, pady=7, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="right", padx=(0, 8))

        # Export destination bar
        exp_bar = tk.Frame(self.trim_tab, bg="#1f2937", padx=10, pady=8)
        exp_bar.pack(fill="x", pady=(0, 10))

        tk.Label(exp_bar, text="Export Folder:", bg="#1f2937", fg="#e5e7eb", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.out_dir_lbl = tk.Label(exp_bar, text=self.output_dir, bg="#1f2937", fg="#93c5fd", font=("Segoe UI", 9))
        self.out_dir_lbl.pack(side="left", padx=(8, 12))

        tk.Button(exp_bar, text="Change Folder...", command=self._choose_output_dir, bg="#374151", fg="#ffffff", relief="flat", padx=10, pady=3, font=("Segoe UI", 9), cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(exp_bar, text="Open Export Folder", command=self._open_output_dir, bg="#374151", fg="#ffffff", relief="flat", padx=10, pady=3, font=("Segoe UI", 9), cursor="hand2").pack(side="left")
        tk.Button(exp_bar, text="Export Selected As...", command=self._export_selected_as, bg="#1d4ed8", fg="#ffffff", relief="flat", padx=10, pady=3, font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="right")

        # Main area: Queue table + Simple Cut Preview
        body = tk.Frame(self.trim_tab, bg="#111827")
        body.pack(fill="both", expand=True)

        left_col = tk.Frame(body, bg="#111827")
        left_col.pack(side="left", fill="both", expand=True)

        cols = ("file", "duration", "start", "end", "kept", "trimmed", "status")
        self.tree = ttk.Treeview(left_col, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("file", text="Video File")
        self.tree.heading("duration", text="Original")
        self.tree.heading("start", text="Start")
        self.tree.heading("end", text="End")
        self.tree.heading("kept", text="Kept")
        self.tree.heading("trimmed", text="Time Trimmed")
        self.tree.heading("status", text="Status")

        self.tree.column("file", width=220)
        self.tree.column("duration", width=70, anchor="center")
        self.tree.column("start", width=70, anchor="center")
        self.tree.column("end", width=70, anchor="center")
        self.tree.column("kept", width=70, anchor="center")
        self.tree.column("trimmed", width=90, anchor="center")
        self.tree.column("status", width=115, anchor="center")
        self.tree.pack(side="top", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_queue_item)

        self.total_dur_lbl = tk.Label(
            left_col,
            text="Total Original: 00:00.00   |   Trimmed Total Duration: 00:00.00   |   Removed: 00:00.00",
            bg="#1f2937",
            fg="#86efac",
            anchor="w",
            padx=10,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        self.total_dur_lbl.pack(side="bottom", fill="x", pady=(6, 0))

        preview_panel = tk.Frame(body, bg="#1f2937", width=270, padx=10, pady=10)
        preview_panel.pack(side="right", fill="y", padx=(10, 0))
        preview_panel.pack_propagate(False)

        tk.Label(preview_panel, text="Cut Preview", bg="#1f2937", fg="#f9fafb", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        tk.Label(preview_panel, text="Start Frame:", bg="#1f2937", fg="#93c5fd", font=("Segoe UI", 9)).pack(anchor="w")
        self.start_thumb_lbl = tk.Label(preview_panel, bg="#0f172a", width=240, height=155)
        self.start_thumb_lbl.pack(pady=(4, 10))

        tk.Label(preview_panel, text="End Frame (before black fade):", bg="#1f2937", fg="#fca5a5", font=("Segoe UI", 9)).pack(anchor="w")
        self.end_thumb_lbl = tk.Label(preview_panel, bg="#0f172a", width=240, height=155)
        self.end_thumb_lbl.pack(pady=(4, 10))

        adj_frame = tk.Frame(preview_panel, bg="#1f2937")
        adj_frame.pack(fill="x", pady=(4, 0))
        tk.Label(adj_frame, text="Start (s):", bg="#1f2937", fg="#e5e7eb", font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w")
        self.adj_start_var = tk.StringVar()
        tk.Entry(adj_frame, textvariable=self.adj_start_var, width=7, bg="#374151", fg="#ffffff", relief="flat").grid(row=0, column=1, padx=3)
        tk.Label(adj_frame, text="End (s):", bg="#1f2937", fg="#e5e7eb", font=("Segoe UI", 8)).grid(row=0, column=2, sticky="w")
        self.adj_end_var = tk.StringVar()
        tk.Entry(adj_frame, textvariable=self.adj_end_var, width=7, bg="#374151", fg="#ffffff", relief="flat").grid(row=0, column=3, padx=3)
        tk.Button(adj_frame, text="Set", command=self._apply_manual_override, bg="#374151", fg="#ffffff", relief="flat", padx=8, pady=2).grid(row=0, column=4, padx=(4, 0))

    # ------------------------------------------------------------------
    # TAB 2: TRAIN MODEL
    # ------------------------------------------------------------------
    def _build_train_tab(self) -> None:
        top = tk.Frame(self.train_tab, bg="#111827")
        top.pack(fill="x", pady=(6, 8))

        btn_blue = {"bg": "#2563eb", "fg": "#ffffff", "relief": "flat", "padx": 12, "pady": 6, "font": ("Segoe UI", 9, "bold"), "cursor": "hand2"}
        btn_gray = {"bg": "#374151", "fg": "#f9fafb", "relief": "flat", "padx": 10, "pady": 6, "font": ("Segoe UI", 9), "cursor": "hand2"}

        tk.Button(top, text="Open Training Video...", command=self._open_train_video, **btn_blue).pack(side="left", padx=(0, 8))
        tk.Button(top, text="Load Model", command=self._load_model_dialog, **btn_gray).pack(side="left", padx=(0, 6))
        tk.Button(top, text="Save Model", command=self._save_model_dialog, **btn_gray).pack(side="left", padx=(0, 6))
        tk.Button(top, text="Reset Model", command=self._reset_model, bg="#991b1b", fg="#ffffff", relief="flat", padx=10, pady=6, font=("Segoe UI", 9), cursor="hand2").pack(side="left")

        main = tk.Frame(self.train_tab, bg="#111827")
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg="#1f2937", padx=12, pady=12)
        left.pack(side="left", fill="both", expand=True)

        self.train_canvas_lbl = tk.Label(left, bg="#0f172a", text="Open a sample video to mark where the center app starts and ends.", fg="#9ca3af", font=("Segoe UI", 10))
        self.train_canvas_lbl.pack(fill="both", expand=True)

        self.scrubber = tk.Scale(
            left,
            from_=0.0,
            to=100.0,
            resolution=0.1,
            orient="horizontal",
            command=self._on_scrub,
            bg="#1f2937",
            fg="#f9fafb",
            troughcolor="#374151",
            highlightthickness=0,
        )
        self.scrubber.pack(fill="x", pady=(8, 6))

        step_bar = tk.Frame(left, bg="#1f2937")
        step_bar.pack(fill="x")
        for label, delta in [("-5s", -5.0), ("-1s", -1.0), ("-1f", -1 / 30), ("+1f", 1 / 30), ("+1s", 1.0), ("+5s", 5.0)]:
            tk.Button(
                step_bar,
                text=label,
                command=lambda d=delta: self._step_train_time(d),
                bg="#374151",
                fg="#ffffff",
                relief="flat",
                padx=8,
                pady=4,
            ).pack(side="left", padx=2)

        self.train_time_lbl = tk.Label(step_bar, text="00:00.00 / 00:00.00", bg="#1f2937", fg="#93c5fd", font=("Segoe UI", 10, "bold"))
        self.train_time_lbl.pack(side="right")

        right = tk.Frame(main, bg="#1f2937", width=300, padx=12, pady=12)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Teach Intro / Outro", bg="#1f2937", fg="#f9fafb", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        tk.Label(
            right,
            text="1. Scrub past the initialization frames.\n2. Click 'Set Start'.\n3. Scrub right before the ending black fade.\n4. Click 'Set End'.\n5. Click 'Train & Save'.",
            bg="#1f2937",
            fg="#d1d5db",
            justify="left",
            wraplength=275,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 12))

        tk.Button(right, text="Set Start (After Init)", command=self._set_keep_start, bg="#2563eb", fg="#ffffff", relief="flat", pady=7, font=("Segoe UI", 9, "bold")).pack(fill="x", pady=(0, 4))
        self.keep_start_lbl = tk.Label(right, text="Start: 00:00.00", bg="#1f2937", fg="#93c5fd", font=("Segoe UI", 9, "bold"))
        self.keep_start_lbl.pack(anchor="w", pady=(0, 10))

        tk.Button(right, text="Set End (Before Black Fade)", command=self._set_keep_end, bg="#dc2626", fg="#ffffff", relief="flat", pady=7, font=("Segoe UI", 9, "bold")).pack(fill="x", pady=(0, 4))
        self.keep_end_lbl = tk.Label(right, text="End: 00:00.00", bg="#1f2937", fg="#fca5a5", font=("Segoe UI", 9, "bold"))
        self.keep_end_lbl.pack(anchor="w", pady=(0, 14))

        tk.Button(right, text="Train & Save Model", command=self._train_current_video_async, bg="#059669", fg="#ffffff", relief="flat", pady=9, font=("Segoe UI", 10, "bold")).pack(fill="x", pady=(0, 12))

        tk.Label(right, text="Trained Videos:", bg="#1f2937", fg="#e5e7eb", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.train_history_box = tk.Listbox(right, bg="#0f172a", fg="#d1d5db", relief="flat", font=("Segoe UI", 8))
        self.train_history_box.pack(fill="both", expand=True, pady=(4, 0))

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def _refresh_model_status(self) -> None:
        if self.model.is_trained():
            n_v = len(self.model.training_records)
            self.model_status_lbl.config(text=f"Model Ready ({n_v} video(s) trained)", fg="#86efac")
        else:
            self.model_status_lbl.config(text="Black-Fade Only (No Model Trained)", fg="#fcd34d")

        self.train_history_box.delete(0, "end")
        for rec in self.model.training_records:
            self.train_history_box.insert("end", f"{rec['video']} ({rec['keep_start_sec']:.1f}s - {rec['keep_end_sec']:.1f}s)")

    def _add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select VR Video(s)",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        for p in paths:
            if p not in self.queue_paths:
                self.queue_paths.append(p)
                try:
                    _, dur, _ = get_video_info(p)
                except Exception:
                    dur = 0.0
                self.tree.insert("", "end", iid=p, values=(os.path.basename(p), fmt_time(dur), "-", "-", "-", "-", "Queued"))

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder of VR Videos")
        if not folder:
            return
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                p = os.path.join(folder, name)
                if p not in self.queue_paths:
                    self.queue_paths.append(p)
                    try:
                        _, dur, _ = get_video_info(p)
                    except Exception:
                        dur = 0.0
                    self.tree.insert("", "end", iid=p, values=(name, fmt_time(dur), "-", "-", "-", "-", "Queued"))

    def _update_total_duration(self) -> None:
        total_orig = sum(r.duration for r in self.detections.values())
        total_kept = sum(max(0.0, r.end_time - r.start_time) for r in self.detections.values())
        total_removed = max(0.0, total_orig - total_kept)
        self.total_dur_lbl.config(
            text=f"Total Original: {fmt_time(total_orig)}   |   Trimmed Total Duration: {fmt_time(total_kept)}   |   Removed: {fmt_time(total_removed)}"
        )

    def _clear_queue(self) -> None:
        self.queue_paths.clear()
        self.detections.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._update_total_duration()

    def _choose_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            self.output_dir = folder
            self.out_dir_lbl.config(text=self.output_dir)

    def _open_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if os.name == "nt":
            os.startfile(self.output_dir)

    def _analyze_all_async(self) -> None:
        if not self.queue_paths:
            messagebox.showinfo("No Videos", "Add at least one video first.")
            return

        def worker() -> None:
            for p in list(self.queue_paths):
                self.after(0, lambda p=p: self.tree.set(p, "status", "Analyzing..."))
                try:
                    res = detect_trim_points(
                        p,
                        model=self.model if self.model.is_trained() else None,
                        progress_cb=lambda msg: self.after(0, lambda m=msg: self.status_var.set(m)),
                    )
                    self.detections[p] = res
                    kept = res.end_time - res.start_time
                    self.after(
                        0,
                        lambda p=p, r=res, k=kept, tr=max(0.0, res.duration - kept): self.tree.item(
                            p,
                            values=(
                                os.path.basename(p),
                                fmt_time(r.duration),
                                fmt_time(r.start_time),
                                fmt_time(r.end_time),
                                fmt_time(k),
                                fmt_time(tr),
                                "Ready to Export",
                            ),
                        ),
                    )
                except Exception as exc:
                    self.after(0, lambda p=p, e=exc: self.tree.set(p, "status", f"Error: {e}"))
            self.after(0, self._update_total_duration)
            self.after(0, lambda: self.status_var.set("Cuts detected. Select a row to preview or click 'Auto-Trim & Export All'."))

        threading.Thread(target=worker, daemon=True).start()

    def _trim_all_async(self) -> None:
        if not self.queue_paths:
            messagebox.showinfo("No Videos", "Add at least one video first.")
            return

        # Ask for output filename(s) before starting export
        export_targets: Dict[str, str] = {}
        os.makedirs(self.output_dir, exist_ok=True)
        if len(self.queue_paths) == 1:
            p = self.queue_paths[0]
            stem, ext = os.path.splitext(os.path.basename(p))
            out_path = filedialog.asksaveasfilename(
                title="Save Trimmed Video As",
                initialdir=self.output_dir,
                initialfile=f"{stem}_trimmed{ext}",
                defaultextension=ext or ".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")],
            )
            if not out_path:
                return
            export_targets[p] = out_path
        else:
            ask_each = messagebox.askyesno(
                "Export Filenames",
                "Do you want to choose a filename for each video?\n\nYes = Ask filename for each\nNo = Auto-name all in Export Folder",
            )
            for p in self.queue_paths:
                stem, ext = os.path.splitext(os.path.basename(p))
                if ask_each:
                    out_path = filedialog.asksaveasfilename(
                        title=f"Save Trimmed Video As ({os.path.basename(p)})",
                        initialdir=self.output_dir,
                        initialfile=f"{stem}_trimmed{ext}",
                        defaultextension=ext or ".mp4",
                        filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")],
                    )
                    if not out_path:
                        continue
                    export_targets[p] = out_path
                else:
                    export_targets[p] = os.path.join(self.output_dir, f"{stem}_trimmed{ext}")

        if not export_targets:
            return

        def worker() -> None:
            exported = 0
            last_out = ""
            for p, out_path in export_targets.items():
                try:
                    if p not in self.detections:
                        self.after(0, lambda p=p: self.tree.set(p, "status", "Analyzing..."))
                        res = detect_trim_points(
                            p,
                            model=self.model if self.model.is_trained() else None,
                            progress_cb=lambda msg: self.after(0, lambda m=msg: self.status_var.set(m)),
                        )
                        self.detections[p] = res
                        kept = res.end_time - res.start_time
                        self.after(
                            0,
                            lambda p=p, r=res, k=kept, tr=max(0.0, res.duration - kept): self.tree.item(
                                p,
                                values=(
                                    os.path.basename(p),
                                    fmt_time(r.duration),
                                    fmt_time(r.start_time),
                                    fmt_time(r.end_time),
                                    fmt_time(k),
                                    fmt_time(tr),
                                    "Exporting...",
                                ),
                            ),
                        )
                    else:
                        self.after(0, lambda p=p: self.tree.set(p, "status", "Exporting..."))

                    res = self.detections[p]
                    self.after(0, lambda fn=os.path.basename(out_path): self.status_var.set(f"Exporting {fn}..."))
                    trim_video_ffmpeg(p, out_path, res.start_time, res.end_time, reencode=True)
                    exported += 1
                    last_out = out_path
                    self.after(0, lambda p=p: self.tree.set(p, "status", "Exported"))
                except Exception as exc:
                    self.after(0, lambda p=p: self.tree.set(p, "status", "Export Error"))
                    self.after(0, lambda e=exc: messagebox.showerror("Export Error", str(e)))

            self.after(0, self._update_total_duration)
            if exported > 0:
                msg = f"Saved to:\n{last_out}" if exported == 1 else f"Exported {exported} video(s)."
                self.after(0, lambda: self.status_var.set(f"Exported {exported} video(s)."))
                self.after(0, lambda m=msg: messagebox.showinfo("Export Complete", m))

        threading.Thread(target=worker, daemon=True).start()

    def _export_selected_as(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select a Video", "Select a video in the list first.")
            return
        p = sel[0]
        stem, ext = os.path.splitext(os.path.basename(p))
        out_path = filedialog.asksaveasfilename(
            title="Export Trimmed Video As",
            initialfile=f"{stem}_trimmed{ext}",
            defaultextension=ext or ".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")],
        )
        if not out_path:
            return

        def worker() -> None:
            try:
                if p not in self.detections:
                    self.after(0, lambda: self.tree.set(p, "status", "Analyzing..."))
                    self.detections[p] = detect_trim_points(p, model=self.model if self.model.is_trained() else None)
                res = self.detections[p]
                self.after(0, lambda: self.tree.set(p, "status", "Exporting..."))
                trim_video_ffmpeg(p, out_path, res.start_time, res.end_time, reencode=True)
                self.after(0, lambda: self.tree.set(p, "status", "Exported"))
                self.after(0, lambda: self.status_var.set(f"Exported to {out_path}"))
                self.after(0, lambda: messagebox.showinfo("Export Complete", f"Saved to:\n{out_path}"))
            except Exception as exc:
                self.after(0, lambda: self.tree.set(p, "status", "Export Error"))
                self.after(0, lambda e=exc: messagebox.showerror("Export Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_select_queue_item(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        p = sel[0]
        if p not in self.detections:
            return
        res = self.detections[p]
        self.adj_start_var.set(f"{res.start_time:.2f}")
        self.adj_end_var.set(f"{res.end_time:.2f}")
        self._load_thumb(p, res.start_time, is_start=True)
        self._load_thumb(p, max(0.0, res.end_time - 0.1), is_start=False)

    def _load_thumb(self, video_path: str, t_sec: float, is_start: bool) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((240, 155))
        photo = ImageTk.PhotoImage(img)
        if is_start:
            self._start_thumb = photo
            self.start_thumb_lbl.config(image=photo)
        else:
            self._end_thumb = photo
            self.end_thumb_lbl.config(image=photo)

    def _apply_manual_override(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        p = sel[0]
        if p not in self.detections:
            return
        try:
            s = float(self.adj_start_var.get())
            e = float(self.adj_end_var.get())
        except ValueError:
            return
        res = self.detections[p]
        if 0 <= s < e <= res.duration:
            res.start_time = round(s, 3)
            res.end_time = round(e, 3)
            self.tree.item(
                p,
                values=(
                    os.path.basename(p),
                    fmt_time(res.duration),
                    fmt_time(res.start_time),
                    fmt_time(res.end_time),
                    fmt_time(res.end_time - res.start_time),
                    fmt_time(max(0.0, res.duration - (res.end_time - res.start_time))),
                    "Adjusted",
                ),
            )
            self._load_thumb(p, res.start_time, is_start=True)
            self._load_thumb(p, max(0.0, res.end_time - 0.1), is_start=False)

    # ------------------------------------------------------------------
    # TRAIN TAB ACTIONS
    # ------------------------------------------------------------------
    def _open_train_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Training Video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if not p:
            return
        if self.train_cap is not None:
            self.train_cap.release()
        self.train_video_path = p
        self.train_cap = cv2.VideoCapture(p)
        self.train_fps = self.train_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.train_total_frames = int(self.train_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.train_duration = self.train_total_frames / self.train_fps if self.train_fps > 0 else 0.0

        self.scrubber.config(to=round(self.train_duration, 2))
        self.scrubber.set(0.0)
        self.keep_start_sec = 0.0
        self.keep_end_sec = self.train_duration
        self.keep_start_lbl.config(text=f"Start: {fmt_time(self.keep_start_sec)}")
        self.keep_end_lbl.config(text=f"End: {fmt_time(self.keep_end_sec)}")
        self._show_train_frame(0.0)

    def _on_scrub(self, val: str) -> None:
        if self.train_cap is None:
            return
        self._show_train_frame(float(val))

    def _step_train_time(self, delta: float) -> None:
        if self.train_cap is None:
            return
        new_t = max(0.0, min(self.train_duration, self.train_current_sec + delta))
        self.scrubber.set(round(new_t, 2))
        self._show_train_frame(new_t)

    def _show_train_frame(self, t_sec: float) -> None:
        if self.train_cap is None:
            return
        self.train_current_sec = t_sec
        idx = min(int(t_sec * self.train_fps), max(0, self.train_total_frames - 1))
        self.train_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.train_cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((600, 440))
        self._train_photo = ImageTk.PhotoImage(img)
        self.train_canvas_lbl.config(image=self._train_photo, text="")
        self.train_time_lbl.config(text=f"{fmt_time(t_sec)} / {fmt_time(self.train_duration)}")

    def _set_keep_start(self) -> None:
        self.keep_start_sec = self.train_current_sec
        self.keep_start_lbl.config(text=f"Start: {fmt_time(self.keep_start_sec)}")

    def _set_keep_end(self) -> None:
        self.keep_end_sec = self.train_current_sec
        self.keep_end_lbl.config(text=f"End: {fmt_time(self.keep_end_sec)}")

    def _train_current_video_async(self) -> None:
        if not self.train_video_path:
            messagebox.showinfo("No Video", "Open a training video first.")
            return
        if self.keep_end_sec <= self.keep_start_sec:
            messagebox.showerror("Invalid Range", "End must be greater than Start.")
            return

        p = self.train_video_path
        ks, ke = self.keep_start_sec, self.keep_end_sec

        def worker() -> None:
            try:
                u, k = self.model.add_training_example(
                    p,
                    keep_start_sec=ks,
                    keep_end_sec=ke,
                    progress_cb=lambda msg: self.after(0, lambda m=msg: self.status_var.set(m)),
                )
                self.model.save(self.default_model_path)
                self.after(0, self._refresh_model_status)
                self.after(0, lambda: self.status_var.set(f"Model trained & saved ({u} unwanted, {k} keep frames)."))
            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Training Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _save_model_dialog(self) -> None:
        if not self.model.is_trained():
            messagebox.showinfo("Empty Model", "Train on at least one video first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Model", "*.npz")])
        if p:
            self.model.save(p)
            self.status_var.set(f"Model saved to {p}")

    def _load_model_dialog(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("NumPy Model", "*.npz")])
        if p:
            self.model = TrainableTrimModel.load(p)
            self._refresh_model_status()
            self.status_var.set(f"Loaded model from {p}")

    def _reset_model(self) -> None:
        self.model = TrainableTrimModel()
        if os.path.isfile(self.default_model_path):
            os.remove(self.default_model_path)
        self._refresh_model_status()
        self.status_var.set("Model reset.")


if __name__ == "__main__":
    app = VRAutoTrimmerApp()
    app.mainloop()
