import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    video_path: str
    duration: float
    fps: float
    start_time: float
    end_time: float
    black_fades: List[Tuple[float, float]]
    confidence_start: float
    confidence_end: float


def find_ffmpeg(custom_path: Optional[str] = None) -> str:
    """Locate ffmpeg executable on Windows/Linux (PyInstaller bundle, app folder, or PATH)."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path
    search_dirs = []
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            search_dirs.append(sys._MEIPASS)
        search_dirs.append(os.path.dirname(sys.executable))
    search_dirs.append(os.path.dirname(os.path.abspath(__file__)))
    for d in search_dirs:
        for sub in ("ffmpeg.exe", os.path.join("ffmpeg", "ffmpeg.exe"), "ffmpeg"):
            c = os.path.join(d, sub)
            if os.path.isfile(c):
                return c
    which_ff = shutil.which("ffmpeg")
    if which_ff:
        return which_ff
    try:
        import imageio_ffmpeg
        ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ff_exe and os.path.isfile(ff_exe):
            return ff_exe
    except Exception:
        pass
    return "ffmpeg"


def extract_frame_features(frame: np.ndarray) -> np.ndarray:
    """Extract a compact, illumination- and color-aware feature vector from a BGR frame."""
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1. HSV histograms (H: 16 bins, S: 8 bins, V: 8 bins) -> 32 dims
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    h_hist /= (h_hist.sum() + 1e-6)
    s_hist /= (s_hist.sum() + 1e-6)
    v_hist /= (v_hist.sum() + 1e-6)

    # 2. 4x4 spatial grid mean BGR + HSV stats -> 96 dims
    grid_feats = []
    step = 16
    for y in range(0, 64, step):
        for x in range(0, 64, step):
            patch_bgr = small[y : y + step, x : x + step].astype(np.float32) / 255.0
            patch_hsv = hsv[y : y + step, x : x + step].astype(np.float32) / 255.0
            grid_feats.extend(patch_bgr.mean(axis=(0, 1)).tolist())
            grid_feats.extend(patch_hsv.mean(axis=(0, 1)).tolist())

    # 3. Global stats: mean/std luma, mean/std saturation, edge density -> 5 dims
    edges = cv2.Canny(gray, 50, 150)
    global_stats = [
        float(gray.mean() / 255.0),
        float(gray.std() / 255.0),
        float(hsv[:, :, 1].mean() / 255.0),
        float(hsv[:, :, 1].std() / 255.0),
        float((edges > 0).mean()),
    ]

    vec = np.concatenate(
        [h_hist, s_hist, v_hist, np.array(grid_feats, dtype=np.float32), np.array(global_stats, dtype=np.float32)]
    ).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec /= norm
    return vec


def get_video_info(video_path: str) -> Tuple[float, float, int]:
    """Return (fps, duration_sec, total_frames)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return float(fps), float(duration), total_frames


class TrainableTrimModel:
    """
    Lightweight trainable model that learns 'unwanted' (app initialization / VR shell / outro)
    vs 'keep' (center app/gameplay) frames using stored feature prototypes +
    top-k nearest-neighbor cosine similarity.
    """

    def __init__(self) -> None:
        self.unwanted_features: Optional[np.ndarray] = None
        self.keep_features: Optional[np.ndarray] = None
        self.training_records: List[dict] = []

    def is_trained(self) -> bool:
        return (
            self.unwanted_features is not None
            and len(self.unwanted_features) > 0
            and self.keep_features is not None
            and len(self.keep_features) > 0
        )

    def add_training_example(
        self,
        video_path: str,
        keep_start_sec: float,
        keep_end_sec: float,
        sample_interval_sec: float = 0.5,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, int]:
        """Single sequential pass to sample frames and add to training set."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        if keep_start_sec < 0 or keep_end_sec <= keep_start_sec or keep_end_sec > duration + 0.5:
            cap.release()
            raise ValueError("Invalid keep_start_sec / keep_end_sec range.")

        step = max(1, int(round(fps * sample_interval_sec)))
        unwanted_list = []
        keep_list = []
        frame_idx = 0

        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    t = frame_idx / fps
                    gray_mean = float(cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY).mean())
                    if gray_mean > 15.0:
                        feat = extract_frame_features(frame)
                        if keep_start_sec <= t <= keep_end_sec:
                            keep_list.append(feat)
                        else:
                            unwanted_list.append(feat)
                    if progress_cb:
                        progress_cb(f"Training on {os.path.basename(video_path)}: {t:.1f}s / {duration:.1f}s")
            frame_idx += 1

        cap.release()

        if unwanted_list:
            u_arr = np.vstack(unwanted_list)
            self.unwanted_features = u_arr if self.unwanted_features is None else np.vstack([self.unwanted_features, u_arr])
        if keep_list:
            k_arr = np.vstack(keep_list)
            self.keep_features = k_arr if self.keep_features is None else np.vstack([self.keep_features, k_arr])

        self.training_records.append(
            {
                "video": os.path.basename(video_path),
                "keep_start_sec": round(keep_start_sec, 3),
                "keep_end_sec": round(keep_end_sec, 3),
                "unwanted_samples": len(unwanted_list),
                "keep_samples": len(keep_list),
            }
        )
        return len(unwanted_list), len(keep_list)

    def score_frame(self, frame: np.ndarray) -> float:
        """Positive score -> 'keep' (center); negative score -> 'unwanted' (intro/outro)."""
        if not self.is_trained():
            return 0.0
        feat = extract_frame_features(frame)
        sim_keep = np.dot(self.keep_features, feat)
        sim_unwanted = np.dot(self.unwanted_features, feat)

        k_top = min(5, len(sim_keep))
        u_top = min(5, len(sim_unwanted))
        keep_score = float(np.sort(sim_keep)[-k_top:].mean())
        unwanted_score = float(np.sort(sim_unwanted)[-u_top:].mean())
        return keep_score - unwanted_score

    def save(self, model_path: str) -> None:
        if not self.is_trained():
            raise ValueError("Model has not been trained yet.")
        np.savez_compressed(
            model_path,
            unwanted_features=self.unwanted_features,
            keep_features=self.keep_features,
            records_json=np.array([json.dumps(self.training_records)]),
        )

    @classmethod
    def load(cls, model_path: str) -> "TrainableTrimModel":
        data = np.load(model_path, allow_pickle=False)
        model = cls()
        model.unwanted_features = data["unwanted_features"]
        model.keep_features = data["keep_features"]
        if "records_json" in data:
            model.training_records = json.loads(str(data["records_json"][0]))
        return model


def detect_trim_points(
    video_path: str,
    model: Optional[TrainableTrimModel] = None,
    black_luma_threshold: float = 18.0,
    black_min_duration: float = 0.05,
    sample_interval_sec: float = 0.2,
    progress_cb: Optional[Callable[[str], None]] = None,
    frame_progress_cb: Optional[Callable[[int, int], None]] = None,
) -> DetectionResult:
    """
    Single-pass sequential scan to:
    1) Detect black fade intervals.
    2) Classify frames using the trained model (if provided).
    3) Return exact (start_time, end_time) omitting app initialization frames and ending black clips.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    # Sample at ~15 fps for black fade detection, and every ~0.2s for classifier
    black_step = max(1, int(round(fps / 15.0)))
    cls_step = max(1, int(round(fps * sample_interval_sec)))

    black_fades: List[Tuple[float, float]] = []
    in_black = False
    black_start = 0.0

    times: List[float] = []
    scores: List[float] = []
    use_model = model is not None and model.is_trained()

    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break

        need_black = (frame_idx % black_step == 0)
        need_cls = use_model and (frame_idx % cls_step == 0)

        if need_black or need_cls:
            ok, frame = cap.retrieve()
            if ok:
                t = frame_idx / fps
                if need_black:
                    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                    mean_luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
                    if mean_luma <= black_luma_threshold:
                        if not in_black:
                            in_black = True
                            black_start = t
                    else:
                        if in_black:
                            in_black = False
                            if (t - black_start) >= black_min_duration:
                                black_fades.append((black_start, t))

                if need_cls:
                    times.append(t)
                    scores.append(model.score_frame(frame))
                    if progress_cb and len(times) % 15 == 0:
                        progress_cb(f"Analyzing {os.path.basename(video_path)}: {t:.1f}s / {duration:.1f}s")
        if frame_progress_cb and (frame_idx % 15 == 0):
            frame_progress_cb(frame_idx, total_frames)
        frame_idx += 1
    if frame_progress_cb:
        frame_progress_cb(total_frames, total_frames)

    if in_black:
        t_end = frame_idx / fps
        if (t_end - black_start) >= black_min_duration:
            black_fades.append((black_start, t_end))

    cap.release()

    start_time = 0.0
    end_time = duration
    conf_start = 0.5
    conf_end = 0.5

    if scores:
        scores_arr = np.array(scores, dtype=np.float32)
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(scores_arr, kernel, mode="same")
        keep_mask = smooth > 0.0

        # First sustained 'keep' region
        run = 0
        for i, is_keep in enumerate(keep_mask):
            if is_keep:
                run += 1
                if run >= 3:
                    start_idx = max(0, i - 2)
                    while start_idx > 0 and scores_arr[start_idx - 1] > 0:
                        start_idx -= 1
                    start_time = times[start_idx]
                    conf_start = float(min(1.0, abs(smooth[i]) * 5.0))
                    break
            else:
                run = 0

        # Last sustained 'keep' region
        run = 0
        for i in range(len(keep_mask) - 1, -1, -1):
            if keep_mask[i]:
                run += 1
                if run >= 3:
                    end_idx = min(len(times) - 1, i + 2)
                    while end_idx < len(times) - 1 and scores_arr[end_idx + 1] > 0:
                        end_idx += 1
                    end_time = times[end_idx]
                    conf_end = float(min(1.0, abs(smooth[i]) * 5.0))
                    break
            else:
                run = 0

    # Snap intro cut past any intro black fade
    for b_start, b_end in black_fades:
        if b_end <= duration * 0.5 and abs(b_end - start_time) <= 2.0:
            start_time = b_end
            conf_start = max(conf_start, 0.9)
            break

    # Snap ending cut right before the ending black fade (omitting black clip + outro)
    if black_fades:
        if scores:
            candidates = [bf for bf in black_fades if bf[0] > start_time + 1.0 and abs(bf[0] - end_time) <= 5.0]
            if candidates:
                end_time = candidates[0][0]
                conf_end = max(conf_end, 0.95)
        else:
            candidates = [bf for bf in black_fades if bf[0] >= duration * 0.5]
            if candidates:
                end_time = candidates[-1][0]
                conf_end = 0.85

    if start_time > 0.5:
        start_time = min(duration, start_time + 0.35)
    start_time = max(0.0, min(start_time, duration))
    end_time = max(start_time + 0.1, min(end_time, duration))

    return DetectionResult(
        video_path=video_path,
        duration=duration,
        fps=fps,
        start_time=round(start_time, 3),
        end_time=round(end_time, 3),
        black_fades=black_fades,
        confidence_start=round(conf_start, 2),
        confidence_end=round(conf_end, 2),
    )


def trim_video_ffmpeg(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    reencode: bool = True,
    ffmpeg_bin: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> None:
    """Trim video to [start_sec, end_sec] using ffmpeg with live percentage progress."""
    exe = find_ffmpeg(ffmpeg_bin)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    duration = max(0.1, end_sec - start_sec)

    base_cmd = [
        exe,
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
    ]
    if reencode:
        base_cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    else:
        base_cmd += ["-c", "copy"]

    cmd = base_cmd + ["-progress", "pipe:1", "-nostats", output_path]
    kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True, "bufsize": 1}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    proc = subprocess.Popen(cmd, **kwargs)
    if proc.stdout:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    val = int(line.split("=", 1)[1])
                    pct = min(100.0, max(0.0, (val / 1_000_000.0) / duration * 100.0))
                    if progress_cb:
                        progress_cb(pct)
                except ValueError:
                    pass
            elif line == "progress=end" and progress_cb:
                progress_cb(100.0)

    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{(err or '')[-800:]}")
