"""
Gaze Tracking Module for ASD Screening
=======================================
Uses MediaPipe FaceMesh to track iris position and compute gaze features.

Key ASD gaze markers (from PMC11719697):
- Reduced joint attention (not following moving stimuli)
- Atypical fixation patterns
- Lower gaze tracking smoothness
- Reduced face-looking time
- Irregular saccade patterns
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


@dataclass
class GazeFrame:
    timestamp: float
    left_iris: Tuple[float, float]
    right_iris: Tuple[float, float]
    gaze_x: float          # normalized -1 (left) to 1 (right)
    gaze_y: float          # normalized -1 (up) to 1 (down)
    dot_x: float           # where the dot was
    dot_y: float           # where the dot was
    following: bool        # is gaze near the dot?
    blink_detected: bool


@dataclass
class GazeSessionResult:
    following_ratio: float       # 0-1, higher = more typical
    avg_gaze_deviation: float    # avg distance gaze->dot (pixels, normalized)
    fixation_count: int          # number of fixations detected
    saccade_smoothness: float    # 0-1, higher = smoother tracking
    mutual_gaze_ratio: float     # ratio of time looking at face region
    blink_rate: float            # blinks per minute
    gaze_variability: float      # std of gaze positions
    frames_tracked: int
    total_frames: int
    asd_risk_score: float        # 0-100, higher = more risk indicators
    raw_frames: List[GazeFrame] = field(default_factory=list)


# MediaPipe landmark indices for iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corners for normalization
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# Eye contour for blink detection
LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_TOP_BOTTOM = [386, 374]


class GazeTracker:
    """Real-time gaze tracker using MediaPipe FaceMesh with iris refinement."""

    def __init__(self, dot_follow_threshold: float = 0.25):
        """
        Args:
            dot_follow_threshold: Normalized distance threshold to consider
                                  gaze as 'following' the dot (0.0-1.0)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.dot_follow_threshold = dot_follow_threshold
        self.frames: List[GazeFrame] = []
        self._blink_history = []
        self._prev_gaze = None
        self._gaze_velocities = []

    def reset(self):
        """Reset session data."""
        self.frames = []
        self._blink_history = []
        self._prev_gaze = None
        self._gaze_velocities = []

    def get_iris_center(self, landmarks, indices, img_w, img_h) -> Optional[Tuple[float, float]]:
        """Get center of iris from landmark indices."""
        points = []
        for idx in indices:
            lm = landmarks[idx]
            points.append((lm.x * img_w, lm.y * img_h))
        if not points:
            return None
        cx = np.mean([p[0] for p in points])
        cy = np.mean([p[1] for p in points])
        return (cx, cy)

    def compute_gaze_ratio(self, iris_center, eye_corners, img_w, img_h) -> float:
        """
        Compute normalized gaze ratio for one eye.
        Returns value -1 (leftmost) to 1 (rightmost).
        """
        left_corner = (eye_corners[0].x * img_w, eye_corners[0].y * img_h)
        right_corner = (eye_corners[1].x * img_w, eye_corners[1].y * img_h)

        eye_width = right_corner[0] - left_corner[0]
        if eye_width < 1:
            return 0.0

        # Position of iris relative to eye width, normalized to [-1, 1]
        relative_pos = (iris_center[0] - left_corner[0]) / eye_width
        return (relative_pos - 0.5) * 2.0  # Center = 0, left = -1, right = 1

    def detect_blink(self, landmarks, img_h) -> bool:
        """Simple blink detection via eye aspect ratio."""
        left_top = landmarks[LEFT_EYE_TOP_BOTTOM[0]]
        left_bot = landmarks[LEFT_EYE_TOP_BOTTOM[1]]
        right_top = landmarks[RIGHT_EYE_TOP_BOTTOM[0]]
        right_bot = landmarks[RIGHT_EYE_TOP_BOTTOM[1]]

        left_ear = abs(left_top.y - left_bot.y) * img_h
        right_ear = abs(right_top.y - right_bot.y) * img_h

        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < 5.0  # Threshold in pixels

    def process_frame(
        self,
        frame: np.ndarray,
        dot_pos: Tuple[float, float],   # Normalized (0-1) dot position
        timestamp: float = None
    ) -> Tuple[np.ndarray, Optional[GazeFrame]]:
        """
        Process a single video frame.

        Args:
            frame: BGR image from webcam
            dot_pos: (x, y) normalized 0-1 position of the tracking dot
            timestamp: time in seconds

        Returns:
            (annotated_frame, GazeFrame or None if face not detected)
        """
        if timestamp is None:
            timestamp = time.time()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        annotated = frame.copy()

        if not results.multi_face_landmarks:
            return annotated, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Get iris centers
        left_iris = self.get_iris_center(landmarks, LEFT_IRIS, w, h)
        right_iris = self.get_iris_center(landmarks, RIGHT_IRIS, w, h)

        if left_iris is None or right_iris is None:
            return annotated, None

        # Compute gaze ratios
        left_corners = [landmarks[LEFT_EYE_CORNERS[0]], landmarks[LEFT_EYE_CORNERS[1]]]
        right_corners = [landmarks[RIGHT_EYE_CORNERS[0]], landmarks[RIGHT_EYE_CORNERS[1]]]

        left_gaze_x = self.compute_gaze_ratio(left_iris, left_corners, w, h)
        right_gaze_x = self.compute_gaze_ratio(right_iris, right_corners, w, h)
        avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0

        # Vertical gaze (simplified - use iris y relative to eye center)
        avg_gaze_y = (left_iris[1] + right_iris[1]) / 2.0 / h - 0.5  # -0.5 to 0.5

        # Blink detection
        blink = self.detect_blink(landmarks, h)
        if blink:
            self._blink_history.append(timestamp)

        # Gaze smoothness (velocity)
        if self._prev_gaze is not None:
            dt = 0.033  # ~30fps
            vx = (avg_gaze_x - self._prev_gaze[0]) / dt
            vy = (avg_gaze_y - self._prev_gaze[1]) / dt
            self._gaze_velocities.append(math.sqrt(vx**2 + vy**2))
        self._prev_gaze = (avg_gaze_x, avg_gaze_y)

        # Check if following the dot
        # Map dot_pos (0-1) to gaze space (-1 to 1)
        dot_gaze_x = (dot_pos[0] - 0.5) * 2.0
        dot_gaze_y = (dot_pos[1] - 0.5) * 2.0

        gaze_dist = math.sqrt(
            (avg_gaze_x - dot_gaze_x)**2 + (avg_gaze_y - dot_gaze_y)**2
        )
        following = gaze_dist < self.dot_follow_threshold

        gaze_frame = GazeFrame(
            timestamp=timestamp,
            left_iris=left_iris,
            right_iris=right_iris,
            gaze_x=avg_gaze_x,
            gaze_y=avg_gaze_y,
            dot_x=dot_pos[0],
            dot_y=dot_pos[1],
            following=following,
            blink_detected=blink
        )
        self.frames.append(gaze_frame)

        # Draw annotations
        annotated = self._draw_annotations(annotated, left_iris, right_iris, following, landmarks, w, h)

        return annotated, gaze_frame

    def _draw_annotations(self, frame, left_iris, right_iris, following, landmarks, w, h):
        """Draw gaze tracking visualizations on frame."""
        # Draw iris circles
        cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 3, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 3, (0, 255, 0), -1)

        # Gaze status indicator
        color = (0, 255, 0) if following else (0, 0, 255)
        status = "FOLLOWING" if following else "NOT FOLLOWING"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Draw face mesh - key landmarks only
        for idx in LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 200, 0), -1)

        return frame

    def compute_session_results(self) -> GazeSessionResult:
        """
        Analyze all collected frames and compute ASD risk indicators.
        Based on findings from PMC11719697:
        - ASD children show reduced gaze following
        - Higher gaze variability
        - Irregular fixation patterns
        """
        if not self.frames:
            return GazeSessionResult(
                following_ratio=0, avg_gaze_deviation=1.0,
                fixation_count=0, saccade_smoothness=0,
                mutual_gaze_ratio=0, blink_rate=0,
                gaze_variability=1.0, frames_tracked=0,
                total_frames=0, asd_risk_score=50.0
            )

        total = len(self.frames)
        following_frames = sum(1 for f in self.frames if f.following)
        following_ratio = following_frames / total

        # Average deviation of gaze from dot
        deviations = [
            math.sqrt((f.gaze_x - (f.dot_x - 0.5) * 2)**2 + (f.gaze_y - (f.dot_y - 0.5) * 2)**2)
            for f in self.frames
        ]
        avg_deviation = np.mean(deviations) if deviations else 1.0

        # Fixation count (periods of low movement)
        fixation_count = self._count_fixations()

        # Saccade smoothness (inverse of velocity variance)
        if self._gaze_velocities:
            vel_var = np.var(self._gaze_velocities)
            smoothness = 1.0 / (1.0 + vel_var / 100.0)
        else:
            smoothness = 0.5

        # Blink rate
        duration_secs = self.frames[-1].timestamp - self.frames[0].timestamp if total > 1 else 1
        blink_rate = len(self._blink_history) / (duration_secs / 60.0) if duration_secs > 0 else 0

        # Gaze variability
        gaze_xs = [f.gaze_x for f in self.frames]
        gaze_variability = np.std(gaze_xs) if gaze_xs else 1.0

        # Mutual gaze ratio (approximated: time looking near center/face region)
        center_gaze = sum(1 for f in self.frames if abs(f.gaze_x) < 0.3 and abs(f.gaze_y) < 0.3)
        mutual_gaze_ratio = center_gaze / total

        # Compute ASD risk score (0-100)
        asd_risk = self._compute_risk_score(
            following_ratio, avg_deviation, smoothness,
            blink_rate, gaze_variability, mutual_gaze_ratio
        )

        return GazeSessionResult(
            following_ratio=following_ratio,
            avg_gaze_deviation=avg_deviation,
            fixation_count=fixation_count,
            saccade_smoothness=smoothness,
            mutual_gaze_ratio=mutual_gaze_ratio,
            blink_rate=blink_rate,
            gaze_variability=gaze_variability,
            frames_tracked=total,
            total_frames=total,
            asd_risk_score=asd_risk,
            raw_frames=self.frames.copy()
        )

    def _count_fixations(self, velocity_threshold: float = 0.05, min_duration: int = 5) -> int:
        """Count fixation events (periods of stable gaze)."""
        if len(self.frames) < min_duration:
            return 0

        fixations = 0
        in_fixation = False
        fixation_len = 0

        for i in range(1, len(self.frames)):
            dx = abs(self.frames[i].gaze_x - self.frames[i-1].gaze_x)
            dy = abs(self.frames[i].gaze_y - self.frames[i-1].gaze_y)
            velocity = math.sqrt(dx**2 + dy**2)

            if velocity < velocity_threshold:
                fixation_len += 1
                if not in_fixation and fixation_len >= min_duration:
                    in_fixation = True
                    fixations += 1
            else:
                in_fixation = False
                fixation_len = 0

        return fixations

    def _compute_risk_score(
        self,
        following_ratio: float,
        avg_deviation: float,
        smoothness: float,
        blink_rate: float,
        gaze_variability: float,
        mutual_gaze_ratio: float
    ) -> float:
        """
        Compute ASD risk score (0-100) from gaze features.
        Higher = more ASD risk indicators present.

        Feature weights based on PMC11719697 findings.
        """
        risk = 0.0

        # Low following ratio → high risk (weight: 35)
        risk += (1.0 - following_ratio) * 35.0

        # High deviation → high risk (weight: 20)
        # deviation ranges roughly 0-2, normalize
        normalized_dev = min(avg_deviation / 1.5, 1.0)
        risk += normalized_dev * 20.0

        # Low smoothness → high risk (weight: 20)
        risk += (1.0 - smoothness) * 20.0

        # Low mutual gaze → high risk (weight: 15)
        risk += (1.0 - mutual_gaze_ratio) * 15.0

        # High gaze variability → moderate risk (weight: 10)
        normalized_var = min(gaze_variability / 0.5, 1.0)
        risk += normalized_var * 10.0

        return min(max(risk, 0.0), 100.0)


class DotAnimator:
    """
    Animates a moving dot on screen for the gaze tracking test.
    Pattern: slow left-right sweeps with pauses (STAT-inspired).
    """

    def __init__(self, canvas_width: int, canvas_height: int):
        self.W = canvas_width
        self.H = canvas_height
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.phase = 0
        self.phase_duration = 3.0  # seconds per phase
        self.phases = [
            # (start_x, end_x, y_frac) - fractions of canvas
            (0.1, 0.9, 0.5),   # Left to right, middle
            (0.9, 0.1, 0.5),   # Right to left, middle
            (0.5, 0.5, 0.5),   # Center hold
            (0.1, 0.9, 0.3),   # Left to right, upper
            (0.9, 0.1, 0.7),   # Right to left, lower
            (0.5, 0.1, 0.5),   # Center to left
            (0.1, 0.9, 0.5),   # Sweep
        ]

    def get_dot_position(self) -> Tuple[float, float]:
        """
        Returns normalized (x, y) position of the dot (0.0 to 1.0).
        """
        elapsed = time.time() - self.start_time
        phase_idx = int(elapsed / self.phase_duration) % len(self.phases)
        phase_t = (elapsed % self.phase_duration) / self.phase_duration

        sx, ex, y_frac = self.phases[phase_idx]
        x = sx + (ex - sx) * phase_t

        return (float(x), float(y_frac))

    def get_dot_pixel_pos(self) -> Tuple[int, int]:
        """Returns pixel position on the overlay canvas."""
        nx, ny = self.get_dot_position()
        return (int(nx * self.W), int(ny * self.H))

    def draw_dot(self, frame: np.ndarray, pos_override=None) -> np.ndarray:
        """Draw animated dot on frame."""
        if pos_override:
            px, py = pos_override
        else:
            px, py = self.get_dot_pixel_pos()

        # Outer ring (attention grabber)
        cv2.circle(frame, (px, py), 20, (0, 120, 255), 2)
        # Inner dot
        cv2.circle(frame, (px, py), 12, (0, 80, 255), -1)
        # Center dot
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)

        return frame
