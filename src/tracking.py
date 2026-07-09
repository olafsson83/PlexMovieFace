"""Optical-flow tracking of face keypoints between full-detection frames,
plus scene-cut detection to force re-detection when tracking would no
longer be valid.

insightface's INSwapper.get() only ever reads target_face.kps (the 5
facial landmark points) to align the swap -- it never touches the bounding
box. That means tracking just those 5 points with cv2.calcOpticalFlowPyrLK
(built for exactly this kind of sparse point tracking) is enough to keep a
swap correctly aligned on frames where we skip the expensive full detection
pass. Only faces that will actually be swapped (matched to a character AND
a source photo provided) are worth tracking -- there's no swap action for
anything else, so no point spending optical-flow work on it.
"""
import cv2
import numpy as np

from config import DETECT_EVERY_N_FRAMES, SCENE_CUT_THRESHOLD

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class TrackedFace:
    """Lightweight stand-in for an insightface Face -- swapper.get() only
    ever reads .kps, so that's all this needs to provide."""
    __slots__ = ("kps", "character_number")

    def __init__(self, kps, character_number):
        self.kps = kps
        self.character_number = character_number


class FaceTracker:
    def __init__(self, detect_every_n_frames=DETECT_EVERY_N_FRAMES):
        self._tracked = []
        self._prev_gray = None
        self._frames_since_detection = 0
        self._detect_every_n_frames = detect_every_n_frames

    def is_scene_cut(self, gray):
        """Checks the given frame against the last frame this tracker saw."""
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return True
        diff = cv2.absdiff(self._prev_gray, gray)
        return float(diff.mean()) > SCENE_CUT_THRESHOLD

    def due_for_detection(self, gray):
        """True if this frame should get a full detection pass rather than
        tracking -- either enough tracked frames have gone by, nothing is
        currently being tracked, or a scene cut just invalidated whatever
        was being tracked.
        """
        if not self._tracked:
            return True
        if self._frames_since_detection >= self._detect_every_n_frames:
            return True
        return self.is_scene_cut(gray)

    def start_from_detection(self, gray, swappable_faces):
        """Call on a full-detection frame. swappable_faces: list of
        (kps, character_number) pairs for faces that will actually be
        swapped this frame. Returns the resulting TrackedFace list (same
        shape as track()'s return), so callers can treat both uniformly.
        """
        self._tracked = [TrackedFace(kps.copy(), number) for kps, number in swappable_faces]
        self._prev_gray = gray
        self._frames_since_detection = 0
        return self._tracked

    def track(self, gray):
        """Call on a non-detection frame. Returns the TrackedFace list
        still successfully tracked -- lost ones are silently dropped and
        picked up again at the next full detection.
        """
        self._frames_since_detection += 1

        if not self._tracked or self._prev_gray is None:
            self._prev_gray = gray
            return []

        h, w = gray.shape[:2]
        still_tracked = []
        for face in self._tracked:
            pts = face.kps.astype(np.float32).reshape(-1, 1, 2)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts, None, **LK_PARAMS)
            if new_pts is None or status is None or not status.all():
                continue  # lost one or more points -- drop until next full detection
            new_kps = new_pts.reshape(-1, 2)
            if (new_kps[:, 0] < 0).any() or (new_kps[:, 0] >= w).any() \
                    or (new_kps[:, 1] < 0).any() or (new_kps[:, 1] >= h).any():
                continue  # a tracked point left the frame
            face.kps = new_kps
            still_tracked.append(face)

        self._tracked = still_tracked
        self._prev_gray = gray
        return still_tracked
