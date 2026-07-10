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

    def hint_for(self, kps):
        """Returns the character_number of a currently-tracked face whose
        last known position is close to `kps`, or None. Call this on a
        freshly detected face *before* classifying it, and classify with a
        lower "maintain" threshold for the hinted number specifically.

        A face that's already been confidently identified and is still
        roughly where it was is much stronger evidence than a single
        frame's raw score -- a real continuous shot's per-frame embedding
        is noisier than a clean discovery-crop match and can dip below the
        "acquire" bar for an instant without the face having changed at
        all. Re-clearing the full bar on every single detection pass turns
        that noise into on/off flicker. A face with no prior track nearby
        still has to clear the full bar, so this can't resurrect a wrong
        first guess the way carrying forward a whole track could.
        """
        for face in self._tracked:
            if self._same_face(kps, face.kps):
                return face.character_number
        return None

    @staticmethod
    def _same_face(kps_a, kps_b):
        """True if two kps sets are roughly at the same on-screen spot. The
        radius scales with kps_a's own spread so it works across video
        resolutions and face sizes.
        """
        spread = kps_a.max(axis=0) - kps_a.min(axis=0)
        radius = 1.5 * float(np.hypot(*spread)) if spread.any() else 40.0
        center_a = kps_a.mean(axis=0)
        center_b = np.asarray(kps_b).mean(axis=0)
        return np.linalg.norm(center_a - center_b) < radius

    def start_from_detection(self, gray, swappable_faces, all_detected_kps=()):
        """Call on a full-detection frame. swappable_faces: list of
        (kps, character_number) pairs for faces that will actually be
        swapped this frame. all_detected_kps: kps for *every* face this
        detection pass found, regardless of classification outcome -- used
        to tell apart "this tracked face's region wasn't examined this
        frame" from "this tracked face's region was examined and is no
        longer a match" (see below). Returns the resulting TrackedFace list
        (same shape as track()'s return), so callers can treat both
        uniformly.

        A full-detection pass missing a character it just tracked fine one
        frame earlier (a bad-angle instant right at this specific checkpoint
        frame) used to wipe that character's track entirely, dropping the
        swap until the *next* successful full detection -- often several
        frames later. Such misses now carry forward via one more
        optical-flow step (the same mechanism track() uses for skipped
        frames) instead of being dropped outright.

        That carry-forward must NOT apply when detection actually examined
        this exact face and reclassified it -- e.g. a borderline embedding
        match gets corrected to its real (unswapped) cluster a few frames
        after an initial false positive. Carrying the old identity forward
        in that case would resurrect a classification detection itself just
        corrected, permanently pasting the wrong character onto that face
        for as long as it stays on screen. So a track is only carried
        forward if no detected face this pass was even near its last known
        position; if one was, that fresh (possibly negative) result wins.
        """
        fresh = [TrackedFace(kps.copy(), number) for kps, number in swappable_faces]
        fresh_numbers = {face.character_number for face in fresh}

        still_missing = [
            face for face in self._tracked
            if face.character_number not in fresh_numbers
            and not self._claimed_by_detection(face.kps, all_detected_kps)
        ]
        carried = self._track_step(self._prev_gray, gray, still_missing)

        self._tracked = fresh + carried
        self._prev_gray = gray
        self._frames_since_detection = 0
        return self._tracked

    def _claimed_by_detection(self, kps, all_detected_kps):
        """True if some detected face this pass sits roughly where `kps`
        (a previously tracked face) last was -- i.e. detection did examine
        this face, whatever it concluded, so a stale track shouldn't
        override that.
        """
        return any(self._same_face(kps, other_kps) for other_kps in all_detected_kps)

    def track(self, gray):
        """Call on a non-detection frame. Returns the TrackedFace list
        still successfully tracked -- lost ones are silently dropped and
        picked up again at the next full detection.
        """
        self._frames_since_detection += 1
        self._tracked = self._track_step(self._prev_gray, gray, self._tracked)
        self._prev_gray = gray
        return self._tracked

    def _track_step(self, prev_gray, gray, faces):
        """One optical-flow step for the given TrackedFace list. Returns the
        survivors with updated kps -- lost ones (points untrackable or that
        left the frame) are dropped.
        """
        if not faces or prev_gray is None:
            return []

        h, w = gray.shape[:2]
        survivors = []
        for face in faces:
            pts = face.kps.astype(np.float32).reshape(-1, 1, 2)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
            if new_pts is None or status is None or not status.all():
                continue  # lost one or more points -- drop until next full detection
            new_kps = new_pts.reshape(-1, 2)
            if (new_kps[:, 0] < 0).any() or (new_kps[:, 0] >= w).any() \
                    or (new_kps[:, 1] < 0).any() or (new_kps[:, 1] >= h).any():
                continue  # a tracked point left the frame
            face.kps = new_kps
            survivors.append(face)
        return survivors
