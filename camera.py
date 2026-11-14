"""Live finger counting with a local webcam (0-10, not ASL numeral signs).

Install: python -m pip install mediapipe opencv-python
Run:     python camera.py
Requires Python 3.10-3.12. The first run downloads Google's hand model.
Camera frames are processed locally and are never recorded or uploaded.
"""
import argparse
import math
from pathlib import Path
import sys
import time
from urllib.request import urlopen

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
               (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20))


def distance(a, b):
    return math.dist(a, b)


def angle(a, b, c):
    """Angle at b, in degrees; independent of hand rotation and mirroring."""
    u = [x - y for x, y in zip(a, b)]
    v = [x - y for x, y in zip(c, b)]
    size = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
    if size < 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(x*y for x, y in zip(u, v)) / size))
    return math.degrees(math.acos(cosine))


def count_fingers(points):
    """Heuristic on 21 world landmarks, ordered thumb through pinky.

    Straight joints plus fingertip distance reject folded fingers. The thumb
    also has to extend away from the index knuckle. Best with palms facing
    the camera and fingers clearly separated; this is not a trained numeral
    classifier and does not implement sign language.
    """
    if len(points) != 21:
        raise ValueError('Expected 21 hand landmarks')
    palm = max(distance(points[0], points[9]), 1e-9)
    thumb = (angle(points[1], points[2], points[3]) > 145
             and angle(points[2], points[3], points[4]) > 150
             and distance(points[4], points[5]) > 0.65 * palm)
    extended = [thumb]
    for base in (5, 9, 13, 17):
        extended.append(
            angle(points[base], points[base+1], points[base+2]) > 155
            and angle(points[base+1], points[base+2], points[base+3]) > 150
            and distance(points[base+3], points[0])
            > distance(points[base+1], points[0]) * 1.12
        )
    return sum(extended), extended


class StableCount:
    """Require a consistent value for a short time; clear immediately on loss."""
    def __init__(self, seconds=0.25):
        self.seconds = seconds
        self.candidate = None
        self.since = 0.0
        self.value = None

    def update(self, value, now):
        if value is None:
            self.candidate = self.value = None
            return None
        if value != self.candidate:
            self.candidate, self.since = value, now
            self.value = None
        if now - self.since >= self.seconds:
            self.value = value
        return self.value


def prepare_model(path):
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    print('Downloading the hand model from Google (first run only)...')
    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            content = response.read()
        if not content:
            raise RuntimeError('Model download was empty')
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=Path,
                        default=Path(__file__).parent / 'models' / 'hand_landmarker.task')
    args = parser.parse_args()
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError('Install camera dependencies: python -m pip install '
                           'mediapipe opencv-python') from exc
    prepare_model(args.model)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    camera = None
    title = 'Hand numbers | Q or Esc to quit'
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
            camera = cv2.VideoCapture(args.camera, backend)
            if not camera.isOpened():
                raise RuntimeError('Cannot open camera. Allow Terminal or your IDE in '
                                   'System Settings > Privacy & Security > Camera, '
                                   'close other camera apps, or try --camera 1.')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            stable = StableCount()
            previous_timestamp = -1
            print('Show one or two palms. Fist = 0. Press Q or Esc to stop.')
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError('Camera stopped delivering frames. Check camera permissions.')
                # Mirror before detection so the landmarks align with the preview.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(previous_timestamp + 1, time.monotonic_ns() // 1_000_000)
                previous_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                total = 0
                height, width = frame.shape[:2]
                for landmarks, world in zip(result.hand_landmarks, result.hand_world_landmarks):
                    count, extended = count_fingers([(p.x, p.y, p.z) for p in world])
                    total += count
                    pixels = [(int(p.x * width), int(p.y * height)) for p in landmarks]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, pixels[a], pixels[b], (110, 220, 130), 2)
                    for i, point in enumerate(pixels):
                        cv2.circle(frame, point, 4, (245, 210, 90), -1)
                    for finger, tip in enumerate((4, 8, 12, 16, 20)):
                        color = (80, 255, 100) if extended[finger] else (70, 70, 240)
                        cv2.circle(frame, pixels[tip], 8, color, -1)
                    x, y = pixels[0]
                    cv2.putText(frame, str(count), (max(0, x-20), max(100, y-20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                value = stable.update(total if result.hand_landmarks else None, time.monotonic())
                if not result.hand_landmarks:
                    label = 'No hand detected'
                elif value is None:
                    label = 'Hold steady...'
                else:
                    label = f'Number: {value}'
                cv2.rectangle(frame, (0, 0), (width, 95), (25, 25, 30), -1)
                cv2.putText(frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (100, 255, 150), 2)
                cv2.putText(frame, 'Palms toward camera | Q / Esc: quit', (20, 77),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1)
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'Camera recognition failed: {error}', file=sys.stderr)
        sys.exit(1)
