"""Display live YOLO instance masks from a local V4L2 camera."""

import argparse
import time

import cv2
from markerless_object_perception.yolo_segmenter import (
    YoloInstanceSegmenter,
    YoloSegmenterConfig,
)
import numpy as np


WINDOW_NAME = 'Objective 3.2 YOLO segmentation - Q/ESC to close'


def draw_detections(frame, detections, fps: float):
    """Overlay masks, class labels, confidence, and track IDs."""
    annotated = frame.copy()
    for detection in detections:
        color = _track_color(detection.track_id)
        colored = np.empty_like(annotated)
        colored[:] = color
        annotated[detection.mask] = cv2.addWeighted(
            annotated,
            0.55,
            colored,
            0.45,
            0.0,
        )[detection.mask]

        mask_u8 = detection.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, color, 2)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label = (
                f'{detection.class_label} '
                f'{detection.confidence:.2f} ID:{detection.track_id}'
            )
            cv2.putText(
                annotated,
                label,
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        annotated,
        f'2D YOLO only | {fps:.1f} FPS | stereo XYZ pending',
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def run_demo(args) -> None:
    """Open the camera, run tracking, and display annotated frames."""
    segmenter = YoloInstanceSegmenter(
        YoloSegmenterConfig(
            model_path=args.model,
            min_confidence=args.confidence,
            tracker=args.tracker,
            device=args.inference_device,
        )
    )

    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        if not capture.isOpened():
            raise RuntimeError(f'could not open /dev/video{args.camera}')

        previous_time = time.perf_counter()
        smoothed_fps = 0.0
        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('camera stopped returning frames')

            detections = segmenter.track(frame)
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1.0e-6)
            previous_time = now
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instant_fps
            )
            annotated = draw_detections(frame, detections, smoothed_fps)
            cv2.imshow(WINDOW_NAME, annotated)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a repeatable bright BGR color for one temporary track."""
    return (
        64 + (37 * track_id) % 192,
        64 + (17 * track_id) % 192,
        64 + (97 * track_id) % 192,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Live YOLO instance-segmentation smoke test.',
    )
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', default='yolo26n-seg.pt')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--tracker', default='bytetrack.yaml')
    parser.add_argument('--inference-device', default='cpu')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Stop automatically after N frames; 0 waits for Q/ESC.',
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line webcam demonstration."""
    run_demo(_parse_args())


if __name__ == '__main__':
    main()
