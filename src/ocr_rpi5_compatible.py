#!/usr/bin/env python3
import cv2
import threading
import time
import numpy as np
from paddleocr import PaddleOCR

# Try to use Pi Camera via picamera2; fall back to USB cam if not present
# USE_PICAMERA2 = False
# try:
#     from picamera2 import Picamera2
#     USE_PICAMERA2 = True
# except Exception:
#     USE_PICAMERA2 = False

latest_frame = None
ocr_results = []
lock = threading.Lock()
running = True

# Tunables for Pi performance
OCR_W, OCR_H = 960, 540
OCR_SCORE_THRESH = 0.80
MAX_OVERLAYS = 3
OCR_PERIOD_S = 0.15  # throttle OCR calls

def preprocess_frame(frame):
    # CLAHE tends to work better than global hist eq for real scenes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb



def ocr_worker():
    global latest_frame, ocr_results, running

    # IMPORTANT: Only use params your PaddleOCR version supports
    ocr = PaddleOCR(
        enable_hpi=True,
        lang="en",
        use_textline_orientation=True,             # supported in your snippet
        text_det_limit_side_len=1536,              # optional, supported
        text_rec_score_thresh=0.3                  # optional, supported
        # No use_angle_cls / use_gpu / show_log here
    )

    last_run = 0.0
    while running:
        now = time.time()
        if now - last_run < OCR_PERIOD_S:
            time.sleep(0.01)
            continue
        last_run = now

        frame = None
        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # Resize for predictable OCR cost
        small = cv2.resize(frame, (OCR_W, OCR_H), interpolation=cv2.INTER_AREA)
        rgb = preprocess_frame(small)

        # Use predict() (your API) and keep your dict-like result shape
        results = ocr.predict(
            rgb,
            use_textline_orientation=True,
            text_rec_score_thresh=0.3,
            # you can also pass:
            # text_det_box_thresh=0.6, text_det_unclip_ratio=1.5, etc.
        )

        # results is already list[dict] with keys like "dt_polys", "rec_texts", "rec_scores"
        # We simply stash it, plus the working size for later scaling on overlay
        norm = []
        for res in results:
            if isinstance(res, dict):
                res = dict(res)  # shallow copy
                res["size"] = (OCR_W, OCR_H)
                norm.append(res)

        with lock:
            ocr_results = norm

def open_camera():
    """Yield BGR frames and return a cleanup callable."""
    # if USE_PICAMERA2:
    #     picam2 = Picamera2()
    #     config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    #     picam2.configure(config)
    #     picam2.start()
    #     time.sleep(0.3)

    #     def gen():
    #         while running:
    #             rgb = picam2.capture_array()
    #             yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    #     def cleanup():
    #         picam2.stop()

    #     return gen(), cleanup

    # USB webcam fallback via V4L2
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise SystemExit(" Could not open camera (PiCam/USB).")

    def gen():
        while running:
            ok, frame = cap.read()
            if ok:
                yield frame

    def cleanup():
        cap.release()

    return gen(), cleanup

def main():
    global latest_frame, ocr_results, running

    t = threading.Thread(target=ocr_worker, daemon=True)
    t.start()

    try:
        frames, cleanup = open_camera()
    except SystemExit as e:
        print(e)
        return

    print("Camera started — press 'q' to quit.")

    try:
        for frame in frames:
            with lock:
                latest_frame = frame

            out = frame.copy()
            with lock:
                results = list(ocr_results)

            # Draw overlays; scale from OCR size to preview size
            for res in results:
                if not isinstance(res, dict):
                    continue
                dt_polys = res.get("dt_polys", [])
                rec_texts = res.get("rec_texts", [])
                rec_scores = res.get("rec_scores", [])
                ocr_w, ocr_h = res.get("size", (out.shape[1], out.shape[0]))

                sx = out.shape[1] / float(ocr_w)
                sy = out.shape[0] / float(ocr_h)

                drawn = 0
                for box, text, score in zip(dt_polys, rec_texts, rec_scores):
                    if score < OCR_SCORE_THRESH:
                        continue
                    if drawn >= MAX_OVERLAYS:
                        break
                    pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in box], dtype=np.int32)
                    cv2.polylines(out, [pts], True, (0, 255, 0), 2)
                    x0, y0 = pts[0]
                    cv2.putText(out, f"{text} ({score:.2f})",
                                (x0, max(20, y0 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    drawn += 1

            cv2.imshow("PaddleOCR Camera (Pi 5)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
                break
    finally:
        running = False
        time.sleep(0.05)
        cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
