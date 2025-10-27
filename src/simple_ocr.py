import cv2
import threading
import time
import numpy as np

from paddleocr import PaddleOCR

# For frame similarity check
from skimage.metrics import structural_similarity as ssim


PROMPTS = [ 

    "Hey EchoLens, what bus is in front of me?", # bus 3 lens 
    "Hey EchoLens, read and summarize document in front of me",         # 
    "Hey EchoLens, what is that sign in front of me",       # 


]


latest_frame = None
ocr_results = []
lock = threading.Lock()
running = True


def capture_frame():
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        return frame
    else:
        return None

def run_ocr_on_frame(frame):
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)
    large = cv2.resize(frame, (960, 540))
    rgb = cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB)
    results = ocr.predict(rgb)
    return results

def preprocess_frame(frame):
    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

def preprocess_frame_bus(frame):
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE (local contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
     # 4. Morphological closing to reduce glare/bloom
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 5. Sharpening to enhance digit edges
    sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpened = cv2.filter2D(closed, -1, sharpen_kernel)
    
    rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return rgb

# ...existing code...

def ocr_worker():
    global latest_frame, ocr_results, running
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)
    prev_gray = None
    while running:
        frame = None
        with lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is not None:
            # Resize and preprocess for comparison
            large = cv2.resize(frame, (960, 540))
            gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
            # Compare with previous frame
            if prev_gray is not None:
                similarity = ssim(gray, prev_gray)
                if similarity > 0.8:
                    time.sleep(0.1)
                    continue  # Skip OCR if frames are too similar
            prev_gray = gray
            print("[OCR Thread] Running OCR")
            rgb = preprocess_frame(large)
            results = ocr.predict(rgb)
            with lock:
                ocr_results = results
        time.sleep(0.1)  # Avoid busy loop

def main():
    global latest_frame, ocr_results, running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    t = threading.Thread(target=ocr_worker, daemon=True)
    t.start()

    print("✅ Camera started — press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        with lock:
            latest_frame = frame

        # Overlay OCR results
        with lock:
            results = ocr_results
        count = 0
        for res in results:
            if not isinstance(res, dict):
                continue
            dt_polys = res.get("dt_polys", [])
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            for box, text, score in zip(dt_polys, rec_texts, rec_scores):
                if score < 0.8:  # Higher threshold for cleaner results
                    continue
                if count >= 3:   # Show only top 5 results
                    break
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0,255,0), 2)
                x0, y0 = pts[0]
                cv2.putText(frame, f"{text} ({score:.2f})",
                            (x0, max(20, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                print(f"OCR: {text} ({score:.2f})")
                count += 1

        cv2.imshow("PaddleOCR Camera (Threaded)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()