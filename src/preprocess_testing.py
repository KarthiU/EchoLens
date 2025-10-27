import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

import re 
import cv2
import numpy as np

def preprocess_storefront(frame):
    # --- 1) Gray-world white balance ---
    b,g,r = cv2.split(frame.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    k = (mean_b + mean_g + mean_r) / 3.0
    b *= (k / (mean_b + 1e-6))
    g *= (k / (mean_g + 1e-6))
    r *= (k / (mean_r + 1e-6))
    wb = np.clip(cv2.merge([b,g,r]), 0, 255).astype(np.uint8)

    # --- 2) Light gamma correction (auto from midtone) ---
    gray_mid = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY).mean() / 255.0
    gamma = 0.8 if gray_mid < 0.45 else 1.2  # brighten shadows or tame glare
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype("uint8")
    wb = cv2.LUT(wb, lut)

    # --- 3) CLAHE on L channel (keeps color info) ---
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    lab_enh = cv2.merge([Lc, A, B])
    enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    # --- 4) High-saturation mask (letters are saturated/orange/etc.) ---
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    # generic: strong saturation + decent brightness
    sat_mask = (S > 80).astype(np.uint8) * 255
    val_mask = (V > 60).astype(np.uint8) * 255
    color_mask = cv2.bitwise_and(sat_mask, val_mask)

    # optional: ORANGE boost (works for Little Caesars)
    # orange1 = cv2.inRange(hsv, (5, 80, 60), (25, 255, 255))
    # color_mask = cv2.bitwise_or(color_mask, orange1)

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 5) Edge-preserving denoise + unsharp ---
    smooth = cv2.bilateralFilter(enh, d=7, sigmaColor=75, sigmaSpace=75)
    sharp = cv2.addWeighted(smooth, 1.5, cv2.GaussianBlur(smooth, (0,0), 1.0), -0.5, 0)

    # --- 6) Black-hat on Value channel to emphasize strokes ---
    V2 = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV)[:,:,2]
    bh = cv2.morphologyEx(V2, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))

    # --- 7) Adaptive binarization on focused region ---
    # Combine structure (bh) with color prior (color_mask)
    focus = cv2.bitwise_and(bh, color_mask)
    focus = cv2.normalize(focus, None, 0, 255, cv2.NORM_MINMAX)
    bin_img = cv2.adaptiveThreshold(focus, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 5)

    # auto invert to "black text on white" for OCR models that prefer it
    if bin_img.mean() > 127:
        bin_img = 255 - bin_img

    # final RGB to feed most OCR models that expect color
    rgb_for_ocr = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    return rgb_for_ocr, bin_img, color_mask


def preprocess_frame(frame):
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE (local contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    #  # 4. Morphological closing to reduce glare/bloom
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    #     # 5. Sharpening to enhance digit edges
    # sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # sharpened = cv2.filter2D(closed, -1, sharpen_kernel)
    
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb


def draw_ocr_results(frame, ocr_results):
    out_img = frame.copy()
    for res in ocr_results:
        if not isinstance(res, dict):
            continue
        dt_polys = res.get("dt_polys", [])
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        for box, text, score in zip(dt_polys, rec_texts, rec_scores):
            # Only draw if text is a 1-3 digit number
            if not re.fullmatch(r"\d{1,3}", text):
                continue
            if score < 0.5:
                continue
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(out_img, [pts], True, (0,255,0), 2)
            x0, y0 = pts[0]
            cv2.putText(out_img, f"{text} ({score:.2f})",
                        (x0, max(20, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return out_img

def run_ocr_on_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return

    # Preprocess
    preprocessed = preprocess_frame(frame)
    preprocessed, _, _ = preprocess_storefront(frame)

    # Run OCR
    ocr = PaddleOCR(
        lang="en", 
        use_textline_orientation=True, 
#text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="PP-OCRv5_mobile_rec",
    ) 
    results = ocr.predict(preprocessed)

    # Draw results
    postprocessed = draw_ocr_results(frame, results)

    # Show images
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Preprocessed")
    plt.imshow(preprocessed)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("OCR Output")
    plt.imshow(cv2.cvtColor(postprocessed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Print OCR results
    print("OCR Results")
    for res in results:
        if not isinstance(res, dict):
            continue
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])
        for text, score in zip(rec_texts, rec_scores):
            print(f"Number: {text}, Score: {score:.2f}")

if __name__ == "__main__":
    image_path = input("Enter image file path: ")
    run_ocr_on_image(image_path)