from __future__ import annotations
from typing import List, Dict, Any
import cv2
import numpy as np
from paddleocr import PaddleOCR

class PaddleOCRBackend:
    """
    Updated PaddleOCR wrapper using the modern `predict()` API.
    Compatible with PaddleOCR ≥ 2.7 (2024–2025 releases).

    Output format (list of dicts):
        {"text": str, "conf": float, "bbox": (x1, y1, x2, y2)}
    """
    def __init__(self, lang: str = "en", enable_angle_cls: bool = True):
        # Build PaddleOCR model with detection, recognition, and (optional) angle classifier.
        # The latest API automatically initializes internal components; no deprecated args used.
        self.ocr = PaddleOCR(lang=lang, use_angle_cls=enable_angle_cls)
        self.enable_angle_cls = enable_angle_cls

    def run(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run OCR and return a list of {text, conf, bbox} dicts."""
        if bgr_image is None or not isinstance(bgr_image, np.ndarray):
            return []

        # Convert BGR → RGB (Paddle expects RGB)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # === Modern API ===
        # predict() returns a dict with keys like 'det', 'rec', and 'cls'
        result = self.ocr.predict(rgb_image)

        items: List[Dict[str, Any]] = []

        if not result or "boxes" not in result or "rec_scores" not in result:
            return items

        boxes = result.get("boxes", [])          # list of 4-point boxes
        texts = result.get("rec_texts", [])      # recognized text strings
        scores = result.get("rec_scores", [])    # confidence scores (floats)

        for box, text, conf in zip(boxes, texts, scores):
            if not box or not text:
                continue

            # Each box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            items.append({
                "text": text.strip(),
                "conf": float(conf),
                "bbox": (x1, y1, x2, y2),
            })

        return items
