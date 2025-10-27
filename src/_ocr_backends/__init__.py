from .paddle_backend import PaddleOCRBackend
try:
    from .tflite_backend import TFLiteOCRBackend
except Exception:
    TFLiteOCRBackend = None

__all__ = ["PaddleOCRBackend", "TFLiteOCRBackend"]


