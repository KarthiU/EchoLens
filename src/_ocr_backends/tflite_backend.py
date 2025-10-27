class TFLiteOCRBackend:
    def __init__(self):
        raise NotImplementedError("TFLite backend not set up. Use Paddle backend or add models.")

    def run(self, bgr_image):
        return []