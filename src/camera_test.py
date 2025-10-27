import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Could not open camera.")
        return

    print("Camera started — press 'q' to quit.")

    prev_time = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            curr_time = time.time()
            fps = 30 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.setWindowTitle("Camera Test", f"Camera Test - FPS: {fps:.2f}")

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()