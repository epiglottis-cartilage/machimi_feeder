import cv2
import numpy as np
import onnxruntime as ort

# ================= 配置 =================
MODEL_PATH = "yolov8n.onnx"
VIDEO_DEVICE = 0
INPUT_SIZE = 640
CONF_THRES = 0.3
# =======================================


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    # --- load model ---
    print("[INFO] loading model...")
    sess = ort.InferenceSession(MODEL_PATH)

    # --- open camera ---
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 12)

    print("[INFO] press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] camera read failed")
            continue

        h0, w0 = frame.shape[:2]

        # -------- preprocess --------
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        blob = img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        # -------- inference --------
        pred = sess.run(None, {"images": blob})[0][0]
        # pred shape: [84, 8400]

        boxes = pred[:4, :]  # cx, cy, w, h
        obj = sigmoid(pred[4, :])  # objectness
        cls = sigmoid(pred[5:, :])  # class scores

        cls_ids = np.argmax(cls, axis=0)
        cls_scores = cls[cls_ids, np.arange(cls.shape[1])]
        scores = obj * cls_scores

        # -------- draw --------
        for i in range(scores.shape[0]):
            if scores[i] < CONF_THRES:
                continue

            cx, cy, w, h = boxes[:, i]

            x1 = int((cx - w / 2) * w0 / INPUT_SIZE)
            y1 = int((cy - h / 2) * h0 / INPUT_SIZE)
            x2 = int((cx + w / 2) * w0 / INPUT_SIZE)
            y2 = int((cy + h / 2) * h0 / INPUT_SIZE)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_ids[i]}:{scores[i]:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            break

        cv2.imshow("YOLOv8 ONNX Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
