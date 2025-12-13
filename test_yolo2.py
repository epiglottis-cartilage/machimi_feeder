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


def nms_xyxy(boxes, scores, iou_thres=0.5):
    def iou_xyxy(box, boxes):
        """
        box:  (4,)   [x1,y1,x2,y2]
        boxes:(N,4)
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area1 + area2 - inter + 1e-6)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thres]

    return keep


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

        boxes = pred[:4, :]
        scores = sigmoid(pred[4, :])
        classes = sigmoid(pred[5:, :])

        cls_ids = np.argmax(classes, axis=0)
        cls_scores = classes[cls_ids, range(classes.shape[1])]
        final_scores = scores * cls_scores

        mask = final_scores >= 0.25
        cx, cy, w, h = boxes[:, mask]
        if len(cx) == 0:
            return False

        # --- xywh -> xyxy ---
        x1 = (cx - w / 2) * frame.shape[1] / 640
        y1 = (cy - h / 2) * frame.shape[0] / 640
        x2 = (cx + w / 2) * frame.shape[1] / 640
        y2 = (cy + h / 2) * frame.shape[0] / 640

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        scores_nms = final_scores[mask]

        # --- NMS ---
        keep = nms_xyxy(boxes_xyxy, scores_nms, iou_thres=0.5)

        # -------- draw --------
        for i in keep:
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
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
