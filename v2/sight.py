import time
import cv2
from ultralytics.models.yolo import YOLO
import threading

HUMAN = 0
CAT = 15


latest_frame = None
frame_lock = threading.Lock()
running = True


def init(cam=0):
    global model, cap
    model = YOLO("./yolov8n_ncnn_model")

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, 10)

    # 启动采集线程
    t = threading.Thread(target=camera_thread, args=(cap,), daemon=True)
    t.start()


def close():
    cap.release()


def camera_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame


def can_see(objs):
    global scan_angle, scan_direction, model

    with frame_lock:
        if latest_frame is None:
            return None
        frame = latest_frame.copy()

    # YOLO 推理（检测猫=15）
    starttime = time.time()
    results = model(frame, classes=objs, conf=0.5)
    result = results[0]
    starttime = time.time() - starttime
    # print(f"eval time: {starttime * 1000:.1f} ms")

    # detected = len(result.boxes) > 0
    biggest_size = 0

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 - x1 > biggest_size:
            biggest_size = x2 - x1

    return biggest_size


def main():
    cam = 0  # int(sys.argv[1]) if len(sys.argv) > 1 else 0
    init(cam)
    while True:
        time.sleep(0.1)
        print(can_see())


if __name__ == "__main__":
    main()
