import time
import cv2
from ultralytics.models.yolo import YOLO

HUMAN = 0
CAT = 15


def init(cam=0):
    global model, cap
    print("loading YOLO ...")
    model = YOLO("./yolov8n_ncnn_model")
    print("Opening camera ...")
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, 10)


def close():
    cap.release()


def can_see(objs):
    global scan_angle, scan_direction, model, cap

    ret = False
    frame = None
    while not ret:
        ret, frame = cap.read()

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
    cam = 3  # int(sys.argv[1]) if len(sys.argv) > 1 else 0
    init(cam)
    while True:
        time.sleep(0.1)
        print(can_see([CAT, HUMAN]))


if __name__ == "__main__":
    main()
