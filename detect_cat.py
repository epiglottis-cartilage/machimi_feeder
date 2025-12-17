import time
import cv2
import onnxruntime as ort
import numpy as np
import sounddevice as sd
import queue
import csv
import math
import tflite_runtime.interpreter as tflite

try:
    import RPi.GPIO as GPIO

    HAS_PI = True
except:
    HAS_PI = False

############################
# 配置区域
############################

VIDEO_DEVICE = 0

# YOLO
MODEL_PATH = "yolov8n.onnx"
CONF_THRES = 0.3
DETECT_CLASS = 14

# HC-SR04 GPIO
TRIG_PIN = 23
ECHO_PIN = 24

# 28BYJ-48 GPIO（IN1~IN4）
STEPPER_PINS = [17, 18, 27, 22]

# 音频参数
AUDIO_DEVICE_INDEX = 1  # 用 arecord -l 查看
AUDIO_THRESHOLD = 500  # 声音能量阈值（可调）

DISTANCE_THRESHOLD_CM = 10
SLEEP_AFTER_FEED = 60

############################
# GPIO 初始化
############################
if HAS_PI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    for pin in STEPPER_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

############################
# 步进电机参数
############################

HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

STEPS_PER_REV = 4096
STEPS_180 = STEPS_PER_REV // 2

############################
# 功能函数
############################

mod_yolo = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
dev_cap = cv2.VideoCapture(VIDEO_DEVICE)
if not dev_cap.isOpened():
    raise RuntimeError("Camera open failed")

dev_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
dev_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
dev_cap.set(cv2.CAP_PROP_FPS, 12)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def detect_cat():
    global dev_cap, mod_yolo

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

    ret, frame = dev_cap.read()
    if not ret:
        return False
    img = cv2.resize(frame, (640, 640))

    blob = img.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None]

    pred = mod_yolo.run(None, {"images": blob})[0][0]

    boxes = pred[:4, :]
    scores = sigmoid(pred[4, :])
    classes = sigmoid(pred[5:, :])

    cls_ids = np.argmax(classes, axis=0)
    cls_scores = classes[cls_ids, range(classes.shape[1])]
    final_scores = scores * cls_scores

    mask = (cls_ids == DETECT_CLASS) & (final_scores > CONF_THRES)
    cx, cy, w, h = boxes[:, mask]
    if len(cx) == 0:
        return False

    # # --- xywh -> xyxy ---
    # x1 = (cx - w / 2) * frame.shape[1] / 640
    # y1 = (cy - h / 2) * frame.shape[0] / 640
    # x2 = (cx + w / 2) * frame.shape[1] / 640
    # y2 = (cy + h / 2) * frame.shape[0] / 640

    # boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    # scores_nms = final_scores[mask]

    # # --- NMS ---
    # keep = nms_xyxy(boxes_xyxy, scores_nms, iou_thres=0.5)

    # # --- output ---
    # for i in keep:
    #     xi1, yi1, xi2, yi2 = boxes_xyxy[i]
    #     print(
    #         f"[cat] {scores_nms[i]:.2f} | ({int(xi1)},{int(yi1)},{int(xi2)},{int(yi2)})"
    #     )

    return True


def get_distance_cm():
    if not HAS_PI:
        return 1000
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.05)

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        start = time.time()

    end = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        end = time.time()

    duration = end - start
    distance = duration * 34300 / 2
    return distance


# 配置参数（需根据实际情况调整）
SAMPLE_RATE = 16000
WINDOW_SIZE = 15600  # 1.0 s
HOP_SIZE = 8000  # 0.5 s
THRESHOLD = 0.2
CONSECUTIVE_HITS = 3  # 连续命中次数

MODEL_PATH = "yamnet.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"

# 加载模型
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 加载类别映射
class_names = []
with open(CLASS_MAP_PATH) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        class_names.append(row[2])

cat_indices = [
    i
    for i, name in enumerate(class_names)
    if "cat" in name.lower() or "meow" in name.lower()
]

print("Cat-related class indices:", cat_indices)
for i in cat_indices:
    print(f"  {i}: {class_names[i]}")

# 音频队列
audio_q = np.zeros(0, dtype=np.float32)


def audio_callback(indata, frames, time_info, status):
    global audio_q
    if status:
        print(status)
    start = max((len(audio_q) - WINDOW_SIZE * 2, 0))
    audio_q = np.concatenate([audio_q[start:], indata[:, 0].copy()])


# 推理函数
def yamnet_infer(waveform: np.ndarray) -> np.ndarray:
    """
    waveform: (N,) float32
    return: scores (frames, 521)
    """
    interpreter.set_tensor(input_details[0]["index"], waveform)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])
    return scores


dev_sound = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
dev_sound.start()


def detect_sound():
    buffer = audio_q
    hit_count = 0

    # 滑窗
    while len(buffer) >= WINDOW_SIZE:
        chunk = buffer[:WINDOW_SIZE]
        buffer = buffer[HOP_SIZE:]

        scores = yamnet_infer(chunk)
        cat_score = scores[:, cat_indices].max()

        if cat_score > THRESHOLD:
            hit_count += 2
        else:
            hit_count = max(0, hit_count - 1)

        # print(f"cat_score={cat_score:.3f}, hit={hit_count}")

        if hit_count >= CONSECUTIVE_HITS:
            return True
    return hit_count


def step_motor(steps, delay=0.001, reverse=False):
    if not HAS_PI:
        return
    seq = HALF_STEP_SEQ[::-1] if reverse else HALF_STEP_SEQ
    for _ in range(steps):
        for step in seq:
            for pin, val in zip(STEPPER_PINS, step):
                GPIO.output(pin, val)
            time.sleep(delay)


def feed():
    print("[Motor] Feeding...")
    step_motor(STEPS_180)
    time.sleep(1)
    step_motor(STEPS_180, reverse=True)
    print("[Motor] Done")


############################
# 主循环
############################


def main():
    try:
        while True:
            cat_in_sight = detect_cat()
            cat_meowing = detect_sound()
            cat_distance = get_distance_cm()

            print(f"{cat_in_sight=}, {cat_meowing=}, {cat_distance=}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing")
        if HAS_PI:
            GPIO.cleanup()
        try:
            dev_cap.release()
            dev_sound.close()
        except:
            pass


if __name__ == "__main__":
    main()
