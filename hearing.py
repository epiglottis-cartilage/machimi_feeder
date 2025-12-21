import time
import numpy as np
import sounddevice as sd
import queue
import csv
import tflite_runtime.interpreter as tflite


# 配置参数（需根据实际情况调整）
SAMPLE_RATE = 16000
WINDOW_SIZE = 15600  # 1.0 s
HOP_SIZE = 8000  # 0.5 s
THRESHOLD = 0.1
CONSECUTIVE_HITS = 3  # 连续命中次数
CONSECUTIVE_SCORE = 5

MODEL_PATH = "yamnet.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"
# 加载模型


def init():
    global model, input_details, output_details, audio_q, cat_indices, buffer, dev_sound
    model = tflite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

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
    audio_q = queue.Queue(maxsize=8000)

    dev_sound = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=audio_callback
    )
    dev_sound.start()
    buffer = np.zeros(0, dtype=np.float32)


def close():
    global dev_sound
    dev_sound.close()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    if audio_q.full():
        audio_q.get()
        print("pop")
    audio_q.put(indata[:, 0].copy())


# 推理函数
def yamnet_infer(waveform: np.ndarray) -> np.ndarray:
    """
    waveform: (N,) float32
    return: scores (frames, 521)
    """
    global model

    model.set_tensor(input_details[0]["index"], waveform)
    model.invoke()
    scores = model.get_tensor(output_details[0]["index"])
    return scores


def meow():
    global buffer
    q = [buffer]
    while not audio_q.empty():
        q.append(audio_q.get())
    buffer = np.concatenate(q)
    hit_count = 0
    score = 0
    # print(len(buffer))
    # 滑窗
    while len(buffer) >= WINDOW_SIZE:
        chunk = buffer[:WINDOW_SIZE]
        buffer = buffer[HOP_SIZE:]

        scores = yamnet_infer(chunk)
        cat_score = scores[:, cat_indices].max()

        if cat_score > THRESHOLD:
            hit_count += 1
            score += 5
        else:
            score = max(0, score - 1)

        # print(f"cat_score={cat_score:.3f}, hit={hit_count}")

        if hit_count >= CONSECUTIVE_HITS or score >= CONSECUTIVE_SCORE:
            return True
    return hit_count


def main():
    init()
    while True:
        time.sleep(0.1)
        print(meow())


if __name__ == "__main__":
    main()
