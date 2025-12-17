import numpy as np
import sounddevice as sd
import queue
import csv
import time
import tflite_runtime.interpreter as tflite

# ======================
# 参数配置
# ======================
SAMPLE_RATE = 15600
WINDOW_SIZE = 15600  # 1.0 s
HOP_SIZE = 8000  # 0.5 s
THRESHOLD = 0.2
CONSECUTIVE_HITS = 2  # 连续命中次数

MODEL_PATH = "yamnet.tflite"
CLASS_MAP_PATH = "yamnet_class_map.csv"

# ======================
# 加载模型
# ======================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ======================
# 加载类别映射
# ======================
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

# ======================
# 音频队列
# ======================
audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata[:, 0].copy())


# ======================
# 推理函数
# ======================
def yamnet_infer(waveform: np.ndarray) -> np.ndarray:
    """
    waveform: (N,) float32
    return: scores (frames, 521)
    """
    interpreter.set_tensor(input_details[0]["index"], waveform)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])
    return scores


# ======================
# 主循环
# ======================
def main():
    buffer = np.zeros(0, dtype=np.float32)
    hit_count = 0

    print("🎤 Listening for cat sounds...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while True:
            # 取音频
            data = audio_q.get()
            buffer = np.concatenate([buffer, data])

            # 滑窗
            while len(buffer) >= WINDOW_SIZE:
                chunk = buffer[:WINDOW_SIZE]
                buffer = buffer[HOP_SIZE:]

                scores = yamnet_infer(chunk)
                cat_score = scores[:, cat_indices].max()

                if cat_score > THRESHOLD:
                    hit_count += 1
                else:
                    hit_count = max(0, hit_count - 1)

                print(f"cat_score={cat_score:.3f}, hit={hit_count}")

                if hit_count >= CONSECUTIVE_HITS:
                    print("🐱🐱🐱 CAT DETECTED 🐱🐱🐱")
                    hit_count = 0

                time.sleep(0.01)


if __name__ == "__main__":
    main()
