import time
from hx711 import HX711


def init():
    global dev
    dev = HX711(5, 6)
    dev.reset()
    dev.setReferenceUnit(1_000)


def close():
    global dev
    del dev


def read_avg(n=3):
    return sum((read() for _ in range(n))) / n


def read():
    def fix(x):
        return (x - 535.5) * (235 / 96.83)

    raw = dev.getRawBytes()
    weight = dev.rawBytesToWeight(raw)
    return fix(weight)


def main():
    init()
    while True:
        time.sleep(0.1)
        print(read_avg(3))


if __name__ == "__main__":
    main()
