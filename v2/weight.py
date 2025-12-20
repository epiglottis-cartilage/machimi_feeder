import time
from hx711 import HX711


def init():
    global dev
    dev = HX711(5, 6)
    dev.setReferenceUnit(100_000)


def read():
    raw = dev.getRawBytes()
    weight = dev.rawBytesToWeight(raw)
    return weight


def main():
    init()
    while True:
        time.sleep(0.1)
        print(read())


if __name__ == "__main__":
    main()
