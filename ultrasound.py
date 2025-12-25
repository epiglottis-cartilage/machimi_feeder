from hscr04 import HCSR04
import time


def init():
    global dev
    dev = HCSR04(echo_pin=24, trigger_pin=16)


def close():
    global dev
    del dev


def read_avg(n=4):
    return sum(sorted((read() for _ in range(n)))[1:-1]) / (n - 2)


def read():
    distance = dev.distance_cm() or 500.0
    time.sleep(0.06)
    return distance


def nearby():
    return read_avg() < 40.0


def main():
    init()
    try:
        while True:
            print(read())
    except KeyboardInterrupt:
        pass
    close()


if __name__ == "__main__":
    main()
