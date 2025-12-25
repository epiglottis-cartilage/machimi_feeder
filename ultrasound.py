from hscr04 import HCSR04


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
    return distance


def main():
    init()
    try:
        while True:
            print(read())
    except Exception:
        pass
    close()
