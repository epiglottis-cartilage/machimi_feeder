import uln2003
from queue import Queue
import threading

killed = False


def _worker():
    global cmds, dev
    while not killed:
        step = cmds.get(timeout=1)
        if step is None:
            continue
        dev.rotate(step)


def init():
    global dev, cmds, t
    cmds = Queue(maxsize=16)
    dev = uln2003.ULN2003([17, 18, 27, 22], step=2)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def rotate(steps):
    global cmds
    cmds.put(int(steps))


def close():
    global dev, killed, cmds, t
    del dev
    killed = True
    t.join()
    del cmds


def main():
    init()
    print("+")
    rotate(300)
    print("-")
    rotate(-300)
    print(".")
    close()


if __name__ == "__main__":
    main()
