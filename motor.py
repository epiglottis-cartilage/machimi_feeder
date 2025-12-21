from .uln2003 import ULN2003
import queue
import threading

killed = False


def _worker():
    global cmds, dev
    while True:
        try:
            step = cmds.get(timeout=1)
            dev.rotate(step)
        except:
            pass
        finally:
            if killed:
                break


def init():
    global dev, cmds, t
    cmds = queue.Queue(maxsize=16)
    dev = ULN2003([17, 18, 27, 22], step=2)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def rotate(steps, nonblock=True):
    global cmds
    if nonblock:
        cmds.put(int(steps))
    else:
        dev.rotate(steps)


def close():
    global dev, killed, cmds, t
    killed = True
    t.join()
    del dev
    del cmds


def main():
    init()
    try:
        while True:
            n = int(input(">"))
            rotate(n)
    except KeyboardInterrupt:
        pass

    close()


if __name__ == "__main__":
    main()
