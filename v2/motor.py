import uln2003


def init():
    global dev
    dev = uln2003.ULN2003([17, 18, 27, 22], step=3)


def rotate(steps):
    dev.rotate(steps)


def close():
    global dev
    del dev


def main():
    init()
    print("+")
    rotate(300)
    print("-")
    rotate(-300)
    print(".")


if __name__ == "__main__":
    main()
