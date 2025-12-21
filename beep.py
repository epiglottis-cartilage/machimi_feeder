import RPi.GPIO as GPIO
import time


class Beep:
    def __init__(self, pin):
        self.pin = pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    def beep(self):
        GPIO.output(self.pin, 1)
        time.sleep(1)
        GPIO.output(self.pin, 0)

    def cleanup(self):
        """释放 GPIO 资源"""
        GPIO.output(self.pin, 0)
        GPIO.cleanup([self.pin])

    # ---------------- 析构保护 ----------------
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def init():
    global dev
    dev = Beep(26)


def beep():
    dev.beep()


def close():
    global dev
    del dev


if __name__ == "__main__":
    init()
    beep()
    close()
