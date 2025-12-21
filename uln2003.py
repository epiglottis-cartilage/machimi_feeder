import RPi.GPIO as GPIO
import time


class ULN2003:
    """
    ULN2003 四相步进电机驱动类
    适用于 28BYJ-48 等 4 相步进电机
    """

    # 半步序列
    HALF_STEP_SEQ = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    def __init__(self, pins, delay=0.002, step=2):
        """
        pins  : [IN1, IN2, IN3, IN4]，BCM 编号
        delay : 每个半步的延时（秒）
        """
        if len(pins) != 4:
            raise ValueError("pins must be a list of 4 GPIO numbers")

        self.pins = pins
        self.delay = delay
        self.step = step

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

    # ---------------- 基本动作 ----------------
    def _output(self, pattern):
        for pin, value in zip(self.pins, pattern):
            GPIO.output(pin, value)

    def rotate(self, steps):
        """
        steps     : 步数, + = 正转, - = 反转
        """
        sequence = self.HALF_STEP_SEQ
        if steps < 0:
            sequence = list(reversed(sequence))

        sequence = sequence[:: self.step]

        for _ in range(abs(steps)):
            for pattern in sequence:
                self._output(pattern)
                time.sleep(self.delay)

    def release(self):
        """释放线圈，防止电机发热"""
        for pin in self.pins:
            GPIO.output(pin, 0)

    def cleanup(self):
        """释放 GPIO 资源"""
        self.release()
        GPIO.cleanup(self.pins)

    # ---------------- 析构保护 ----------------
    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
