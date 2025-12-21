import time
from v2 import sight, hearing, weight, motor
from v2.sight import HUMAN, CAT


if __name__ == "__main__":
    sight.init()
    hearing.init()
    weight.init()
    motor.init()
    try:
        while True:
            # print(sight.can_see(), hearing.meow(), weight.read_avg())
            if sight.can_see([HUMAN, CAT]):
                motor.rotate(450, nonblock=False)
                motor.rotate(-450, nonblock=False)
                time.sleep(10)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    sight.close()
    hearing.close()
    weight.close()
    motor.close()
