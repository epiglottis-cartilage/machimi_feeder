import time
import sight
import weight
import motor
import beep
import ultrasound

# import hearing
from sight import HUMAN, CAT


if __name__ == "__main__":
    sight.init()
    # hearing.init()
    weight.init()
    motor.init()
    ultrasound.init()
    beep.init()
    try:
        while True:
            w = weight.read_avg()
            print(f"{w - 288.5}g remain...")
            if w - 288.5 < 40:
                beep.beep()
                time.sleep(3)
                continue

            if not ultrasound.nearby():
                time.sleep(1)
                continue

            step = 430 if w < 600 else 400

            # print(sight.can_see(), hearing.meow(), weight.read_avg())
            if sight.can_see([HUMAN, CAT]):
                print("I seeeee you, feeding...")
                motor.rotate(-step)
                motor.rotate(step)
                time.sleep(10)

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    sight.close()
    # hearing.close()
    weight.close()
    motor.close()
    ultrasound.close()
    beep.close()
