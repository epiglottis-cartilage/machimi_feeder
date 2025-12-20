import time
from v2 import sight, hearing


if __name__ == "__main__":
    sight.init()
    hearing.init()
    try:
        while True:
            print(sight.can_see(), hearing.meow())
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    sight.close()
    hearing.close()
