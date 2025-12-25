import RPi.GPIO as GPIO
import time
from typing import Optional


class HCSR04:
    """
    Driver to use the ultrasonic sensor HC-SR04.
    The sensor range is between 2cm and 4m.

    The timeouts received listening to echo pin are converted
    to Optional(None) meaning 'Out of range'.
    """

    SPEED_OF_SOUND = 34300  # cm/s

    def __init__(self, trigger_pin: int, echo_pin: int, echo_timeout: float = 0.02):
        """
        :param trigger_pin: BCM GPIO pin used for TRIG
        :param echo_pin: BCM GPIO pin used for ECHO
        :param echo_timeout: timeout in seconds (default 20 ms)
        """
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.echo_timeout = echo_timeout

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

        GPIO.output(self.trigger_pin, False)
        time.sleep(0.05)  # sensor settle time

    def _send_trigger_pulse(self) -> None:
        """Send a 10us trigger pulse."""
        GPIO.output(self.trigger_pin, True)
        time.sleep(10e-6)
        GPIO.output(self.trigger_pin, False)

    def _wait_for_echo(self, value: int) -> Optional[float]:
        """
        Wait for echo pin to become `value`.
        Returns timestamp or None on timeout.
        """
        timeout = time.perf_counter() + self.echo_timeout
        while GPIO.input(self.echo_pin) != value:
            if time.perf_counter() > timeout:
                return None
        return time.perf_counter()

    def distance_cm(self) -> Optional[float]:
        """
        Measure distance in centimeters.

        :return: distance in cm, or None if out of range / timeout
        """
        self._send_trigger_pulse()

        start = self._wait_for_echo(1)
        if start is None:
            return None

        end = self._wait_for_echo(0)
        if end is None:
            return None

        pulse_duration = end - start
        distance = (pulse_duration * self.SPEED_OF_SOUND) / 2

        # HC-SR04 valid range check
        if distance < 2 or distance > 400:
            return None

        return round(distance, 2)

    def cleanup(self) -> None:
        """Release GPIO resources."""
        GPIO.cleanup((self.trigger_pin, self.echo_pin))

    def __del__(self):
        self.cleanup()
