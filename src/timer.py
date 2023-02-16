import os
import timeit

from typing import Callable


class PieceTimer:
    """A timer to measure a piece of code's execution.
    Funcs:
        __call__: Return the passed time in [sec].
        reset: Reset the timer to the current time.
    """
    def __init__(self) -> None:
        self._instant = timeit.default_timer()

    def __call__(self, round_decimals:int=4) -> float:
        return round(timeit.default_timer()-self._instant, round_decimals)

    def reset(self):
        self._instant = timeit.default_timer()


class LoopTimer:
    """A timer for loop.
    Properties:
        t: (Read only) The current time.
        k: (Read only) The current time step.
    """
    __EXIST_TIMERS = []

    def __init__(self, sampling_time:float, time_out:float, timer_id:int, start_time=0.0) -> None:
        """All temporal variables are in the unit [second] if not specified.
        Args:
            sampling_time: Sampling time.
            time_out: Maximal permitted time. If time>time_out, push warning or raise error.
            timer_id: The ID of the timer. Timers cannot have the same ID.
            start_time: The start time, default 0.
        """
        if timer_id in self.__EXIST_TIMERS:
            raise ValueError(f"Timer ID {timer_id} exists!")
        self.__EXIST_TIMERS.append(timer_id)
        self._id = timer_id

        self._ts = sampling_time
        self._time_out = time_out

        self._t = start_time # current time
        self._k = 0 # current time step

        self.running_time = []
        self._running_timer = PieceTimer()

    def __call__(self, running_function:Callable, *args, **kwargs):
        self._t += self._ts
        self._k += 1
        if self._t > self._time_out:
            raise TimeoutError("Time out!")
        self._running_timer.reset()
        output = running_function(*args, **kwargs)
        self.running_time.append(self._running_timer())
        return output

    @property
    def timer_id(self):
        return self._id

    @property
    def k(self):
        return self._k

    @property
    def t(self):
        return self._t


if __name__ == "__main__":
    import time
    def add(a, b):
        time.sleep(0.2)
        return a+b
    lt = LoopTimer(0.2, 2, 1)
    try:
        while 1:
            output = lt(add, 1, 1)
            print(lt.running_time)
    except:
        print(output)
