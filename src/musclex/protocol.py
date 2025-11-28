import scipy


class Event:
    def __init__(self, start_time, end_time, exercise):
        """Event of either growth or exercise.

        Args:
            start_time (float): Event start time
            end_time (float): Event end time
            exercise (bool): True if event is exercise, False if growth
        """
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.exercise: bool = exercise
        self.no: int = None
        self.solution: scipy.integrate._ivp.ivp.OdeResult = None


class ExerciseEvent(Event):
    def __init__(self, start_time, end_time, beta=2 / 50, I=0.8, F_max=150):
        """Event of either growth or exercise.

        Args:
            start_time (float): Event start time
            end_time (float): Event end time
            beta (float, optional): Scale factor. Defaults to 2/50 (hours*kgf)^-1.
            I (float, optional): Fraction of maximum force. Defaults to 0.8.
            F_max (int, optional): Maximum force. Defaults to 150 kgf.
        """
        super().__init__(start_time, end_time, True)
        self.beta: float = beta
        self.I: float = I
        self.F_max: float = F_max


class GrowthEvent(Event):
    def __init__(self, start_time, end_time):
        super().__init__(start_time, end_time, False)


class ExerciseProtocol:
    def __init__(self):
        """Protocol of exercise and growth periods."""
        self.events: list[Event] = list()
        self.numEvents = 0

    def add_event(
        self,
        event: Event,
    ):
        """Add an event to the protocol.

        Args:
            event (Event): Growth or exercise period
        """
        event.no = self.numEvents
        self.events.append(event)
        self.numEvents += 1

    def get_previous_exercise_start_time(
        self,
        t: float,
    ):
        t_ex = None
        for event in self.events:
            if event.start_time <= t and event.exercise:
                t_ex = event.start_time
        return t_ex

    def verify(self):
        """Verify the exercise protocol.
        Events should be in chronological order, non-overlapping, and with no gaps in between.
        """
        for i in range(len(self.events) - 1):
            if self.events[i].end_time > self.events[i + 1].start_time:
                raise ValueError("Events {} and {} overlap".format(i, i + 1))
            if self.events[i].end_time < self.events[i + 1].start_time:
                raise ValueError("Gap between events number {} and {}".format(i, i + 1))
            if self.events[i].exercise == self.events[i + 1].exercise:
                raise ValueError(
                    "Events {} and {} are of the same type".format(i, i + 1)
                )


class RegularExercise(ExerciseProtocol):
    def __init__(
        self,
        N: int,
        exercise_duration: float,
        growth_duration: float,
        end_time: float,
        intensity: float=0.8,
        initial_rest: float = 0.0,
    ):
        """Exercise protocol with num_exercise_sessions exercise events of exercise_duration hours each, and
        growth_duration hours of growth in between.

        Args:
            N (int): Number of exercise sessions
            exercise_duration (int): Duration of each exercise session (hours)
            growth_duration (int): Duration of each growth period (hours)
            end_time (int): End time of protocol (hours)
            intensity (float): Intensity of exercise (fraction of maximum force)
        """
        super().__init__()

        if initial_rest > 0:
            self.add_event(GrowthEvent(0, initial_rest))
            t0 = initial_rest
        else:
            t0 = 0

        for i in range(N):
            self.add_event(
                ExerciseEvent(
                    t0 + i * (growth_duration + exercise_duration),
                    t0 + i * growth_duration + (i + 1) * exercise_duration,
                    I=intensity,
                )
            )
            self.add_event(
                GrowthEvent(
                    t0 + i * growth_duration + (i + 1) * exercise_duration,
                    t0 + (i + 1) * (growth_duration + exercise_duration),
                )
            )
        # Replace the final growth period with a longer one
        self.events.pop()
        self.numEvents -= 1
        self.add_event(
            GrowthEvent(t0 + i * growth_duration + (i + 1) * exercise_duration, end_time)
        )


class SingleExercise(ExerciseProtocol):
    def __init__(
        self,
        exercise_start: float = 24,
        exercise_duration: float = 1,
        end_time: float = 60,
    ):
        """Exercise protocol with a single exercise session.

        Args:
            exercise_start (int, optional): Start time of the session. Defaults to 24.
            exercise_duration (int, optional): Duration of the session. Defaults to 1.
            end_time (int, optional): End time of protocol. Defaults to 60.
        """

        super().__init__()

        self.add_event(GrowthEvent(0, exercise_start))
        self.add_event(
            ExerciseEvent(exercise_start, exercise_start + exercise_duration)
        )
        self.add_event(GrowthEvent(exercise_start + exercise_duration, end_time))


class DeFreitasProtocol(ExerciseProtocol):
    def __init__(self):
        """Exercise protocol matching the experiments in De Freitas et al 2011.

        First two weeks without exercise, then exercise for one hour at 80% intensity
        on days 1, 3, and 5 every week the following 8 weeks.
        24 exercise sessions in total.
        """
        super().__init__()
        t_ex = 24 * 7  # start of first exercise session

        self.add_event(GrowthEvent(0, 24 * 7))  # one resting week prior to exercise
        for w in range(8):
            week = 7 * 24 * w + t_ex  # start of the week
            day = 24
            self.add_event(ExerciseEvent(week, week + 1))  # day 1
            self.add_event(GrowthEvent(week + 1, week + 2 * day))
            self.add_event(ExerciseEvent(week + 2 * day, week + 2 * day + 1))  # day 3
            self.add_event(GrowthEvent(week + 2 * day + 1, week + 4 * day))
            self.add_event(ExerciseEvent(week + 4 * day, week + 4 * day + 1))  # day 5
            self.add_event(GrowthEvent(week + 4 * day + 1, week + 7 * day))


class RestProtocol(ExerciseProtocol):
    def __init__(self, end_time=24*7*5):
        """Exercise protocol with no exercise.
        
        Args:
            end_time (int, optional): Duration of the protocol in hours. Defaults to 5 weeks.
        """
        super().__init__()
        self.add_event(GrowthEvent(0, end_time))
