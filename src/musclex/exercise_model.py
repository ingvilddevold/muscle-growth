import dolfinx
from musclex.protocol import Event, ExerciseProtocol
import scipy
import yaml
import numpy as np
from pathlib import Path


class ExerciseModel:
    def __init__(
        self,
        protocol: ExerciseProtocol,
        conf_file: str,
        results_path: str = None,
    ):
        """Model of the IGF1-AKT-FOXO-mTOR exercise signaling network.

        Args:
            protocol (ExerciseProtocol): Exercise protocol with training and growth periods
            parameter_file_path (str): Path to parameter config file
            dt (float): Time step for ODE solver
            results_path (str): Path to folder where results will be saved. Default is None.
        """
        self.protocol = protocol
        protocol.verify()
        self.results_path = results_path

        # Read parameters from file
        with open(conf_file) as config:
            conf = yaml.load(config, Loader=yaml.FullLoader)
            self.params = conf["exercise_parameters"]

        # Initial conditions
        self.y0 = (
            self.params["x1_0"],
            self.params["x2_0"],
            self.params["x3_0"],
            self.params["x4_0"],
            self.params["z_0"],
        )
        # Keep track of previous state for continuity across events
        self.y_prev = self.y0

        # ODE time step
        self.dt = self.params["dt"]

        # Make results folder if it does not exist
        if results_path:
            Path(results_path).mkdir(parents=True, exist_ok=True)

    def set_protein_synthesis_rate(self, k1: float):
        """Update the protein synthesis rate k1.
        Used in feedback from mechanical model."""
        if k1 < 0:
            raise ValueError("k1 must be non-negative.")
        self.params["k1"] = k1

    def get_duration(self):
        """Find the start and end time of the protocol.
        Assumes that the protocol events are sorted by increasing start time."""
        T0 = self.protocol.events[0].start_time
        Tf = self.protocol.events[-1].end_time
        return T0, Tf

    def a1(
        self,
        t: float,
    ):
        """Intrinsic growth rate of x1 / IGF1.

        Args:
            t (float): Time

        Returns:
            float: a1(t)
        """
        a1_val = self.params["a10"]  # baseline value
        for e in self.protocol.events:
            # Check if t is within exercise session
            if e.start_time <= t <= e.end_time and e.exercise:
                a1_val = e.beta * e.I * e.F_max
        return a1_val

    def a2(
        self,
        t: float,
    ):
        """Intrinsic growth rate of x2 / AKT.

        Args:
            t (float): Time

        Returns:
            float: a2(t)
        """
        a20 = self.params["a20"]
        t1 = self.params["t1"]
        tau_h = self.params["tau_h"]

        t_ex = self.protocol.get_previous_exercise_start_time(t)

        if t_ex is None:  # no previous exercise, revert to baseline
            a2_val = a20
        else:
            # Integrate from start of last exercise session to current time
            a2_int, _ = scipy.integrate.quad(
                lambda t: 0.5
                * (-1 / tau_h - (t - t_ex - t1) / tau_h**2)
                * np.exp((t - t_ex - t1) / tau_h),
                t_ex,
                t,
            )
            a2_val = a20 + max(a2_int, 0)
        return a2_val

    def f(
        self,
        x3: float,
        x4: float,
        z: float,
    ):
        """Rate of change of myofibril population z.

        Args:
            x3 (float): Current value of x3 / FOXO
            x4 (float): Current value of x4 / mTOR
            z (float): Current value of z / Myofibril population

        Returns:
            float: dz_dt
        """
        z_min = self.params["z_min"]
        z_max = self.params["z_max"]
        k1 = self.params["k1"]
        k2 = self.params["k2"]
        x3_th = self.params["x3_th"]
        x4_th = self.params["x4_th"]

        # Checks if z is outside the range [z_min, z_max], in which case
        # no myofibril growth is possible. Otherwise, add contributions from
        # FOXO and mTOR if they are above their respective thresholds.
        if z < z_min or z > z_max:
            roc = 0
        elif x3 < x3_th and x4 < x4_th:
            roc = 0
        elif x3 > x3_th and x4 > x4_th:
            roc = k1 * (x4 - x4_th) - k2 * (x3 - x3_th)
        elif x3 > x3_th and x4 < x4_th:
            roc = -k2 * (x3 - x3_th)
        elif x3 < x3_th and x4 > x4_th:
            roc = k1 * (x4 - x4_th)
        else:
            raise (ValueError)
        return roc

    def rhs(
        self,
        t: float,
        y: tuple,
    ):
        """Right-hand side of the ODE system, as required by SciPy's solve_ivp.

        Args:
            t (float): Time
            y (tuple): State vector

        Returns:
            tuple: Derivatives of the state variables
        """
        # Unpack state vector
        x1, x2, x3, x4, z = y

        # Load parameters
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        c21 = self.params["c21"]
        a3 = self.params["a3"]
        b3 = self.params["b3"]
        c32 = self.params["c32"]
        a4 = self.params["a4"]
        b4 = self.params["b4"]
        c42 = self.params["c42"]
        c43 = self.params["c43"]

        # Lotka-Volterra system for x1-4
        dx1_dt = x1 * (self.a1(t - 12) - b1 * x1)  # NB: 12 hours delay
        dx2_dt = x2 * (self.a2(t) - b2 * x2 + c21 * x1)
        dx3_dt = x3 * (a3 - b3 * x3 - c32 * x2)
        dx4_dt = x4 * (a4 - b4 * x4 + c42 * x2 - c43 * x3)

        # Rate of change of myofibril population z given by f
        dz_dt = self.f(x3, x4, z)

        return (dx1_dt, dx2_dt, dx3_dt, dx4_dt, dz_dt)

    def solve(
        self,
        event: Event,
        y0: tuple,
    ):
        """Solve the ODE system for a single event.

        Args:
            event (Event): Growth or exercise period
            y0 (tuple): Initial state vector
        """
        event.solution = scipy.integrate.solve_ivp(
            self.rhs,
            (event.start_time, event.end_time),
            y0,
            max_step=self.dt,
        )

    def assemble_solution(self):
        """Assemble the solution from the individual event solutions to
        single t and y arrays."""
        t = np.array([])
        y = []
        for event in self.protocol.events:
            t = np.append(t, event.solution.t)
            if len(y) == 0:  # first event
                y = event.solution.y
            else:
                y = np.hstack((y, event.solution.y))
            pass
        self.t = t
        self.y = y

    def solution_dataframe(self):
        """Create a pandas DataFrame from the solution."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "t": self.t,
                "igf1": self.y[0, :],
                "akt": self.y[1, :],
                "foxo": self.y[2, :],
                "mtor": self.y[3, :],
                "z": self.y[4, :],
            }
        )
        return df

    def presolve_event(self, event):
        """
        Solves the ODE system for a single event and returns the raw time history
        and corresponding myofibril growth rate `f`.
        """
        # Solve the ODE system for the entire event duration
        solution = scipy.integrate.solve_ivp(
            self.rhs,
            (event.start_time, event.end_time),
            self.y_prev,
            max_step=self.dt,
        )
        self.y_prev = tuple(solution.y[:, -1])  # Update state for next event

        # Directly use the solver's output times
        event_times = solution.t

        # Calculate f_rate for each state returned by the solver
        f_rate_history = np.array(
            [self.f(state[2], state[3], state[4]) for state in solution.y.T]
        ).reshape(
            -1, 1
        )  # Reshape to (n_steps, 1)

        solution_dict = {"t": event_times, "f_rate": f_rate_history}
        return solution_dict

    def set_feedback(self, k1_function: dolfinx.fem.Function):
        """
        Sets the protein synthesis rate from a dolfinx Function.
        For the non-spatial model, it extracts a single scalar value.
        """
        # Assume the feedback is uniform and take the first value from the array
        k1_scalar = k1_function.x.array[0]
        self.params["k1"] = k1_scalar

    def simulate(self):
        """Simulate the model for the given exercise protocol."""

        # Solve for each event (exercise+growth period) separately
        for event in self.protocol.events:

            self.solve(
                event,
                self.y_prev,
            )
            self.y_prev = tuple(event.solution.y[:, -1])  # initial state for next event

        # Assemble the full solution from individual solutions
        self.assemble_solution()
