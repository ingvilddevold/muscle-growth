import dolfinx
import numpy as np
import adios4dolfinx
import yaml
import scipy.integrate
from musclex.protocol import ExerciseProtocol, Event
from musclex.utils import mpiprint
from pathlib import Path


DEFAULT_VARIATION_PARAMETERS = [
    "a10",
    "a20",
    "a3",
    "a4",
    "b1",
    "b2",
    "b3",
    "b4",
    "c21",
    "c32",
    "c42",
    "c43",
]


class SpatialExerciseModel:
    """
    Spatially varying model of the IGF1-AKT-FOXO-mTOR signaling network,
    driven by an ExerciseProtocol.

    This model represents all state variables and key parameters as FEniCSx
    functions on a DG-0 space, allowing for spatial variation. It is advanced
    in time with a manual fourth-order Runge-Kutta (RK4) scheme. It includes
    an explicit 12-hour delay for the a1 stimulus.
    """

    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        protocol: ExerciseProtocol,
        conf_file: str,
        output_dir: Path,
        variation_magnitude: float = 0.1,
        spatially_varied_parameters: list[str] = DEFAULT_VARIATION_PARAMETERS,
        seed: int=42,
        write_freq: int = 20,
    ):
        """
        Args:
            domain (dolfinx.mesh.Mesh): The computational domain.
            protocol (ExerciseProtocol): The protocol defining exercise and rest periods.
            conf_file (str): Path to the parameter configuration file.
            output_dir (Path): Directory to save output files.
            variation_magnitude (float): Fractional magnitude of random variation.
                                         e.g., 0.1 means +/- 10%.
            spatially_varied_parameters (list of str): List of parameter names to vary spatially.
            seed (int): Random seed for reproducibility.
        """
        # Store inputs
        self.domain = domain
        self.protocol = protocol
        self.t = 0.0
        self.variation_magnitude = variation_magnitude
        self.write_freq = write_freq

        # Read BASELINE parameters from the config file
        with open(conf_file) as config:
            conf = yaml.load(config, Loader=yaml.FullLoader)
            self.base_params = conf["exercise_parameters"]

        self.dt = self.base_params["dt"]  # Time step for RK4 integration

        # Set random seed for reproducibility
        np.random.seed(seed)

        # --- Setup FEM for the ODE system ---
        # Use DG0 space for piecewise constant functions
        self.V_ode = dolfinx.fem.functionspace(self.domain, ("DG", 0))

        # --- Create a dictionary to hold all parameter functions ---
        self.param_funcs = {}
        self.spatially_varied_parameters = spatially_varied_parameters
        self._initialize_spatial_parameters()
        self._save_spatial_parameters(output_dir)
        self._save_spatial_parameters_bp(output_dir)

        # --- Initialize state variables as FEniCSx Functions ---
        self.x1 = dolfinx.fem.Function(self.V_ode, name="igf1")
        self.x2 = dolfinx.fem.Function(self.V_ode, name="akt")
        self.x3 = dolfinx.fem.Function(self.V_ode, name="foxo")
        self.x4 = dolfinx.fem.Function(self.V_ode, name="mtor")
        self.z = dolfinx.fem.Function(self.V_ode, name="z")
        # Function to hold the growth rate (dz/dt)
        self.f_rate = dolfinx.fem.Function(self.V_ode, name="f_rate")

        # --- Setup ADIOS Writer ---
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bp_file_path = self.output_dir / "ode_spatial_history.bp"

        # Write mesh ONCE (important for readers)
        adios4dolfinx.write_mesh(self.bp_file_path, self.domain)

        # Store functions to save
        self.state_functions = {
            "igf1": self.x1, "akt": self.x2, "foxo": self.x3,
            "mtor": self.x4, "z": self.z
        }

        # Set initial conditions
        self._set_initial_conditions()

    def _initialize_spatial_parameters(self):
        """
        Create a spatially varying FEniCSx Function for each
        parameter defined in the config file.
        """
        num_cells = self.V_ode.dofmap.index_map.size_local

        for name, baseline_value in self.base_params.items():
            param_func = dolfinx.fem.Function(self.V_ode, name=name)

            if name in self.spatially_varied_parameters:

                # Define the spatial variation (random noise around the baseline)
                # Draw num_cells samples from uniform(-1, 1)
                random_noise = (np.random.rand(num_cells) - 0.5) * 2
                # Scale by variation magnitude and baseline value
                variation = self.variation_magnitude * baseline_value * random_noise
                final_values = baseline_value + variation

                # Ensure non-negativity for inherently non-negative parameters
                if baseline_value >= 0:
                    np.clip(final_values, 0, None, out=final_values)

                param_func.x.array[:] = final_values
            else:
                # No spatial variation, just set to baseline
                param_func.x.array[:] = baseline_value

            # Store in dictionary
            self.param_funcs[name] = param_func

    def _save_spatial_parameters(self, output_dir: Path):
        """Saves the spatially varied parameters to a CSV file."""
        import pandas as pd
        data = {}
        for name in self.spatially_varied_parameters:
            if name in self.param_funcs:
                data[name] = self.param_funcs[name].x.array
            
        df = pd.DataFrame(data)
        output_file = output_dir / "spatial_parameters.csv"
        df.to_csv(output_file, index=False)
        mpiprint(f"Saved spatial parameter distributions to {output_file}")

    def _save_spatial_parameters_bp(self, output_dir: Path):
        """Saves the spatially varied parameters to an ADIOS2 BP file."""
        filepath = output_dir / "spatial_parameters.bp"
        adios4dolfinx.write_mesh(filepath, self.domain)
        for name, func in self.param_funcs.items():
            adios4dolfinx.write_function(
                filepath, func, time=0.0
            )

    def _set_initial_conditions(self):
        """
        Set initial conditions for state variables using the now-spatial
        initial condition parameters.
        """
        self.x1.x.array[:] = self.param_funcs["x1_0"].x.array
        self.x2.x.array[:] = self.param_funcs["x2_0"].x.array
        self.x3.x.array[:] = self.param_funcs["x3_0"].x.array
        self.x4.x.array[:] = self.param_funcs["x4_0"].x.array
        self.z.x.array[:] = self.param_funcs["z_0"].x.array
        self.t = 0.0

    def prepare_for_event(self, event: Event, y0=None):
        self.t = event.start_time

    def get_f_rate(self, t_next: float) -> dolfinx.fem.Function:
        dt = t_next - self.t
        if dt <= 0.0:
            self._update_f_rate()
            return self.f_rate
        self._step()
        self.t = t_next
        self._update_f_rate()
        return self.f_rate

    def set_feedback(self, k1_function: dolfinx.fem.Function):
        self.param_funcs["k1"] = k1_function

    def _a1(self, t: float) -> float:
        """
        Intrinsic growth rate of x1 / IGF1 with explicit 12h time delay.
        """
        #a1_val = self.base_params["a10"]  # baseline value
        a1_vals = self.param_funcs["a10"].x.array # baseline values (non-exercise)
        for e in self.protocol.events:
            # Check if t is within an exercise session
            if e.start_time <= t <= e.end_time and e.exercise:
                a1_val = e.beta * e.I * e.F_max
                # convert to array
                a1_vals = a1_val * np.ones_like(self.param_funcs["a10"].x.array)
        return a1_vals

    def _a2(self, t: float) -> float:
        """
        Intrinsic growth rate of x2 / AKT.
        """
        #a20 = self.base_params["a20"]
        a20 = self.param_funcs["a20"].x.array
        t1 = self.base_params["t1"]
        tau_h = self.base_params["tau_h"]

        t_ex = self.protocol.get_previous_exercise_start_time(t)

        if t_ex is None:  # no previous exercise, revert to baseline
            return a20
        else:
            # Integrate from start of latest exercise to current time
            a2_int, _ = scipy.integrate.quad(
                lambda t_prime: 0.5
                * (-1 / tau_h - (t_prime - t_ex - t1) / tau_h**2)
                * np.exp((t_prime - t_ex - t1) / tau_h),
                t_ex,
                t,
            )
            return a20 + max(a2_int, 0) * np.ones_like(a20)

    def _get_state_arrays(self):
        """Helper to get a tuple of the current state arrays."""
        return (
            self.x1.x.array,
            self.x2.x.array,
            self.x3.x.array,
            self.x4.x.array,
            self.z.x.array,
        )

    def _rhs(self, t: float, y: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        """
        Computes the right-hand side of the ODE system for a given time and state.
        Takes the state `y` as an argument to support RK4 intermediate steps.
        """
        x1, x2, x3, x4, z = y

        b1, b2, c21 = (
            self.param_funcs["b1"].x.array,
            self.param_funcs["b2"].x.array,
            self.param_funcs["c21"].x.array,
        )
        a3, b3, c32 = (
            self.param_funcs["a3"].x.array,
            self.param_funcs["b3"].x.array,
            self.param_funcs["c32"].x.array,
        )
        a4, b4, c42, c43 = (
            self.param_funcs["a4"].x.array,
            self.param_funcs["b4"].x.array,
            self.param_funcs["c42"].x.array,
            self.param_funcs["c43"].x.array,
        )

        a1_stimulus = self._a1(t - 12.0)  # NOTE the 12h delay
        a2_t = self._a2(t)

        dx1_dt = x1 * (a1_stimulus - b1 * x1)
        dx2_dt = x2 * (a2_t - b2 * x2 + c21 * x1)
        dx3_dt = x3 * (a3 - b3 * x3 - c32 * x2)
        dx4_dt = x4 * (a4 - b4 * x4 + c42 * x2 - c43 * x3)
        dz_dt = self._calculate_f_rate(x3, x4, z)

        return (dx1_dt, dx2_dt, dx3_dt, dx4_dt, dz_dt)

    def _calculate_f_rate(self, x3_vals, x4_vals, z_vals):
        """
        Calculates the rate of change of z (dz/dt) for a given state.
        """
        # Load parameters needed for f_rate calculation
        k1_vals = self.param_funcs["k1"].x.array
        k2_vals = self.param_funcs["k2"].x.array
        z_min_vals = self.param_funcs["z_min"].x.array
        z_max_vals = self.param_funcs["z_max"].x.array
        x3_th_vals = self.param_funcs["x3_th"].x.array
        x4_th_vals = self.param_funcs["x4_th"].x.array

        # Initialize roc (rate of change) array
        roc = np.zeros_like(z_vals)

        # Update roc based on conditions

        # Update roc where mTOR is active (above threshold)
        mTOR_active = x4_vals > x4_th_vals  # indices
        roc[mTOR_active] = k1_vals[mTOR_active] * (
            x4_vals[mTOR_active] - x4_th_vals[mTOR_active]
        )
        # Update roc where FOXO is active (above threshold)
        FOXO_active = x3_vals > x3_th_vals  # indices
        roc[FOXO_active] -= k2_vals[FOXO_active] * (
            x3_vals[FOXO_active] - x3_th_vals[FOXO_active]
        )
        # Set roc to zero where z is out of bounds
        out_of_bounds = (z_vals < z_min_vals) | (z_vals > z_max_vals)
        roc[out_of_bounds] = 0.0

        return roc

    def _update_f_rate(self):
        """
        Updates the self.f_rate function to reflect the current state.
        """
        current_state = (self.x3.x.array, self.x4.x.array, self.z.x.array)
        self.f_rate.x.array[:] = self._calculate_f_rate(
            current_state[0], current_state[1], current_state[2]
        )

    def _step(self):
        """Perform a single fourth-order Runge-Kutta (RK4) step."""
        y_n = self._get_state_arrays()
        t_n = self.t
        dt = self.dt

        # Calculate k1
        k1 = self._rhs(t_n, y_n)

        # Calculate k2
        y_k2 = tuple(y + (dt / 2.0) * k for y, k in zip(y_n, k1))
        k2 = self._rhs(t_n + dt / 2.0, y_k2)

        # Calculate k3
        y_k3 = tuple(y + (dt / 2.0) * k for y, k in zip(y_n, k2))
        k3 = self._rhs(t_n + dt / 2.0, y_k3)

        # Calculate k4
        y_k4 = tuple(y + dt * k for y, k in zip(y_n, k3))
        k4 = self._rhs(t_n + dt, y_k4)

        # Update state variables
        self.x1.x.array[:] += (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.x2.x.array[:] += (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.x3.x.array[:] += (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.x4.x.array[:] += (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self.z.x.array[:] += (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])

    def presolve_event(self, event: Event, dt: float = 0.1):
        """
        Steps through a single event and returns the history.
        Uses the current state of the model as the initial condition.
        """
        self.t = event.start_time
        # We only need to return the history of f_rate for the growth model
        history = {
            "t": [],
            "f_rate": [],
            "igf1": [],
            "akt": [],
            "foxo": [],
            "mtor": [],
            "z": [],
        }
        # Write initial states at t=event.start_time
        for name, func in self.state_functions.items():
            adios4dolfinx.write_function(self.bp_file_path, func, time=self.t)
            mpiprint(f"Wrote {name} at time {self.t:.2f} to {self.bp_file_path}")
        ode_output_times = [self.t]

        t_steps = np.arange(event.start_time, event.end_time, dt)  # ODE time steps
        for i, t in enumerate(t_steps):
            self.get_f_rate(t + dt)
            history["t"].append(t + dt)
            history["f_rate"].append(self.f_rate.x.array.copy())
            history["igf1"].append(self.x1.x.array.copy())
            history["akt"].append(self.x2.x.array.copy())
            history["foxo"].append(self.x3.x.array.copy())
            history["mtor"].append(self.x4.x.array.copy())
            history["z"].append(self.z.x.array.copy())
            if ((i + 1) % self.write_freq == 0) or (self.t == t_steps[-1]):
                mpiprint("Writing ODE state at time {:.2f}...".format(self.t))
                for name, func in self.state_functions.items():
                    adios4dolfinx.write_function(self.bp_file_path, func, time=self.t)
                ode_output_times.append(self.t)
        
        # Convert lists to arrays
        for key in history:
            if key == "t":
                history[key] = np.array(history[key])
            else:
                history[key] = np.vstack(history[key])

        # Append times to ODE times file
        existing_times = []
        ode_times_file = self.output_dir / "ode_times.npy"
        if ode_times_file.exists():
            existing_times = np.load(ode_times_file).tolist()
        all_times = existing_times + ode_output_times
        np.save(ode_times_file, np.array(all_times))

        return history

    def simulate(self):
        """
        Steps through the protocol and simulates the spatial model over time.
        """
        times = []

        mpiprint(f"\n--- Running protocol: {type(self.protocol).__name__} ---")
        mpiprint(f"Saving ODE history to: {self.bp_file_path}")

        # --- Save Initial State (t=0) ---
        current_time = 0.0
        for name, func in self.state_functions.items():
             adios4dolfinx.write_function(self.bp_file_path, func, time=current_time)
        times.append(current_time)

        for event in self.protocol.events:
            mpiprint(
                f"Simulating event from t={event.start_time:.1f} to t={event.end_time:.1f}..."
            )

            t_steps = np.arange(event.start_time, event.end_time, self.dt)
            for t in t_steps:
                self.get_f_rate(t + self.dt)
                current_time = self.t
                times.append(current_time)
                
                # --- Save current state ---
                for name, func in self.state_functions.items():
                    # Append data for the current time step
                    adios4dolfinx.write_function(self.bp_file_path, func, time=current_time)

        # write ode times to npy
        print(f"Saving ODE time steps to: {self.output_dir / 'ode_times.npy'}")
        print(f"Number of time steps: {len(times)}")
        np.save(self.output_dir / "ode_times.npy", np.array(times))
        
        return np.array(times)


    def assemble_and_save_results(self, events: list, output_dir: Path):
        """
        Assembles the full solution from all event solutions and saves the
        results to Parquet files in the specified directory.
        
        Saves data in a 'long' format (time, cell, value) for fast plotting.
        """
        import pandas as pd

        # Check if the output directory exists, create if not
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lists to hold the concatenated data from all events
        full_history = {
            "t": [], "igf1": [], "akt": [], "foxo": [], "mtor": [], "z": []
        }
        
        # Loop through events and concatenate their solutions
        for event in events:
            if event.solution and event.solution['t'].size > 0:
                full_history["t"].append(event.solution["t"])
                full_history["igf1"].append(event.solution["igf1"])
                full_history["akt"].append(event.solution["akt"])
                full_history["foxo"].append(event.solution["foxo"])
                full_history["mtor"].append(event.solution["mtor"])
                full_history["z"].append(event.solution["z"])

        # Concatenate lists of arrays into single arrays
        time_vector = np.concatenate(full_history["t"])
        
        # Get the number of cells from the first data matrix
        if "z" in full_history and len(full_history["z"]) > 0:
            num_cells = full_history["z"][0].shape[1]
            # Create column names as strings: '0', '1', '2', ...
            cell_column_names = [str(i) for i in range(num_cells)]
        else:
            mpiprint("No data to save. assemble_and_save_results is exiting.")
            return

        for key in ["igf1", "akt", "foxo", "mtor", "z"]:
            
            if not full_history[key]:
                mpiprint(f"No data for '{key}', skipping.")
                continue

            # Concatenate along the time axis (axis 0)
            data_matrix = np.concatenate(full_history[key], axis=0)
            
            # 1. Create a DataFrame in WIDE format (time, '0', '1', '2', ...)
            df_wide = pd.DataFrame(data_matrix, columns=cell_column_names)
            df_wide.insert(0, "time", time_vector)
            
            # 2. Melt the DataFrame to LONG format
            mpiprint(f"Melting data for '{key}'...")
            df_long = df_wide.melt(
                id_vars=['time'],
                var_name='cell',
                value_name=key  # The value column will be named 'z', 'igf1', etc.
            )
            
            # Convert cell column to integer for better storage/lookup
            df_long['cell'] = df_long['cell'].astype(int)
            
            # 3. Define the Parquet output path
            output_path = output_dir / f"ode_spatial_{key}.parquet"
            
            # 4. Save the new LONG DataFrame to Parquet
            df_long.to_parquet(output_path, index=False, engine='pyarrow')
            
            mpiprint(f"Saved LONG format history for {key} to {output_path}")
