import dolfinx
import adios4dolfinx
import musclex
import ufl
import numpy as np
import pandas as pd
import time
from pathlib import Path
import yaml
from musclex.utils import get_interpolation_points, mpiprint


class MuscleGrowthModel:

    def __init__(
        self,
        exercise_model,
        material_model,
        output_dir,
        csa_function,
        feedback="hill",
        output_freq: int = 10,
    ):
        """Initialize the coupled exercise-mechanics model.
        Args:
            exercise_model: Exercise model incl protocol
            material_model: Material model incl geometry
            output_dir: Results output directory
            csa_function: Function to compute cross-sectional area from displacement field u
            feedback: Feedback function type for protein synthesis rate.
                      Options are 'hill', 'linear', or False.
            output_freq: Frequency for writing visualization files during simulation
        """
        # --- Store inputs ---
        self.exercise_model: musclex.exercise_model.ExerciseModel = exercise_model
        self.material_model: musclex.material.MuscleRohrle = material_model
        self.output_dir: Path = output_dir
        self.csa_function: callable = csa_function

        self.csa0: float = self.csa_function(u=None)  # initial cross-sectional area
        self.csa: dolfinx.fem.Constant = dolfinx.fem.Constant(
            self.material_model.domain, self.csa0
        )
        self.kappa: float = 1.0 / self.csa0
        self.dt_growth: float = 1.0  # growth time step in hours
        self.output_freq: int = output_freq

        # --- Create growth_factor object ---
        # Type (Constant or Function) depends on model type.
        if isinstance(
            self.exercise_model, musclex.exercise_model_spatial.SpatialExerciseModel
        ):
            self.is_spatial = True
            V_dg0 = self.exercise_model.V_ode
            self.growth_factor = dolfinx.fem.Function(V_dg0, name="growth_factor")
        else:
            self.is_spatial = False
            self.growth_factor = dolfinx.fem.Constant(self.material_model.domain, 1.0)

        # --- Set feedback function ---
        feedback_functions = {
            "hill": (hill_feedback, "Using Hill feedback function"),
            "linear": (linear_feedback, "Using linear feedback function"),
            False: (lambda x: 1.0, "No feedback used"),
        }
        try:
            self.feedback_function, msg = feedback_functions[feedback]
            mpiprint(msg)
        except KeyError:
            raise ValueError(
                f"Unknown feedback function: {feedback}. Use 'hill', 'linear', or False."
            )

    def prepare_output(self):
        """Prepare output files for visualization and post-processing."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create displacement solution function
        self.u_sol = dolfinx.fem.Function(self.material_model.V, name="u_growth")
        self.u_sol.x.array[:] = self.material_model.state.x.array[
            self.material_model.V_to_W
        ]  # initial guess for displacement

        # File for visualization in ParaView
        self.outfile_vis = dolfinx.io.VTXWriter(
            self.material_model.domain.comm,
            self.output_dir / "u_growth.bp",
            self.u_sol,
            engine="BP4",
        )

        # File for post-processing with Pyvista
        self.outfile_pp = self.output_dir / "u_growth_pp.bp"
        adios4dolfinx.write_mesh(self.outfile_pp, self.material_model.domain)
        adios4dolfinx.write_meshtags(
            self.outfile_pp,
            self.material_model.domain,
            self.material_model.ft,
            meshtag_name="facet_tags",
        )

        # Initialize list to track output times
        self.output_times = []

    def _update_growth_factor(self, growth_rate_array):
        """Helper to update the growth_factor object, whether it's a Constant or Function."""
        value = 1.0 + growth_rate_array * self.dt_growth / (self.kappa * self.csa.value)
        if self.is_spatial:
            self.growth_factor.x.array[:] = value
        else:
            self.growth_factor.value = value[0]

    def write_output(self, t):
        """Write output to files at time t."""

        # Write displacement field to u_growth.bp
        self.outfile_vis.write(t)
        mpiprint(f"Output written for time {t:.2f} s to {self.outfile_vis}")

        # Write displacement field to u_growth_pp.bp
        adios4dolfinx.write_function(self.outfile_pp, self.u_sol, time=t)
        mpiprint(f"Output written for time {t:.2f} s to {self.outfile_pp}")

        # Track output time
        self.output_times.append(t)

    def simulate(self):
        """Simulate the coupled exercise-mechanics model."""

        # --- Prepare output files ---
        self.prepare_output()

        # Track state before each growth step
        state_prev = self.material_model.state.x.array.copy()

        # --- Initialize tensors for cumulative growth ---
        #   G     : Instantaneous growth tensor
        #   A_tot : Cumulative elastic deformation gradient
        #   G_tot : Cumulative growth tensor
        #   F_tot : Total deformation gradient
        def identity_tensor(x):
            """Helper to create identity tensor for interpolation."""
            values = np.eye(3).reshape(9, 1)
            return np.tile(values, (1, x.shape[1]))

        Q = dolfinx.fem.functionspace(self.material_model.domain, ("DG", 1, (3, 3)))

        G = dolfinx.fem.Function(Q)
        G.interpolate(identity_tensor)
        A_tot = dolfinx.fem.Function(Q)
        A_tot.interpolate(identity_tensor)
        G_tot = dolfinx.fem.Function(Q)
        G_tot.interpolate(identity_tensor)
        F_tot = A_tot * G_tot

        # --- Define expression for transversely isotropic growth tensor ---
        #   growth_factor : cross-sectional area growth factor (theta^2)
        #                   (defined in __init__ as Constant or Function)
        #   eta : along-fiber growth factor (=1 for purely transverse growth)
        eta = dolfinx.fem.Constant(
            self.material_model.domain, dolfinx.default_scalar_type(1.0)
        )
        growth_tensor = musclex.growth.GrowthTensor(
            theta=self.growth_factor ** (1 / 2), eta=eta, a0=self.material_model.a0
        )
        G_update_expr = dolfinx.fem.Expression(
            growth_tensor.tensor(), get_interpolation_points(Q.element)
        )

        # --- Define expressions for updating cumulative tensors ---
        # (See Goriely and Ben Amar (2007))
        F = ufl.Identity(3) + ufl.grad(self.u_sol)  # deformation gradient
        # Update rules. These expressions will be re-evaluated at each growth step.
        G_tot_update_rule = dolfinx.fem.Expression(
            ufl.inv(A_tot) * G * A_tot * G_tot, get_interpolation_points(Q.element)
        )
        A_tot_update_rule = dolfinx.fem.Expression(
            F * ufl.inv(G_tot), get_interpolation_points(Q.element)
        )

        # --- Get baseline protein synthesis rate k1 ---
        # (will be a Function for spatial, scalar for non-spatial)
        if self.is_spatial:
            k10 = self.exercise_model.param_funcs["k1"]
        else:
            k10 = self.exercise_model.params["k1"]

        # --- Initialize result trackers ---
        csas_history = []
        times_history = []
        k1_history = []
        volumes_history = []
        tic = time.perf_counter()  # start timer

        # --- Event-based outer loop ---
        for event in self.exercise_model.protocol.events:
            mpiprint(
                f"\n{'='*70}\nProcessing event from t={event.start_time} to {event.end_time}"
            )

            # --- ODE solve ---
            # Pre-solve the signaling model for the current event
            # and store the solution in the event object
            event.solution = self.exercise_model.presolve_event(event)

            if event.exercise:
                mpiprint("Stimulus event processed. No growth applied.")
                continue  # move to next event

            # Extract signaling solution
            f_rate_history = event.solution["f_rate"]
            signal_times = event.solution["t"]

            # --- Growth mechanics solve ---
            # Define growth time steps within the event duration
            growth_times = np.arange(event.start_time, event.end_time, self.dt_growth)

            # Iterate over growth time steps
            for i, t_growth in enumerate(growth_times):

                # --- Find growth rate at this time step ---
                # Find the index in the signaling history closest to the current growth time
                idx = np.argmin(np.abs(signal_times - t_growth))
                # Get the corresponding myofibril rate of change (f)
                growth_rate_at_step = f_rate_history[idx, :]

                self._update_growth_factor(growth_rate_at_step)
                print(
                    f"At growth time {t_growth:.2f}, growth rate is {self.growth_factor.x.array[0] if self.is_spatial else self.growth_factor.value}"
                )

                # --- Update the growth tensor with the new growth factor ---
                G.interpolate(G_update_expr)
                G_tot.interpolate(G_tot_update_rule)

                # --- Solve for mechanical equilibrium with updated growth ---
                # set initial guess
                self.material_model.state.x.array[:] = state_prev[:]
                # solve
                converged, _ = self.material_model.solve(Fg=G_tot)
                # stop simulation if mechanics solver fails
                if not converged:
                    mpiprint("Mechanics solver failed. Stopping growth simulation.")
                    return

                # --- Update cumulative elastic deformation ---
                state_prev[:] = self.material_model.state.x.array[:]
                self.u_sol.x.array[:] = self.material_model.state.x.array[
                    self.material_model.V_to_W
                ]
                A_tot.interpolate(A_tot_update_rule)

                # --- Track results at each growth step ---
                csa_current_step = self.csa_function(self.u_sol)
                mpiprint(
                    "CSA at time", t_growth + self.dt_growth, "is", csa_current_step
                )
                csas_history.append(csa_current_step)
                times_history.append(t_growth + self.dt_growth)

                if self.is_spatial:
                    k1_mean = np.mean(self.exercise_model.param_funcs["k1"].x.array)
                    k1_history.append(k1_mean)
                else:
                    k1_history.append(self.exercise_model.params["k1"])

                volume = dolfinx.fem.assemble_scalar(
                    dolfinx.fem.form(
                        ufl.det(F_tot) * ufl.dx
                    )  # F_tot implicitly updated
                )
                volumes_history.append(volume)

                # --- Write output ---
                # Write every output_freq steps and at the last step
                if i % self.output_freq == 0 or i == len(growth_times) - 1:
                    self.write_output(t_growth + self.dt_growth)

                # --- Feedback ---
                # After the event, calculate and set feedback for the next event
                csa_new = csas_history[-1]  # Use the most recent CSA
                self.csa.value = csa_new
                feedback_scalar = self.feedback_function(csa_new / self.csa0)

                V_dg0 = dolfinx.fem.functionspace(self.material_model.domain, ("DG", 0))
                k1_feedback_field = dolfinx.fem.Function(V_dg0)

                if self.is_spatial:
                    k1_feedback_field.x.array[:] = feedback_scalar * k10.x.array
                else:
                    k1_feedback_field.x.array[:] = feedback_scalar * k10

                self.exercise_model.set_feedback(k1_feedback_field)
            mpiprint(
                f"Event finished. New CSA: {csa_new:.4e}. Feedback scalar: {feedback_scalar:.3f}"
            )

        time_elapsed = time.perf_counter() - tic  # stop timer
        mpiprint(f"\n--- Growth simulation finished in {time_elapsed:.2f} seconds ---")

        # --- Save results to file ---
        # Collect results into a DataFrame and save
        growth_results_df = pd.DataFrame(
            {
                "t": times_history,
                "csa": csas_history,
                "k1": k1_history,
                "volume": volumes_history,
            }
        )
        growth_results_df.to_csv(self.output_dir / "growth_results.csv")
        # Save output times
        np.save(self.output_dir / "output_times.npy", np.array(self.output_times))

        # Collect simulation statistics and save to YAML file
        summary = {
            "simulation_time_seconds": round(time_elapsed, 2),
            "degrees_of_freedom": int(self.material_model.ndofs),
            "number_of_cells": int(self.material_model.ncells),
            "mpi_processes": self.material_model.domain.comm.size,
        }
        with open(self.output_dir / "simulation_summary.yml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        # FIXME: Assemble ODE solutions and save to file
        # self.exercise_model.assemble_solution()
        # ode_results_df = self.exercise_model.solution_dataframe()
        # ode_results_df.to_csv(self.output_dir / "ode_results.csv")

        mpiprint(f"Simulation results saved to {self.output_dir}")

        return


def hill_function(x, K_A=1.0, n=4, max_response=2):
    """Hill function"""
    L_pow_n = x**n
    K_A_pow_n = K_A**n
    return max_response * L_pow_n / (K_A_pow_n + L_pow_n)


def hill_feedback(x, K_A=1.0, n=4, max_response=2):
    """Hill-type feedback function for protein synthesis rate"""
    return max_response - hill_function(x, K_A, n, max_response)


def linear_feedback(x, a=-2, b=3):
    """Linear feedback function for protein synthesis rate"""
    return a * x + b
