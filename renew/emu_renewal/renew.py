from typing import Union, Optional
from typing import NamedTuple
from jax import lax, vmap
from jax import numpy as jnp
from datetime import datetime

from summer2.utils import Epoch

from emu_renewal.process import cosine_multicurve, sinterp
from emu_renewal.distributions import Dens
from emu_renewal.utils import format_date_for_str, round_sigfig


class RenewalState(NamedTuple):
    incidence: jnp.array
    suscept: float


class ModelResult(NamedTuple):
    incidence: jnp.array
    suscept: jnp.array
    r_t: jnp.array
    process: jnp.array


class RenewalModel:
    def __init__(
        self, 
        population: float, 
        start: Union[datetime, int], 
        end: Union[datetime, int], 
        run_in_req: int, 
        proc_update_freq: int, 
        dens_obj: Dens, 
        window_len: int, 
        epoch: Optional[Epoch],
    ):
        """Standard renewal model object.

        Args:
            population: Starting population value
            start: Start time for the analysis period (excluding run-in)
            end: End time for the analysis period
            run_in_req: Duration of additional period preceding the analysis period
            proc_update_freq: Frequency with with the vairable process is updated
            dens_obj: Generation time distribution
            window_len: How far to look back in calculating the renewal process
            epoch: Reference epoch for calculating dates. Defaults to None.
        """

        # Times
        self.epoch = epoch
        self.start = self.process_time_req(start)
        self.end = self.process_time_req(end)
        self.simulation_start = self.start - run_in_req
        self.model_times = jnp.arange(self.simulation_start, self.end + 1)
        self.description = (
            "\n### Fixed parameters\n"
            f"The main analysis period runs from {format_date_for_str(start)} "
            f"to {format_date_for_str(end)}, "
            f"with a preceding run in period of {run_in_req} days. "
        )

        # Population
        self.pop = population
        self.description += (
            f"The starting model population is {round_sigfig(population / 1e6, 2)} million persons. "
        )

        # Window        
        self.window_len = window_len
        self.description += (
            "The preceding time window for renewal calculations "
            f"is set to {window_len} days. "
        )

        # Process
        self.x_proc_vals = jnp.arange(self.end, self.start, -proc_update_freq)[::-1]
        self.x_proc_data = sinterp.get_scale_data(self.x_proc_vals)
        process_start = int(self.x_proc_vals[0])
        self.run_in = process_start - self.simulation_start
        if run_in_req != self.run_in:
            self.description += (
                "\n### Variable process\n"
                "Because the analysis period is not an exact multiple "
                "of the duration of a process interval, "
                f"the run-in period is extended from {run_in_req} days "
                f"to {self.run_in} days. "
            )

        # Generation
        self.dens_obj = dens_obj
        self.description += f"\n{self.dens_obj.get_desc()}"

        # Seeding
        self.seed_x_vals = jnp.linspace(self.simulation_start, self.start, 3)
        self.start_seed = 0.0
        self.end_seed = 0.0
        self.describe_seed_func()

    def process_time_req(
        self, 
        req: Union[datetime, int],
    ) -> int:
        """Sort out a user requested date.

        Args:
            req: The request

        Raises:
            ValueError: If neither date nor int

        Returns:
            The request converted to int according to the model's epoch
        """
        msg = "Time data type not supported"
        if isinstance(req, int):
            return req
        elif isinstance(req, datetime):
            return int(self.epoch.dti_to_index(req))
        else:
            raise ValueError(msg)

    def seed_func(
        self, 
        t: float, 
        seed_peak: float,
    ) -> float:
        """See describe_seed_func

        Args:
            t: Model time
            seed_peak: Peak seeding rate 

        Returns:
            Seeding rate at the requested time
        """
        x_vals = sinterp.get_scale_data(jnp.array(self.seed_x_vals))
        y_vals = sinterp.get_scale_data(jnp.array([self.start_seed, jnp.exp(seed_peak), self.end_seed]))
        return cosine_multicurve(t, x_vals, y_vals)
    
    def describe_seed_func(self):
        self.description += (
            "\n### Seeding\n   "
            f"The seeding process scales up from a value of {self.start_seed} "
            "at the start of the run-in period to it's peak value "
            "through the first half of the run-in, "
            f"and then decays back to {self.end_seed} at the end of the run-in. "
            "This is implemented using a half-cosine function "
            "that is translated and scaled to reach zero gradient at each of the "
            "three specified points (start, peak, end). "
        )

    def fit_process_curve(self, y_proc_vals):
        cos_func = vmap(cosine_multicurve, in_axes=(0, None, None))
        return jnp.exp(cos_func(self.model_times, self.x_proc_data, y_proc_vals))

    def func(self, gen_mean, gen_sd, y_proc_req, seed):
        densities = self.dens_obj.get_densities(self.window_len, gen_mean, gen_sd)
        y_proc_vals = jnp.cumsum(jnp.concatenate([jnp.array((0,)), y_proc_req]))
        y_proc_data = sinterp.get_scale_data(y_proc_vals)
        process_vals = self.fit_process_curve(y_proc_data)
        init_state = RenewalState(jnp.zeros(self.window_len), self.pop)

        def state_update(state: RenewalState, t) -> tuple[RenewalState, jnp.array]:
            proc_val = jnp.where(t < self.run_in, 1.0, process_vals[t - self.simulation_start])
            r_t = proc_val * state.suscept / self.pop
            renewal = (densities * state.incidence).sum() * r_t
            seed_component = self.seed_func(t, seed)
            total_new_inc = renewal + seed_component
            total_new_inc = jnp.where(total_new_inc > state.suscept, state.suscept, total_new_inc)
            suscept = state.suscept - total_new_inc
            incidence = jnp.zeros_like(state.incidence)
            incidence = incidence.at[1:].set(state.incidence[:-1])
            incidence = incidence.at[0].set(total_new_inc)
            out = {"incidence": total_new_inc, "suscept": suscept, "r_t": r_t, "process": proc_val}
            return RenewalState(incidence, suscept), out

        end_state, outputs = lax.scan(state_update, init_state, self.model_times)
        return ModelResult(**outputs)

    def get_model_desc(self):
        renew_desc = (
            "## Renewal process\n"
            "Calculation of the renewal process "
            "consists of multiplying the incidence values for the preceding days "
            "by the reversed generation time distribution values. "
            "This follows a standard formula, "
            "described elsewhere by several groups,[@cori2013; @faria2021] i.e. "
            "$$i_t = R_t\sum_{\\tau<t} i_\\tau g_{t-\\tau}$$\n"
            "$R_t$ is calculated as the product of the proportion "
            "of the population remaining susceptible "
            "and the non-mechanistic random process "
            "generated external to the renewal model. "
            "The susceptible population is calculated by "
            "subtracting the number of new incident cases from the "
            "running total of susceptibles at each iteration.\n"
        )

        non_mech_desc = (
            "## Non-mechanistic process\n"
            "The time values corresponding to the submitted process values "
            "are set to be evenly spaced throughout the simulation period. "
            "Next, a continuous function of time was constructed from "
            "the non-mechanistic process series values submitted to the model. "
            "After curve fitting, the sequence of parameter values pertaining to "
            "the non-mechanistic process are exponentiated, "
            "such that parameter exploration for these quantities is "
            "undertaken in the log-transformed space. "
        )

        return renew_desc + non_mech_desc

    def get_full_desc(self):

        return self.dens_obj.get_desc() + self.get_model_desc()
