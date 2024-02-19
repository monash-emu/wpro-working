from typing import List
from typing import NamedTuple
from jax import lax, vmap
from jax import numpy as jnp
from datetime import datetime

from emu_renewal.process import cosine_multicurve, sinterp


class RenewalState(NamedTuple):
    incidence: jnp.array
    suscept: float


class ModelResult(NamedTuple):
    incidence: jnp.array
    suscept: jnp.array
    r_t: jnp.array
    process: jnp.array


class RenewalModel:
    def __init__(self, population, start, end, run_in_req, proc_update_freq, dens_obj, window_len, epoch=None):
        self.epoch = epoch
        self.start = self.process_time_req(start)
        self.end = self.process_time_req(end)
        self.pop = population
        self.seed_duration = run_in_req
        self.window_len = window_len
        self.simulation_start = self.start - run_in_req
        self.model_times = jnp.arange(self.simulation_start, self.end + 1)
        self.x_proc_vals = jnp.arange(self.end, self.start, -proc_update_freq)[::-1]
        self.x_proc_data = sinterp.get_scale_data(self.x_proc_vals)
        self.run_in = int(self.x_proc_vals[0] - self.simulation_start)
        self.dens_obj = dens_obj
        self.seed_x_vals = jnp.linspace(self.simulation_start, self.start, 3)
        self.start_seed = 0.0
        self.end_seed = 0.0

    def process_time_req(self, req):
        msg = "Time data type not supported"
        if isinstance(req, int):
            return req
        elif isinstance(req, datetime):
            return int(self.epoch.dti_to_index(req))
        else:
            raise ValueError(msg)

    def seed_func(self, t, seed):
        x_vals = sinterp.get_scale_data(jnp.array(self.seed_x_vals))
        y_vals = sinterp.get_scale_data(jnp.array([self.start_seed, jnp.exp(seed), self.end_seed]))
        return cosine_multicurve(t, x_vals, y_vals)

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
            "\n\n### Renewal process\n"
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
            "\n\n### Non-mechanistic process\n"
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

        seed_desc = (
            "\n\n### Seeding\n"
            "Seeding was achieved by interpolating using a cosine function. "
            f"The number of seeded cases scaled from {self.start_seed} at time {self.seed_x_vals[0]} "
            "to the peak value at half way through the burn-in period "
            f"back to {self.end_seed} at the end of the burn-in period ({self.seed_x_vals[-1]}). "
        )

        return self.dens_obj.get_desc() + self.get_model_desc() + seed_desc
