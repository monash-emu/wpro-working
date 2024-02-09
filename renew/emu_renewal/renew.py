from typing import List
import numpy as np
from typing import NamedTuple
from jax import lax, vmap
from jax import numpy as jnp

from emu_renewal.process import cosine_multicurve, sinterp
from .process import LinearInterpFunc, CosInterpFunc
from .distributions import GammaDens


class RenewalState(NamedTuple):
    incidence: jnp.array
    suscept: float


class ModelResult(NamedTuple):
    incidence: jnp.array
    suscept: jnp.array
    r_t: jnp.array
    process: jnp.array


class RenewalModel:
    def __init__(self, pop, n_times, run_in, n_process_periods):
        self.pop = pop
        self.n_times = n_times
        self.run_in = run_in
        self.n_process_periods = n_process_periods
        self.req_x_vals = np.linspace(0.0, n_times, n_process_periods)
        self.interp = CosInterpFunc(self.req_x_vals)
        self.dens_obj = GammaDens()
        self.model_times = np.array([float(t) for t in range(self.n_times)])
        self.seeder = CosInterpFunc([0.0, round(self.run_in / 2.0), self.run_in])

    def seeding_func(self, seed_param):
        return self.seeder.get_interp_func([0.0, np.exp(seed_param), 0.0])
       
    def func(self, gen_time_mean: float, gen_time_sd: float, process_req: List[float], seed: int) -> tuple:
        densities = self.dens_obj.get_densities(self.n_times, gen_time_mean, gen_time_sd)
        process_func = self.interp.get_interp_func(process_req)
        process_vals_exp = np.exp(process_func(self.model_times))

        incidence = np.zeros(self.n_times)
        suscept = np.zeros(self.n_times)
        r_t = np.zeros(self.n_times)

        seed_func = self.seeding_func(seed)
        seed_0 = seed_func(0.0)
        incidence[0] = seed_0
        suscept[0] = self.pop - seed_0
        r_t[0] = process_vals_exp[0] * suscept[0] / self.pop

        for t in range(1, self.n_times):
            r_t[t] = process_vals_exp[t] * suscept[t - 1] / self.pop
            contribution_by_day = incidence[:t] * densities[:t][::-1]
            seeding_component = seed_func(float(t))
            renewal_component = contribution_by_day.sum() * r_t[t]
            incidence[t] = seeding_component + renewal_component
            suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)

        return ModelResult(incidence, suscept, r_t, process_vals_exp)

    def get_model_desc(self):
        renew_desc = (
            '\n\n### Renewal process\n'
            'Calculation of the renewal process '
            'consists of multiplying the incidence values for the preceding days '
            'by the reversed generation time distribution values. '
            'This follows a standard formula, '
            'described elsewhere by several groups,[@cori2013; @faria2021] i.e. '
            '$$i_t = R_t\sum_{\\tau<t} i_\\tau g_{t-\\tau}$$\n'
            '$R_t$ is calculated as the product of the proportion '
            'of the population remaining susceptible '
            'and the non-mechanistic random process '
            'generated external to the renewal model. '
            'The susceptible population is calculated by '
            'subtracting the number of new incident cases from the '
            'running total of susceptibles at each iteration.\n'
        )

        non_mech_desc = (
            '\n\n### Non-mechanistic process\n'
            'The time values corresponding to the submitted process values '
            'are set to be evenly spaced throughout the simulation period. '
            'Next, a continuous function of time was constructed from '
            'the non-mechanistic process series values submitted to the model. '
            'After curve fitting, the sequence of parameter values pertaining to '
            'the non-mechanistic process are exponentiated, '
            'such that parameter exploration for these quantities is '
            'undertaken in the log-transformed space. '
        )

        return renew_desc + non_mech_desc

    def get_full_desc(self):

        return (
            self.dens_obj.get_description() +
            self.get_model_desc() +
            self.interp.get_description()
        )


class JaxModel(RenewalModel):
    def __init__(self, population, start, end, seed_duration, n_process_periods, dens_obj, window_len):
        self.pop = population
        self.seed_duration = seed_duration
        self.n_process_periods = n_process_periods
        self.window_len = window_len
        self.x_proc_vals = sinterp.get_scale_data(jnp.linspace(start, end, self.n_process_periods))
        self.dens_obj = dens_obj
        self.start = start
        self.model_times = jnp.arange(start, end)
        self.seed_x_vals = [start, start + self.seed_duration * 0.5, start + self.seed_duration]
        self.start_seed = 0.0
        self.end_seed = 0.0

    def seed_func(self, t, seed):
        x_vals = sinterp.get_scale_data(jnp.array(self.seed_x_vals))
        y_vals = sinterp.get_scale_data(jnp.array([self.start_seed, jnp.exp(seed), self.end_seed]))
        return cosine_multicurve(t, x_vals, y_vals)
    
    def fit_process_curve(self, y_proc_vals):
        return jnp.exp(vmap(cosine_multicurve, in_axes=(0, None, None))(self.model_times, self.x_proc_vals, y_proc_vals))

    def func(self, gen_time_mean, gen_time_sd, process_req, seed):
        densities = self.dens_obj.get_densities(self.window_len, gen_time_mean, gen_time_sd)

        y_proc_vals = sinterp.get_scale_data(process_req)
        process_vals = self.fit_process_curve(y_proc_vals)

        init_state = RenewalState(jnp.zeros(self.window_len), self.pop)
        
        def state_update(state: RenewalState, t) -> tuple[RenewalState, jnp.array]:
            r_t = process_vals[t - self.start] * state.suscept / self.pop  # Index is the order in the process_vals sequence, rather than the model time value
            renewal = (densities * state.incidence).sum() * r_t
            seed_component = self.seed_func(t, seed)
            total_new_incidence = renewal + seed_component
            total_new_incidence = jnp.where(total_new_incidence > state.suscept, state.suscept, total_new_incidence)
            suscept = state.suscept - total_new_incidence
            incidence = jnp.zeros_like(state.incidence)
            incidence = incidence.at[1:].set(state.incidence[:-1])
            incidence = incidence.at[0].set(total_new_incidence)
            return RenewalState(incidence, suscept), jnp.array([total_new_incidence, suscept, r_t])

        end_state, outputs = lax.scan(state_update, init_state, self.model_times)
        return ModelResult(outputs[:, 0], outputs[:, 1], outputs[:, 2], process_vals)
    
    def get_full_desc(self):
       
        seed_desc = (
            '\n\n### Seeding\n'
            'Seeding was achieved by interpolating using a cosine function. '
            f'The number of seeded cases scaled from {self.start_seed} at time {self.seed_x_vals[0]} '
            'to the peak value at half way through the burn-in period '
            f'back to {self.end_seed} at the end of the burn-in period ({self.seed_x_vals[-1]}). '
        )
        
        return (
            self.dens_obj.get_description() +
            self.get_model_desc() +
            seed_desc
        )
    