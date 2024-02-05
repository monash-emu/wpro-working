from typing import List
import numpy as np
from collections import namedtuple

from .process import LinearInterpFunc, CosInterpFunc
from .distributions import GammaDens


ModelResult = namedtuple("ModelResult", ["incidence", "suscept", "r_t", "process"])


class RenewalModel:
    def __init__(self, pop, n_times, run_in, n_process_periods):
        self.pop = pop
        self.n_times = n_times
        self.run_in = run_in
        self.n_process_periods = n_process_periods
        req_x_vals = np.linspace(0.0, n_times, n_process_periods)
        self.interp = CosInterpFunc(req_x_vals)
        self.dens_obj = GammaDens()
        self.model_times = np.array([float(t) for t in range(self.n_times)])
        self.seeder = CosInterpFunc([0.0, self.run_in * 0.5, self.run_in])

    def func(
        self, gen_time_mean: float, gen_time_sd: float, process_req: List[float], seed: int
    ) -> tuple:
        densities = self.dens_obj.get_densities(self.n_times, gen_time_mean, gen_time_sd)
        process_func = self.interp.get_interp_func(process_req)
        process_vals_exp = np.exp(process_func(self.model_times))

        incidence = np.zeros(self.n_times)
        suscept = np.zeros(self.n_times)
        r_t = np.zeros(self.n_times)

        seed_func = self.seeder.get_interp_func([0.0, seed, 0.0])

        # seed_peak = np.exp(seed)
        incidence[0] = seed_func(0.0)
        suscept[0] = self.pop - incidence[0]
        r_t[0] = process_vals_exp[0] * suscept[0] / self.pop

        for t in range(1, self.n_times):
            r_t[t] = process_vals_exp[t] * suscept[t - 1] / self.pop
            contribution_by_day = incidence[:t] * densities[:t][::-1]
            seeding_component = seed_func(float(t))
            renewal_component = contribution_by_day.sum() * r_t[t]
            incidence[t] = seeding_component + renewal_component
            suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)

        return ModelResult(incidence, suscept, r_t, process_vals_exp)

    def get_description(self):
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

        return (
            self.dens_obj.get_description()
            + renew_desc
            + non_mech_desc
            + self.interp.get_description()
        )


class TruncRenewalModel(RenewalModel):
    def __init__(self, pop, n_times, run_in, n_process_periods, gen_times_end=1000):
        super().__init__(pop, n_times, run_in, n_process_periods)
        self.gen_times_end = gen_times_end

    def func(
        self, gen_time_mean: float, gen_time_sd: float, process_req: List[float], seed: int
    ) -> tuple:
        densities = self.dens_obj.get_densities(self.n_times, gen_time_mean, gen_time_sd)
        process_func = self.interp.get_interp_func(process_req)
        process_vals_exp = np.exp(process_func(self.model_times))

        incidence = np.zeros(self.n_times)
        suscept = np.zeros(self.n_times)
        r_t = np.zeros(self.n_times)

        seed_peak = np.exp(seed)
        incidence[0] = seed_peak
        suscept[0] = self.pop - seed_peak
        r_t[0] = process_vals_exp[0] * suscept[0] / self.pop

        seed_func = self.seeder.get_interp_func([seed_peak, 0.0])
        for t in range(1, self.n_times):
            gen_times_interest = min(
                t, self.gen_times_end
            )  # Truncate generation times if requested
            inc_vals = incidence[t - gen_times_interest : t]  # Incidence series
            gen_vals = densities[:gen_times_interest]  # Generation series

            r_t[t] = process_vals_exp[t] * suscept[t - 1] / self.pop
            contribution_by_day = inc_vals * gen_vals[::-1]
            seeding_component = seed_func(t)
            renewal_component = contribution_by_day.sum() * r_t[t]
            incidence[t] = seeding_component + renewal_component
            suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)

        return ModelResult(incidence, suscept, r_t, process_vals_exp)
