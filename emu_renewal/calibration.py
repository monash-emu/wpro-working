from typing import Dict
from jax import numpy as jnp
from jax import jit
import numpy as np
import pandas as pd
import numpyro
from numpyro import distributions as dist
pd.options.plotting.backend = "plotly"

from emu_renewal.renew import RenewalModel


class Calibration:
    def __init__(
        self, 
        epi_model: RenewalModel, 
        data: pd.Series,
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
        """
        self.epi_model = epi_model
        self.n_process_periods = len(self.epi_model.x_proc_data.points)

        analysis_dates_idx = self.epi_model.epoch.index_to_dti(self.epi_model.analysis_times)
        common_dates_idx = data.index.intersection(analysis_dates_idx)
        self.data = jnp.array(data.loc[common_dates_idx])
        self.common_model_idx = np.array(self.epi_model.epoch.dti_to_index(common_dates_idx).astype(int)) - self.epi_model.model_times[0]

    def calibration(self):
        pass
    def get_description(self):
        pass


class StandardCalib(Calibration):
    def __init__(
        self, 
        epi_model: RenewalModel, 
        data: pd.Series,
    ):
        """Set up calibration object with epi model and data.

        Args:
            epi_model: The renewal model
            data: The data targets
        """
        super().__init__(epi_model, data)
        self.data_disp_range = [jnp.log(1.0), jnp.log(1.5)]
        self.proc_disp_dist_val = 1.0

    def get_model_notifications(self, gen_mean, gen_sd, proc, seed, cdr):
        """Get the modelled notifications from a set of epi parameters.

        Args:
            gen_mean: Generation time mean
            gen_sd: Generation time standard deviation
            proc: Values of the variable process
            seed: Log-transformed peak seeding value
            cdr: Case detection rate/proportion

        Returns:
            Case notification rate
        """
        return self.epi_model.renewal_func(gen_mean, gen_sd, proc, seed).incidence[self.common_model_idx] * cdr

    def calibration(
        self, 
        params: Dict[str, float],
    ):
        """See get_description below.

        Args:
            params: Parameters with single value
        """
        param_updates = {k: numpyro.sample(k, dist.Uniform(v["lower"], v["upper"])) for k, v in params.items()}
        proc_dispersion = numpyro.sample("proc_dispersion", dist.HalfNormal(self.proc_disp_dist_val))
        n_process_periods = self.n_process_periods
        proc_dist = dist.Normal(jnp.repeat(0.0, n_process_periods), proc_dispersion)
        param_updates["proc"] = numpyro.sample("proc", proc_dist)
        log_model_res = jnp.log(jit(self.get_model_notifications)(**param_updates))
        log_target = jnp.log(self.data)
        dispersion = numpyro.sample("dispersion", dist.Uniform(*self.data_disp_range))
        like = dist.Normal(log_model_res, dispersion).log_prob(log_target).sum()
        numpyro.factor("notifications_ll", like)

    def get_description(self) -> str:
        return (
            f"The calibration process calibrates parameters for {self.n_process_periods} "
            "values for periods of the variable process to the data. "
            "The relative values pertaining to each period of the variable process "
            "are estimated from normal prior distributions centred at no "
            "change from the value of the previous stage of the process. "
            "The dispersion of the variable process is calibrated, "
            "using a half-normal distribution. "
            "The log of the modelled notification rate for each parameter set "
            "is compared against the data from the end of the run-in phase "
            "through to the end of the analysis. "
            "Modelled notifications are calculated as the product of modelled incidence and the "
            "(constant through time) case detection proportion. "
            "The dispersion parameter for this comparison of log values is "
            "also calibrated using a uniform distribution, "
            f"which is calibrated in the range {round(float(self.data_disp_range[0]), 3)} "
            f"to {round(float(self.data_disp_range[1]), 3)}. "
        )
