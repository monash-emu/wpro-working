import numpy as np

from .outputs import Outputs
from .process import get_piecewise_cosine


def renew_basic(
    gen_time_densities: np.array,
    process_vals: np.array,
    pop: float,
    seed: float,
    n_times: int,
) -> Outputs:
    """Implementation of the renewal process.

    Args:
        gen_time_densities: Incremental CDF values for the generation time
        process_vals: The non-mechanistic adjustment process to the reproduction number
        pop: The population size
        seed: Starting infectious seed
        n_times: Number of time points for simulation

    Returns:
        Standard output arrays
    """
    incidence = np.zeros(n_times)
    suscept = np.zeros(n_times)
    r_t = np.zeros(n_times)

    incidence[0] = seed
    suscept[0] = pop - seed
    r_t[0] = process_vals[0] * suscept[0] / pop

    for t in range(1, n_times):
        contribution_by_day = incidence[:t] * gen_time_densities[:t][::-1]  # Product of incidence values and reversed generation times
        r_t[t] = process_vals[t] * suscept[t - 1] / pop  # Pre-specified process by the proportion susceptible
        incidence[t] = contribution_by_day.sum() * r_t[t]  # Incidence
        suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)  # Zero out any small negative susceptible values
        
    return Outputs(incidence, suscept, r_t)


def renew_trunc_gen(
    gen_time_densities: np.array,
    process_vals: np.array,
    pop: float,
    seed: float,
    n_times: int,
    gen_times_end: int,
) -> Outputs:
    """Implementation of the renewal process,
    with option to truncate the generation interval distribution,
    because density values have fallen low enough to be ignored.

    Args:
        gen_time_densities: Incremental CDF values for the generation time
        process_vals: The non-mechanistic adjustment process to the reproduction number
        pop: The population size
        seed: Starting infectious seed
        n_times: Number of time points for simulation
        gen_times_end: Index for the last generation time of interest

    Returns:
        Standard output arrays
    """
    incidence = np.zeros(n_times)
    suscept = np.zeros(n_times)
    r_t = np.zeros(n_times)

    incidence[0] = seed
    suscept[0] = pop - seed
    r_t[0] = process_vals[0] * suscept[0] / pop

    for t in range(1, n_times):
        gen_times_interest = min(t, gen_times_end)  # Number of generation times relevant to current loop
        inc_vals = incidence[t - gen_times_interest :t]  # Incidence series
        gen_vals = gen_time_densities[:gen_times_interest]  # Generation series
        contribution_by_day = inc_vals * gen_vals[::-1]  # Product of incidence values and reversed generation times
        r_t[t] = process_vals[t] * suscept[t - 1] / pop  # Pre-specified process by the proportion susceptible
        incidence[t] = contribution_by_day.sum() * r_t[t]  # Incidence for this time point
        suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)  # Zero out any small negative susceptible values

    return Outputs(incidence, suscept, r_t)


def renew_taper_seed(
    gen_time_densities: np.array,
    process_vals: np.array,
    pop: float,
    seed_peak: float,
    n_times: int,
    run_in: int,
) -> Outputs:
    incidence = np.zeros(n_times)
    suscept = np.zeros(n_times)
    r_t = np.zeros(n_times)

    incidence[0] = seed_peak
    suscept[0] = pop - seed_peak
    r_t[0] = process_vals[0] * suscept[0] / pop

    seed_func = get_piecewise_cosine([0.0, run_in], [seed_peak, 0.0])

    for t in range(1, n_times):
        r_t[t] = process_vals[t] * suscept[t - 1] / pop
        contribution_by_day = incidence[:t] * gen_time_densities[:t][::-1]
        seeding_component = seed_func(t)
        renewal_component = contribution_by_day.sum() * r_t[t]
        incidence[t] = seeding_component + renewal_component
        suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)

    return Outputs(incidence, suscept, r_t)
