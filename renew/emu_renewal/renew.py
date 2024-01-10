import numpy as np
from collections import namedtuple

from .process import get_piecewise_cosine


Outputs = namedtuple('outputs', ['incidence', 'suscept', 'r_t', 'description'])


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
        
    return Outputs(incidence, suscept, r_t, '')


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

    return Outputs(incidence, suscept, r_t, '')


def renew_taper_seed(
    gen_time_densities: np.array,
    process_vals: np.array,
    pop: float,
    seed_peak: float,
    n_times: int,
    seed_duration: int,
) -> Outputs:
    """Renewal process as described below.

    Args:
        gen_time_densities: Generation time densities by day
        process_vals: Non-mechanistic variation in reproduction number
        pop: Starting population
        seed_peak: Peak starting seed value
        n_times: Number of time points for analysis
        seed_duration: Time for seeding function to decay to zero

    Returns:
        Collection of renewal process calculations and description
    """
    incidence = np.zeros(n_times)
    suscept = np.zeros(n_times)
    r_t = np.zeros(n_times)

    incidence[0] = seed_peak
    suscept[0] = pop - seed_peak
    r_t[0] = process_vals[0] * suscept[0] / pop

    seed_desc = 'The model was seeded using a translated, ' \
        'scaled cosine function that declines from ' \
        f'a starting value of {round(seed_peak)} to zero ' \
        f'over the first {seed_duration} days of the simuilation. '
    seed_func = get_piecewise_cosine([seed_peak, 0.0], [0.0, seed_duration])

    renew_desc = 'Calculation of the renewal process ' \
        'consists of multiplying the incidence values for the preceding days ' \
        'by the reversed generation time distribution values. ' \
        'This follows a standard formula, ' \
        'described elsewhere by several groups,[@cori2013; @faria2021] i.e. ' \
        '$$i_t = R_t\sum_{\\tau<t} i_\\tau g_{t-\\tau}$$\n' \
        '$R_t$ is calculated as the product of the proportion ' \
        'of the population remaining susceptible ' \
        'and the non-mechanistic random process ' \
        'generated external to the renewal model. ' \
        'The susceptible population is calculated by ' \
        'subtracting the number of new incident cases from the ' \
        'running total of susceptibles at each iteration.\n'
    for t in range(1, n_times):
        r_t[t] = process_vals[t] * suscept[t - 1] / pop
        contribution_by_day = incidence[:t] * gen_time_densities[:t][::-1]
        seeding_component = seed_func(t)
        renewal_component = contribution_by_day.sum() * r_t[t]
        incidence[t] = seeding_component + renewal_component
        suscept[t] = max(suscept[t - 1] - incidence[t], 0.0)

    return Outputs(incidence, suscept, r_t, renew_desc + seed_desc)
