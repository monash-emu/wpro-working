import numpy as np

from outputs import Outputs


def renew_basic(gen_time_densities, process_vals, pop, seed, n_times) -> Outputs:
    """The renewal process.
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


def renew_trunc_gen(gen_time_densities, process_vals, pop, seed, n_times, gen_times_end) -> Outputs:
    """The renewal process.
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