from typing import List
import numpy as np

def get_interp_vals_over_model_time(req: List[float], n_times: int) -> np.array:
    """Linear interpolation at requested values at regular intervals over simulation period.
    """
    return np.interp(range(n_times), np.linspace(0.0, n_times, len(req)), req)