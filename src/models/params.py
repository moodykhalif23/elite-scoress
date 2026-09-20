from __future__ import annotations

import numpy as np

NAMES = ("att", "def", "intercept", "home_adv", "rho")


def posterior_arrays(result) -> dict[str, np.ndarray]:
    if hasattr(result, "posterior"):
        return {n: result.posterior[n].stack(s=("chain", "draw")).to_numpy() for n in NAMES}
    return {n: np.atleast_1d(np.asarray(result[n]))[..., None] for n in NAMES}
