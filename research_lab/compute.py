import hashlib
import json
from datetime import UTC, datetime

import numpy as np

from .schemas import ComputeRequest

MODEL_SOURCE = b"sheet_resistance=length/(conductivity*thickness*width); monte-carlo lognormal conductivity"


def run_sheet_resistance(request: ComputeRequest) -> dict:
    """Propagate conductivity uncertainty through a transparent analytical model."""
    rng = np.random.default_rng(request.seed)
    sigma = np.sqrt(np.log(1 + request.conductivity_relative_uncertainty**2))
    conductivity = rng.lognormal(
        mean=np.log(request.conductivity_s_m) - sigma**2 / 2,
        sigma=sigma,
        size=request.samples,
    )
    resistance = request.length_m / (conductivity * request.thickness_m * request.width_m)
    result = {
        "model": "uniform rectangular conductor; analytical, not high-fidelity simulation",
        "units": "ohm",
        "mean": float(np.mean(resistance)),
        "median": float(np.median(resistance)),
        "p05": float(np.quantile(resistance, 0.05)),
        "p95": float(np.quantile(resistance, 0.95)),
        "samples": request.samples,
        "seed": request.seed,
        "assumptions": [
            "uniform isotropic conductivity",
            "constant temperature",
            "ideal contacts",
            "independent lognormal conductivity uncertainty",
        ],
    }
    artifact = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return {
        "result": result,
        "code_sha256": hashlib.sha256(MODEL_SOURCE).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "finished_at": datetime.now(UTC),
    }
