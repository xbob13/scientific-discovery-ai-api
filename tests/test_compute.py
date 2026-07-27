from research_lab.compute import run_sheet_resistance
from research_lab.schemas import ComputeRequest


def test_compute_is_reproducible():
    request = ComputeRequest(conductivity_s_m=5.8e7, thickness_m=1e-4, width_m=0.01, length_m=1)
    first = run_sheet_resistance(request)
    second = run_sheet_resistance(request)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert first["result"]["p05"] < first["result"]["p95"]


def test_seed_changes_artifact():
    values = dict(conductivity_s_m=1e6, thickness_m=1e-5, width_m=0.02, length_m=0.5)
    assert (
        run_sheet_resistance(ComputeRequest(**values, seed=1))["artifact_sha256"]
        != run_sheet_resistance(ComputeRequest(**values, seed=2))["artifact_sha256"]
    )
