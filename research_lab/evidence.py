from dataclasses import dataclass


@dataclass(frozen=True)
class EngineeringEvidence:
    result_basis: str
    has_uncertainty: bool
    sample_count: int | None
    independent_replications: int
    data_available: bool
    code_available: bool
    conditions_reported: bool
    calibrated: bool

    def score(self) -> tuple[float, dict]:
        basis = {"theoretical": 0.05, "modeled": 0.15, "laboratory": 0.30, "device": 0.38, "commercial": 0.42}
        features = {
            "result_basis": self.result_basis,
            "basis_weight": basis.get(self.result_basis, 0),
            "uncertainty_weight": 0.10 if self.has_uncertainty else 0,
            "sample_weight": min((self.sample_count or 0) / 100, 1) * 0.08,
            "replication_weight": min(self.independent_replications, 3) / 3 * 0.15,
            "availability_weight": (0.05 if self.data_available else 0)
            + (0.05 if self.code_available else 0),
            "quality_weight": (0.04 if self.conditions_reported else 0) + (0.03 if self.calibrated else 0),
        }
        return min(sum(v for k, v in features.items() if k.endswith("_weight")), 1.0), features
