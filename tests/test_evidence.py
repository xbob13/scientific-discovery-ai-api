from research_lab.evidence import EngineeringEvidence


def test_measured_replicated_evidence_scores_above_theory_only():
    theory, _ = EngineeringEvidence("theoretical", False, None, 0, False, False, False, False).score()
    measured, features = EngineeringEvidence("laboratory", True, 20, 2, True, True, True, True).score()
    assert measured > theory
    assert features["result_basis"] == "laboratory"
    assert measured <= 1
