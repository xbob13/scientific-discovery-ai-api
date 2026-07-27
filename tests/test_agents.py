from research_lab.agents import LiteratureCartographer, ResearchDocument


def test_cartographer_proposes_explainable_deterministic_connection():
    documents = [
        ResearchDocument("a", "Graphene sensor", "A graphene surface detects protein binding.", True),
        ResearchDocument("b", "Protein adsorption", "Protein binding changes graphene conductivity.", True),
        ResearchDocument("c", "Unrelated astronomy", "A distant galaxy contains old stars.", True),
    ]
    first = LiteratureCartographer().propose(documents)
    second = LiteratureCartographer().propose(documents)
    assert first == second
    assert len(first) == 1
    assert {first[0].left_id, first[0].right_id} == {"a", "b"}
    assert "graphene" in first[0].bridge_terms


def test_cartographer_does_not_invent_connection_without_shared_signal():
    documents = [
        ResearchDocument("a", "Graphene sensor", "Conductivity measurement.", True),
        ResearchDocument("b", "Protein assay", "Fluorescent antibody.", True),
    ]
    assert LiteratureCartographer().propose(documents) == []
