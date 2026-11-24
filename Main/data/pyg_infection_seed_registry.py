"""Names for the prespecified Infection graph-seed replication cohort."""

ADDITIONAL_GRAPH_SEEDS = (1, 2, 3, 4)


def seeded_infection_names():
    for family in ("ER", "BA"):
        for graph_seed in ADDITIONAL_GRAPH_SEEDS:
            canonical = f"PyG-Infection-{family}-L3-GraphSeed{graph_seed}-Split43"
            path = f"pyg_infection_{family.lower()}_l3_graphseed{graph_seed}_split43"
            yield canonical, path, family.lower(), graph_seed
