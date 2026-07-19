from urcm.core.ingest import KnowledgeIngestion


def discover_chain(relations, start, goal, max_hops=5):
    graph = {}
    for r in relations:
        if len(r) == 3 and r[0] in ("all","some","most","coref","implies"):
            graph.setdefault(r[1], []).append(r[2])
    visited = {start}
    frontier = [(start, [start])]
    while frontier:
        node, path = frontier.pop(0)
        for nbr in graph.get(node, []):
            if nbr.lower() == goal.lower():
                return path + [nbr]
            if nbr not in visited and len(path) < max_hops:
                visited.add(nbr)
                frontier.append((nbr, path + [nbr]))
    return []

def detect_contradictions(relations, start, goal):
    tags = set()
    for r in relations:
        if len(r) == 3 and r[1].lower() == start.lower() and r[2].lower() == goal.lower():
            if r[0] in ("all","no"):
                tags.add(r[0])
    return tags

def test_explanation_path_and_contradiction():
    ing = KnowledgeIngestion(l2_dim=512)
    ing.ingest_text("All glorps are zinks. All zinks are quibbles.")
    ing.ingest_text("No glorps are quibbles.")
    rels = ing.relations
    chain = discover_chain(rels, "glorps", "quibbles")
    assert chain == ["glorps", "zinks", "quibbles"]
    tags = detect_contradictions(rels, "glorps", "quibbles")
    assert "no" in tags
    assert chain == ["glorps", "zinks", "quibbles"]
