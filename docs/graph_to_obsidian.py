"""Export graphify-out/graph.json as Obsidian notes (one note per node).

Obsidian has no importer for graph.json, so this writes plain markdown with
[[wikilinks]]: Obsidian's own graph view then draws the graph. The "Research"
folder on the Desktop is already an Obsidian vault, so the output directory
sits inside it and needs no vault setup.

    python docs/graph_to_obsidian.py [--graph graphify-out/graph.json]
                                     [--out graphify-out/obsidian]

Re-run after `graphify update .`; the output directory is rewritten in place.
"""

import argparse
import json
import pathlib
import re
import shutil
from collections import defaultdict

SAFE = re.compile(r'[\\/:*?"<>|#^\[\]]+')


def note_names(nodes):
    """Map node id -> unique note filename (stem), preferring the label."""
    by_stem = defaultdict(list)
    for node in nodes:
        stem = SAFE.sub("-", node.get("label") or node["id"]).strip()[:160] or node["id"]
        # macOS/APFS is case-insensitive, so "Foo" and "foo" are one file: key on casefold
        by_stem[stem.casefold()].append((stem, node["id"]))

    names = {}
    for group in by_stem.values():
        for stem, node_id in group:
            # ponytail: disambiguate collisions with the id, which is already unique
            names[node_id] = stem if len(group) == 1 else f"{stem} ({node_id})"
    return names


def export(graph_path, out_dir):
    graph = json.loads(pathlib.Path(graph_path).read_text())
    nodes = graph["nodes"]
    names = note_names(nodes)

    outgoing, incoming = defaultdict(list), defaultdict(list)
    for link in graph["links"]:
        source, target = link["source"], link["target"]
        if source in names and target in names:
            outgoing[source].append((link.get("relation", "related"), target))
            incoming[target].append((link.get("relation", "related"), source))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for node in nodes:
        node_id = node["id"]
        lines = [
            "---",
            f'label: "{node.get("label", "")}"',
            f'source_file: "{node.get("source_file", "")}"',
            f'source_location: "{node.get("source_location", "")}"',
            f'community: {node.get("community", "")}',
            f'tags: [graphify, {node.get("file_type", "unknown")}]',
            "---",
            "",
            f"# {node.get('label', node_id)}",
            "",
        ]
        if node.get("source_file"):
            location = node.get("source_location", "")
            lines += [f"`{node['source_file']}{':' + location if location else ''}`", ""]
        if node.get("description"):
            lines += [node["description"], ""]

        for heading, edges in (("Outgoing", outgoing[node_id]), ("Incoming", incoming[node_id])):
            if not edges:
                continue
            lines.append(f"## {heading}")
            by_relation = defaultdict(list)
            for relation, other in edges:
                by_relation[relation].append(other)
            for relation in sorted(by_relation):
                targets = sorted({names[o] for o in by_relation[relation]})
                lines.append(f"- **{relation}**: " + ", ".join(f"[[{t}]]" for t in targets))
            lines.append("")

        (out_dir / f"{names[node_id]}.md").write_text("\n".join(lines))

    return len(nodes), sum(len(v) for v in outgoing.values())


def demo():
    """Self-check: colliding labels stay distinct and links resolve to note names."""
    graph = {
        "nodes": [
            {"id": "a_init", "label": "__init__", "source_file": "a.py"},
            {"id": "b_init", "label": "__init__", "source_file": "b.py"},
            {"id": "solver", "label": "solve/fixed:yc", "source_file": "s.py"},
        ],
        "links": [{"source": "solver", "target": "a_init", "relation": "calls"}],
    }
    tmp = pathlib.Path("/tmp/graphify_obsidian_demo")
    graph_file = tmp.with_suffix(".json")
    graph_file.write_text(json.dumps(graph))
    n_nodes, n_edges = export(graph_file, tmp)
    assert (n_nodes, n_edges) == (3, 1)
    written = sorted(p.name for p in tmp.iterdir())
    assert written == [
        "__init__ (a_init).md",
        "__init__ (b_init).md",
        "solve-fixed-yc.md",
    ], written
    assert "[[__init__ (a_init)]]" in (tmp / "solve-fixed-yc.md").read_text()
    shutil.rmtree(tmp)
    graph_file.unlink()
    print("ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="graphify-out/graph.json")
    parser.add_argument("--out", default="graphify-out/obsidian")
    parser.add_argument("--demo", action="store_true", help="run the self-check and exit")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        n_nodes, n_edges = export(args.graph, pathlib.Path(args.out))
        print(f"wrote {n_nodes} notes ({n_edges} links) to {args.out}")
