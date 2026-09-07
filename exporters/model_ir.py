"""Canonical ModelIR construction and validation. Python 3.10+, standard library."""

from __future__ import annotations
import gzip
import json
import math
from pathlib import Path


class Builder:
    def __init__(self, name, source):
        self.ir = dict(
            version="1.0",
            name=name,
            rootId="model",
            source=source,
            entities=[],
            tensors=[],
            edges=[],
        )
        self.entities = {}
        self.node("model", None, "Model", name)

    def node(self, id, parent, kind, label, attrs=None, evidence="derived"):
        if id in self.entities:
            raise ValueError(f"Duplicate entity: {id}")
        entity = dict(id=id, parentId=parent, kind=kind, label=label, evidence=evidence)
        if attrs:
            entity["attrs"] = attrs
        self.entities[id] = entity
        self.ir["entities"].append(entity)
        return entity

    def tensor(
        self,
        name,
        parent,
        shape,
        dtype="unspecified",
        role="parameter",
        evidence="derived",
        values=None,
        storage=None,
        label=None,
    ):
        tensor = dict(
            id=name, shape=list(shape), dtype=dtype, role=role, evidence=evidence
        )
        if values is not None:
            tensor["values"] = values
        if storage:
            tensor["storage"] = storage
        self.ir["tensors"].append(tensor)
        kind = (
            "ScalarCell"
            if not shape
            else "VectorColumn"
            if len(shape) == 1
            else "MatrixPlane"
            if len(shape) == 2
            else "TensorVolume"
        )
        entity = self.node(
            "tensor:" + name,
            parent,
            kind,
            label or name.rsplit(".", 1)[-1],
            evidence=evidence,
        )
        entity["tensorId"] = name
        return tensor

    def edge(self, source, target, label, kind="dataflow", evidence="derived"):
        self.ir["edges"].append(
            dict(
                id=f"edge:{len(self.ir['edges'])}",
                from_=source,
                to=target,
                kind=kind,
                label=label,
                evidence=evidence,
            )
        )
        self.ir["edges"][-1]["from"] = self.ir["edges"][-1].pop("from_")

    def norm(self, id, parent, width, label, rms=False):
        self.node(
            id,
            parent,
            "NormalizationPlane",
            label,
            {
                "formula": "x / sqrt(mean(x²) + ε) ⊙ γ"
                if rms
                else "(x − mean(x)) / sqrt(var(x) + ε) ⊙ γ + β"
            },
        )
        self.tensor(id + ".weight", id, [width])
        if not rms:
            self.tensor(id + ".bias", id, [width])

    def projection(self, id, parent, rows, cols, label, bias=False, transposed=False):
        self.node(
            id,
            parent,
            "ProjectionBlock",
            label,
            {
                "formula": "y = Wx + b" if bias else "y = Wx",
                "inputWidth": cols,
                "outputWidth": rows,
                "storageOrientation": "[input, output]"
                if transposed
                else "[output, input]",
            },
        )
        self.tensor(id + ".weight", id, [cols, rows] if transposed else [rows, cols])
        if bias:
            self.tensor(id + ".bias", id, [rows])


def parameter_count(ir):
    if any(t["role"] == "unknown" for t in ir["tensors"]):
        return None
    return sum(math.prod(t["shape"]) for t in ir["tensors"] if t["role"] == "parameter")


def validate(ir):
    if ir.get("version") != "1.0":
        raise ValueError("Unsupported ModelIR version")
    entities = {e["id"]: e for e in ir["entities"]}
    tensors = {t["id"]: t for t in ir["tensors"]}
    if len(entities) != len(ir["entities"]) or len(tensors) != len(ir["tensors"]):
        raise ValueError("Duplicate identity")
    if ir["rootId"] not in entities or entities[ir["rootId"]]["parentId"] is not None:
        raise ValueError("Invalid root")
    done = {ir["rootId"]}
    for entity in entities.values():
        cursor, seen = entity, set()
        while cursor["id"] not in done:
            if cursor["id"] in seen:
                raise ValueError("Containment cycle")
            seen.add(cursor["id"])
            cursor = entities[cursor["parentId"]]
        done.update(seen)
        for tid in entity.get("tensorRefs", []) + (
            [entity["tensorId"]] if "tensorId" in entity else []
        ):
            if tid not in tensors:
                raise ValueError(f"Unknown tensor {tid}")
    for tensor in tensors.values():
        if any(
            not isinstance(d, (int, str))
            or isinstance(d, bool)
            or (isinstance(d, int) and d < 0)
            for d in tensor["shape"]
        ):
            raise ValueError("Invalid tensor shape")
    for edge in ir["edges"]:
        if edge["from"] not in entities or edge["to"] not in entities:
            raise ValueError("Dangling dependency")
    return ir


def write_package(ir, path):
    validate(ir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        ir, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    # mtime=0 makes byte-for-byte packages reproducible.
    path.write_bytes(
        gzip.compress(raw, compresslevel=9, mtime=0) if path.suffix == ".gz" else raw
    )
    count = parameter_count(ir)
    return {
        "path": str(path),
        "entities": len(ir["entities"]),
        "tensors": len(ir["tensors"]),
        "parameters": str(count) if count is not None else "unknown",
        "package_bytes": path.stat().st_size,
        "uncompressed_bytes": len(raw),
    }
