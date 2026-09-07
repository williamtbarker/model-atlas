"""Metadata-only extraction CLI. No inference, CUDA, or model code download.

Examples:
  python3 -m exporters.extract --bundled --output public/models
  python3 -m exporters.extract --config config.json --output model.atlas.json.gz
  python3 -m exporters.extract --safetensors model.safetensors --output catalog.atlas.json.gz
  python3 -m exporters.extract --onnx model.onnx --output graph.atlas.json.gz
"""

from __future__ import annotations
import argparse
import json
import math
import struct
from pathlib import Path
from .model_ir import Builder, validate, write_package
from .bundled import gpt2, deepseek4, tiny_example, ROOT


def tensor_catalog(tensors, name="Tensor catalog", notes=None):
    b = Builder(
        name,
        {
            "kind": "tensor catalog",
            "evidence": "observed",
            "notes": notes
            or "Header tensor metadata. Prefix grouping does not establish computation order; parameter roles are unknown.",
        },
    )
    for tensor in tensors:
        parts = tensor["id"].split(".")
        parent = "model"
        for i in range(len(parts) - 1):
            id = "module:" + ".".join(parts[: i + 1])
            if id not in b.entities:
                b.node(id, parent, "Module", parts[i], evidence="observed")
            parent = id
        b.tensor(
            tensor["id"],
            parent,
            tensor["shape"],
            tensor["dtype"],
            tensor.get("role", "unknown"),
            "observed",
        )
    return validate(b.ir)


def safetensors_header(path):
    path = Path(path)
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError("Truncated safetensors file")
        size = struct.unpack("<Q", first)[0]
        if not 2 <= size <= 16_000_000 or size + 8 > path.stat().st_size:
            raise ValueError("Invalid header size")
        header = json.loads(f.read(size))
    tensors = []
    spans = []
    width = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "U64": 8,
        "I32": 4,
        "U32": 4,
        "I16": 2,
        "U16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape = tensor["shape"]
        span = tensor["data_offsets"]
        if (
            any(type(d) is not int or d < 0 for d in shape)
            or len(span) != 2
            or any(type(x) is not int or x < 0 for x in span)
            or span[0] > span[1]
        ):
            raise ValueError("Invalid tensor bounds")
        if (
            tensor["dtype"] in width
            and math.prod(shape) * width[tensor["dtype"]] != span[1] - span[0]
        ):
            raise ValueError("Tensor byte/shape mismatch")
        spans.append(span)
        tensors.append(
            dict(id=name, shape=shape, dtype=tensor["dtype"], role="unknown")
        )
    end = 0
    for start, stop in sorted(spans):
        if start != end:
            raise ValueError("Overlapping or incomplete data spans")
        end = stop
    if end + size + 8 != path.stat().st_size:
        raise ValueError("Tensor spans do not match file size")
    return tensor_catalog(tensors, path.name)


def from_torch(model, name=None):
    """Inspect an already-created nn.Module. Never call forward or materialize values.

    Can be used with a model constructed on torch.device('meta'). Parameter-object
    identity reconciles tied weights. Distinct Parameter views sharing overlapping
    storage are not coalesced; named_modules() visits each reused module once.
    """
    b = Builder(
        name or type(model).__name__,
        {
            "kind": "PyTorch module traversal",
            "evidence": "observed",
            "notes": "Containment and registered parameter ownership only; functional operations and execution order require an exported graph. Shared Parameter objects are deduplicated; overlapping views are not coalesced.",
        },
    )
    parameter_ids = {}
    modules = list(model.named_modules())
    if len(modules) > 100_000:
        raise ValueError("Module budget exceeded")
    for path, module in modules:
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        node_id = "module:" + path if path else "model"
        if path:
            b.node(
                node_id,
                "module:" + parent_path if parent_path else "model",
                "Module",
                path.rsplit(".", 1)[-1],
                {"pythonType": type(module).__qualname__},
                "observed",
            )
        refs = []
        for local, param in module.named_parameters(recurse=False):
            full = path + "." + local if path else local
            if id(param) in parameter_ids:
                refs.append(parameter_ids[id(param)])
                continue
            parameter_ids[id(param)] = full
            b.tensor(
                full,
                node_id,
                list(param.shape),
                str(param.dtype).removeprefix("torch."),
                evidence="observed",
            )
        if refs:
            b.entities[node_id]["tensorRefs"] = refs
        for local, buffer in module.named_buffers(recurse=False):
            full = (path + "." if path else "") + local
            b.tensor(
                "buffer:" + full,
                node_id,
                list(buffer.shape),
                str(buffer.dtype).removeprefix("torch."),
                role="buffer",
                evidence="observed",
            )
    return validate(b.ir)


def from_onnx(path):
    """Read an ONNX graph, without loading external tensor data or running it."""
    try:
        import onnx
    except ImportError as e:
        raise RuntimeError("Optional ONNX adapter needs: pip install onnx") from e
    model = onnx.load_model(str(path), load_external_data=False)
    b = Builder(
        Path(path).name,
        {
            "kind": "ONNX graph",
            "evidence": "observed",
            "notes": "Declared graph dependencies; no inference or external tensor data loaded. Inline ONNX protobuf tensor payloads are parsed by ONNX but not rendered. Initializer roles may include constants as well as learned parameters.",
        },
    )
    b.node(
        "initializers", "model", "ParameterVolume", "Initializers", evidence="observed"
    )
    initializers = {x.name for x in model.graph.initializer}
    for t in model.graph.initializer:
        b.tensor(
            t.name,
            "initializers",
            list(t.dims),
            onnx.TensorProto.DataType.Name(t.data_type),
            role="unknown",
            evidence="observed",
        )
    producers = {}
    for i, value in enumerate(model.graph.input):
        if value.name in initializers:
            continue
        id = f"input:{i}"
        b.node(
            id, "model", "TokenStream", value.name, {"role": "graph input"}, "observed"
        )
        producers[value.name] = id
    for i, node in enumerate(model.graph.node):
        id = f"op:{i}"
        entity = b.node(
            id,
            "model",
            "Module",
            node.name or node.op_type,
            {"operation": node.op_type, "domain": node.domain},
            "observed",
        )
        entity["tensorRefs"] = [x for x in node.input if x in initializers]
        for output in node.output:
            producers[output] = id
    for i, node in enumerate(model.graph.node):
        for value in node.input:
            if value in producers:
                b.edge(producers[value], f"op:{i}", value, evidence="observed")
    for i, value in enumerate(model.graph.output):
        id = f"output:{i}"
        b.node(
            id, "model", "TokenStream", value.name, {"role": "graph output"}, "observed"
        )
        if value.name in producers:
            b.edge(producers[value.name], id, value.name, evidence="observed")
    return validate(b.ir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundled", action="store_true")
    mode.add_argument("--config", type=Path)
    mode.add_argument("--safetensors", type=Path)
    mode.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bundled:
        for name, adapter in [
            ("gpt2", gpt2),
            ("deepseek4", deepseek4),
            ("tiny", tiny_example),
        ]:
            print(
                json.dumps(
                    write_package(adapter(), args.output / f"{name}.atlas.json.gz")
                )
            )
        return
    if args.config:
        config = json.loads(args.config.read_text())
        if config.get("model_type") == "gpt2":
            ir = gpt2(config)
        elif config == json.loads(
            (ROOT / "references/deepseek4-config.json").read_text()
        ):
            ir = deepseek4()
        else:
            raise ValueError(
                "No verified adapter for this config. Use module/ONNX/header inspection; do not invent topology from model names."
            )
    elif args.safetensors:
        ir = safetensors_header(args.safetensors)
    else:
        ir = from_onnx(args.onnx)
    print(json.dumps(write_package(ir, args.output)))


if __name__ == "__main__":
    main()
