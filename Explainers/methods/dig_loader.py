"""Load the pinned DIG attribution modules."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types
import torch

_DIG_SPEC = importlib.util.find_spec("dig")
DIG_ROOT = Path(_DIG_SPEC.origin).resolve().parent.parent if _DIG_SPEC and _DIG_SPEC.origin else None


def load_dig():
    """Execute the original modules under an isolated package namespace."""
    if DIG_ROOT is None:
        raise ImportError("Install Explainers/requirements.txt to run DIG explainers.")
    prefix = "_gnn_fidelity_dig"
    if f"{prefix}.method.pgexplainer" in sys.modules:
        return tuple(sys.modules[f"{prefix}.method.{name}"] for name in
                     ("gnnexplainer", "pgexplainer"))
    sys.path.insert(0, str(DIG_ROOT))  # original GNNExplainer imports dig.version
    for name, relative in ((prefix, "dig/xgraph"),
                           (prefix + ".models", "dig/xgraph/models"),
                           (prefix + ".method", "dig/xgraph/method")):
        package = types.ModuleType(name)
        package.__path__ = [str(DIG_ROOT / relative)]
        package.__package__ = name
        sys.modules[name] = package
    # The base explainer imports this marker class for isinstance checks only.
    # Load its exact declaration without importing unused SparseTensor models.
    source = DIG_ROOT / "dig/xgraph/models/models.py"
    declaration = next(n for n in ast.parse(source.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == "GNNPool")
    namespace = sys.modules[prefix + ".models"].__dict__
    namespace["nn"] = torch.nn
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(source), "exec"), namespace)
    for name in ("models.utils", "method.base_explainer", "method.gnnexplainer",
                 "method.pgexplainer"):
        path = DIG_ROOT / "dig/xgraph" / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(prefix + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return load_dig()
