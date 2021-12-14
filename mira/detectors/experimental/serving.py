# type: ignore
# pylint: disable=exec-used,missing-class-docstring,missing-function-docstring,too-few-public-methods
import types
import typing
import zipfile
import json

import ts
import torch
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext


def import_code(code, name):
    """Import code from a string into a module."""
    module = types.ModuleType(name)
    exec(code, module.__dict__)
    return module


def load_mar_model(filepath: str, gpu_id: int = None):
    """Load a PyTorch model into a customized handler. Used for testing
    handlers to make sure they produce the same output locally as when serving.

    Args:
        filepath: Path to the .MAR file.
        gpu_id: When using a GPU, the ID of the GPU to use (e.g., 0).
    """
    with zipfile.ZipFile(filepath) as z:
        with z.open("MAR-INF/MANIFEST.json", "r") as f:
            manifest = json.loads(f.read().decode("utf-8"))
        handler_name = manifest["model"]["handler"]
        if handler_name.endswith(".py"):
            with z.open(manifest["model"]["handler"]) as f:
                handler_module = import_code(
                    f.read(), manifest["model"]["handler"].split(".py")[0]
                )
        else:
            handler_module = getattr(ts.torch_handler, handler_name)
        candidate_handlers = [
            cls
            for name, cls in handler_module.__dict__.items()
            if isinstance(cls, type)
            and issubclass(cls, ts.torch_handler.vision_handler.VisionHandler)
        ]
        with z.open(manifest["model"]["modelFile"]) as f:
            model_module = import_code(
                f.read(), manifest["model"]["modelFile"].split(".py")[0]
            )
        candidate_models = [
            cls
            for name, cls in model_module.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, torch.nn.Module)
        ]
        if len(candidate_handlers) == 0:
            raise ValueError("No handler found.")
        if len(candidate_models) == 0:
            raise ValueError("No model found.")

        class SimulatedHandler(candidate_handlers[-1]):
            def handle_files(self, filepaths: typing.List[str]):
                data = []
                for filepath in filepaths:
                    with open(filepath, "rb") as f:
                        data.append({"data": f.read()})
                return self.handle(data=data, context=MockContext())

        handler = SimulatedHandler()
        handler.model = candidate_models[-1]()
        # This replaces the "initialize" step in the handler.
        handler.map_location = (
            "cuda" if torch.cuda.is_available() and gpu_id is not None else "cpu"
        )
        handler.device = torch.device(
            handler.map_location + ":" + str(gpu_id)
            if torch.cuda.is_available() and gpu_id is not None
            else handler.map_location
        )
        handler.model.to(handler.device)
        with z.open(manifest["model"]["serializedFile"]) as f:
            handler.model.load_state_dict(
                torch.load(f, map_location=handler.map_location)
            )
        with z.open("index_to_name.json") as f:
            handler.mapping = json.loads(f.read().decode("utf8"))
        handler.model.eval()
        handler.initialized = True
        return handler
