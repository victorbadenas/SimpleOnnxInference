import argparse
import sys
from inspect import getmembers, isfunction
from pathlib import Path
from typing import Union

import torch
import torchvision

AVAILABLE_MODELS = dict(getmembers(torchvision.models, isfunction))

# remove models that don't have pretrained versions
for model in [
    "mnasnet0_75",
    "mnasnet1_3",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0"
]:
    AVAILABLE_MODELS.pop(model)


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-m",
        "--model_arch",
        type=str,
        help="model architecture",
        choices=list(AVAILABLE_MODELS.keys()),
        required=True,
    )
    parser.add_argument(
        "-O",
        "--opset_version",
        type=int,
        help="onnx opset version",
        default=13
    )
    parser.add_argument(
        "-o",
        "--out_folder",
        type=Path,
        help="output folder path",
        default=Path("./models/"),
    )
    parser.add_argument(
        "-i",
        "--input_shape",
        nargs="+",
        type=int,
        required=True,
        help="input tensor's shape (not including batch size)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        help="path to checkpoint to load",
        default=None
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="launch script in debug mode",
    )
    return parser.parse_args()


def get_model(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            "model is not available in module."
            f" available models: {AVAILABLE_MODELS.keys()}"
        )
    return AVAILABLE_MODELS[model_name](pretrained=pretrained)


def export_to_onnx(
    model_arch: str,
    input_shape: list,
    out_folder: Union[str, Path],
    checkpoint: Union[str, Path] = None,
    debug: bool = False,
    opset_version: int = 13,
    **kwargs,
):
    input_shape = tuple([1] + input_shape)
    out_folder = Path(out_folder)
    checkpoint = Path(checkpoint) if checkpoint is not None else None

    print(f"running {sys.argv[0]} with parameters:")
    print(f"model_arch: {model_arch}")
    print(f"input_shape: {input_shape}")
    print(f"out_folder: {out_folder}")
    print(f"checkpoint: {checkpoint}")
    print(f"debug: {debug}")

    for k, v in kwargs.items():
        print(f"{k}: {v}")

    model = AVAILABLE_MODELS[model_arch]()
    if isinstance(checkpoint, Path):
        if checkpoint.exists():
            print("loading checkpoint...")
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict)

    out_folder.mkdir(exist_ok=True, parents=True)
    out_path = out_folder / f"{model_arch}.onnx"
    torch.onnx.export(
        model=model,  # model to export
        args=torch.rand(input_shape),  # random tensor of desired shape
        f=out_path,  # output file path
        verbose=True,  # verbose
        opset_version=opset_version,  # onnx opset version arg
    )


if __name__ == "__main__":
    args = parseArgumentsFromCommandLine()
    export_to_onnx(**args.__dict__)
