"""Export trained PyTorch checkpoints to ONNX format.

Outputs .onnx files to js/models/ for inclusion in the npm package.
"""

from pathlib import Path
import torch
from model import HandPositionTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
JS_MODELS_DIR = PROJECT_ROOT / "js" / "models"


def export_model(hand: str):
    """Export a trained model to ONNX format."""
    pt_path = CHECKPOINTS_DIR / f"hand_position_{hand}.pt"
    JS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = JS_MODELS_DIR / f"hand_position_{hand}.onnx"

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)

    model = HandPositionTransformer(
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        dim_feedforward=checkpoint.get("dim_feedforward", 128),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Input: (batch, 21 tokens, 4 features)
    dummy_input = torch.randn(1, 21, 4)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["tokens"],
        output_names=["positions"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "positions": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    for hand in ["left", "right"]:
        export_model(hand)
