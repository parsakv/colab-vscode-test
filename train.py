import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

input_file = Path("input/sample.txt")
text = input_file.read_text()

# fake GPU workload
x = torch.randn(3000, 3000, device=device)
y = x @ x

out = f"""
TEXT:
{text}

DEVICE:
{device}

RESULT SHAPE:
{y.shape}
"""

# IMPORTANT: write to Drive-synced folder
output_dir = Path("/content/drive/MyDrive/colab_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

(output_dir / "result.txt").write_text(out)

print("DONE")