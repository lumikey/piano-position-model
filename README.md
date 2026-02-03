# Piano Hand Position Model

A Transformer model that predicts hand positions for piano playing. Given a sequence of fingered notes (MIDI + finger assignment), the model outputs wrist and fingertip positions for natural hand placement.

Separate models are trained for left and right hands.

## Project structure

```
python/              Python training and inference
  model.py           Model architecture (HandPositionTransformer)
  train.py           Training script
  inference.py       Batch inference from JSON files
  export_onnx.py     Export PyTorch checkpoints to ONNX format
js/                  TypeScript npm package (@lumikey/piano-position-model)
demo/                Browser demo (GitHub Pages)
checkpoints/         Trained model weights (.pt)
data/                Training datasets (.npz)
```

## Model architecture

The model is a Transformer encoder that predicts hand positions. For each chord, the input is a sequence of 21 tokens (4 features each):

| Tokens | Count | Description |
|--------|-------|-------------|
| Previous hand | 6 | Wrist + 5 fingertip positions from previous prediction |
| Current notes | 5 | One per finger slot: MIDI, black/white key, active flag |
| Lookahead | 10 | Next fingered notes: MIDI, black/white key, finger, time until |

Default hyperparameters: `d_model=64`, `nhead=4`, `num_layers=3`, `dim_feedforward=128` (~100K parameters).

Output: 12 values — wrist (x, y) + 5 fingertips (x, y each).

## Training

Requires Python 3.10+ with PyTorch and NumPy.

```bash
cd python

# Train both hands
python train.py

# Train one hand with custom hyperparameters
python train.py --hand right --d_model 64 --num_layers 3 --epochs 300
```

Training uses:
- AdamW optimizer with weight decay
- MSE loss
- Learning rate reduction on plateau
- Early stopping (patience=50 epochs)

Trained checkpoints are saved to `checkpoints/`.

### Training data

The `data/` directory contains preprocessed datasets in `.npz` format:

- `hand_position_data.npz` — Contains `X_left`, `Y_left`, `X_right`, `Y_right` arrays


## Python inference

```bash
cd python
python inference.py notes_fingered.json -o output.json
```

Input: JSON array of note objects with finger assignments:

```json
[
  {"left": true, "note": 60, "time": 0, "duration": 500, "finger": 1},
  {"left": true, "note": 64, "time": 0, "duration": 500, "finger": 3}
]
```

Output: JSON with predicted hand positions for each chord:

```json
{
  "left": [
    {
      "time": 0,
      "notes": [{"note": 60, "finger": 1}, {"note": 64, "finger": 3}],
      "wrist": [0.45, 1.2],
      "fingertips": [[0.4, 0.5], [0.45, 0.4], ...]
    }
  ],
  "right": []
}
```

## JavaScript / TypeScript

The model is available as an npm package for use in Node.js or the browser. The main entry point is browser-safe — Node.js helpers are in a separate `/node` subpath.

```bash
npm install @lumikey/piano-position-model onnxruntime-node
```

```typescript
import { predictPositions } from "@lumikey/piano-position-model";
import { loadModels } from "@lumikey/piano-position-model/node";

const models = await loadModels();
const result = await predictPositions([
  { left: false, note: 60, time: 0, duration: 500, finger: 1 },
  { left: false, note: 64, time: 0, duration: 500, finger: 3 },
], models);

console.log(result.right[0].position);
// { wrist: [x, y], fingertips: [[x, y], ...] }
```

### Exporting models for JS

To update the ONNX models shipped with the npm package after retraining:

```bash
cd python
python export_onnx.py
```

This reads the `.pt` checkpoints and writes `.onnx` files to `js/models/`.
