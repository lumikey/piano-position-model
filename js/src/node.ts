import { InferenceSession } from "onnxruntime-common";
import type { Models } from "./inference";

export { predictPositions, predictHandPositions } from "./inference";
export type { Note, Models, HandPosition, PositionResult, PredictionResult } from "./inference";

let cachedModels: Models | null = null;

/**
 * Load the bundled ONNX models from the package directory.
 *
 * Node.js only — requires `fs` and `path`. For browser usage, create
 * InferenceSession objects manually and pass them to `predictPositions()`.
 *
 * Sessions are cached after the first call.
 */
export async function loadModels(): Promise<Models> {
  if (cachedModels) return cachedModels;

  const fs = await import("fs/promises");
  const path = await import("path");

  const modelsDir = path.resolve(__dirname, "..", "models");

  const [leftBuf, rightBuf] = await Promise.all([
    fs.readFile(path.join(modelsDir, "hand_position_left.onnx")),
    fs.readFile(path.join(modelsDir, "hand_position_right.onnx")),
  ]);

  const [left, right] = await Promise.all([
    InferenceSession.create(leftBuf.buffer as ArrayBuffer),
    InferenceSession.create(rightBuf.buffer as ArrayBuffer),
  ]);

  cachedModels = { left, right };
  return cachedModels;
}
