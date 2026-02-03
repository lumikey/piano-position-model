import { InferenceSession, Tensor } from "onnxruntime-common";

const BLACK_KEY_NOTES = new Set([1, 4, 6, 9, 11]); // A#, C#, D#, F#, G#

function isBlackKey(midi: number): number {
  if (midi < 0) return -1.0;
  const keyIndex = midi - 21;
  return BLACK_KEY_NOTES.has(((keyIndex % 12) + 12) % 12) ? 1.0 : 0.0;
}

export interface HandPosition {
  wrist: [number, number];
  fingertips: [[number, number], [number, number], [number, number], [number, number], [number, number]];
}

export interface Note {
  left: boolean;
  note: number;
  time: number;
  duration: number;
  finger: number; // 1-5
  velocity?: number;
}

interface LookaheadNote {
  note: number;
  finger: number;
  timeUntil: number;
}

/**
 * Build 21x4 input tokens for hand position prediction.
 * Returns a flat Float32Array of 84 values in row-major order.
 */
function buildTokens(
  prevHand: HandPosition,
  currentNotes: { note: number; finger: number }[],
  lookahead: LookaheadNote[]
): Float32Array {
  const tokens = new Float32Array(21 * 4);

  // --- Previous hand tokens (0-5) ---
  // Wrist (token 0)
  tokens[0 * 4 + 0] = prevHand.wrist[0];
  tokens[0 * 4 + 1] = prevHand.wrist[1];
  tokens[0 * 4 + 2] = 0.0; // part_id
  tokens[0 * 4 + 3] = 0.0; // token_type

  // Fingertips (tokens 1-5)
  for (let f = 0; f < 5; f++) {
    const offset = (1 + f) * 4;
    tokens[offset + 0] = prevHand.fingertips[f][0];
    tokens[offset + 1] = prevHand.fingertips[f][1];
    tokens[offset + 2] = (f + 1) / 5.0;
    tokens[offset + 3] = 0.0;
  }

  // --- Current note tokens (6-10): one per finger slot ---
  const fingerToMidi = new Map<number, number>();
  for (const note of currentNotes) {
    fingerToMidi.set(note.finger - 1, note.note);
  }

  for (let f = 0; f < 5; f++) {
    const offset = (6 + f) * 4;
    const midi = fingerToMidi.get(f);

    if (midi !== undefined) {
      tokens[offset + 0] = (midi - 21) / 87.0;
      tokens[offset + 1] = isBlackKey(midi);
      tokens[offset + 2] = 1.0; // is_active
    } else {
      tokens[offset + 0] = -1.0;
      tokens[offset + 1] = -1.0;
      tokens[offset + 2] = -1.0;
    }
    tokens[offset + 3] = 0.33;
  }

  // --- Lookahead tokens (11-20) ---
  for (let k = 0; k < 10; k++) {
    const offset = (11 + k) * 4;

    if (k < lookahead.length) {
      const ln = lookahead[k];
      tokens[offset + 0] = (ln.note - 21) / 87.0;
      tokens[offset + 1] = isBlackKey(ln.note);
      tokens[offset + 2] = (ln.finger - 1) / 4.0;
      tokens[offset + 3] = Math.min(ln.timeUntil, 10.0) / 10.0;
    } else {
      tokens[offset + 0] = -1.0;
      tokens[offset + 1] = -1.0;
      tokens[offset + 2] = -1.0;
      tokens[offset + 3] = -1.0;
    }
  }

  return tokens;
}

function parseOutput(data: Float32Array): HandPosition {
  return {
    wrist: [data[0], data[1]],
    fingertips: [
      [data[2], data[3]],
      [data[4], data[5]],
      [data[6], data[7]],
      [data[8], data[9]],
      [data[10], data[11]],
    ],
  };
}

interface NoteGroup {
  time: number;
  notes: { note: number; finger: number }[];
}

function groupNotesByTime(notes: Note[], thresholdMs = 1.0): NoteGroup[] {
  if (notes.length === 0) return [];

  const sorted = [...notes].sort((a, b) => a.time - b.time);
  const groups: NoteGroup[] = [];
  let currentGroup: NoteGroup = { time: sorted[0].time, notes: [] };

  for (const note of sorted) {
    if (note.time - currentGroup.time <= thresholdMs) {
      currentGroup.notes.push({ note: note.note, finger: note.finger });
    } else {
      groups.push(currentGroup);
      currentGroup = { time: note.time, notes: [{ note: note.note, finger: note.finger }] };
    }
  }
  groups.push(currentGroup);

  return groups;
}

export interface PositionResult {
  time: number;
  notes: { note: number; finger: number }[];
  position: HandPosition;
}

/**
 * Predict hand positions for all chords of one hand.
 */
export async function predictHandPositions(
  notes: Note[],
  isLeft: boolean,
  model: InferenceSession
): Promise<PositionResult[]> {
  const handNotes = notes.filter((n) => n.left === isLeft);

  if (handNotes.length === 0) return [];

  const groups = groupNotesByTime(handNotes);

  // Initialize hand position at center
  let prevHand: HandPosition = {
    wrist: [0.5, 1.5],
    fingertips: [
      [0.46, 0.8],
      [0.48, 0.6],
      [0.5, 0.5],
      [0.52, 0.6],
      [0.54, 0.8],
    ],
  };

  const results: PositionResult[] = [];

  for (let i = 0; i < groups.length; i++) {
    const group = groups[i];
    const currentTime = group.time;

    // Build lookahead from future groups
    const lookahead: LookaheadNote[] = [];
    for (let j = i + 1; j < groups.length && lookahead.length < 10; j++) {
      const futureGroup = groups[j];
      const timeUntil = (futureGroup.time - currentTime) / 1000.0;

      for (const note of futureGroup.notes) {
        lookahead.push({
          note: note.note,
          finger: note.finger,
          timeUntil,
        });
        if (lookahead.length >= 10) break;
      }
    }

    // Build tokens and run model
    const tokenData = buildTokens(prevHand, group.notes, lookahead);
    const inputTensor = new Tensor("float32", tokenData, [1, 21, 4]);
    const output = await model.run({ tokens: inputTensor });
    const positions = output.positions.data as Float32Array;
    const handPos = parseOutput(positions);

    results.push({
      time: currentTime,
      notes: group.notes,
      position: handPos,
    });

    prevHand = handPos;
  }

  return results;
}

export type Models = { left: InferenceSession; right: InferenceSession };

export interface PredictionResult {
  left: PositionResult[];
  right: PositionResult[];
}

/**
 * Predict hand positions for a sequence of fingered piano notes.
 *
 * @param notes - Array of notes with finger assignments (1-5)
 * @param models - Pre-loaded ONNX inference sessions for left and right hand models
 * @returns Hand positions for each chord, grouped by hand
 */
export async function predictPositions(
  notes: Note[],
  models: Models
): Promise<PredictionResult> {
  const [leftResults, rightResults] = await Promise.all([
    predictHandPositions(notes, true, models.left),
    predictHandPositions(notes, false, models.right),
  ]);

  return {
    left: leftResults,
    right: rightResults,
  };
}
