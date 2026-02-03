"""
Hand position inference from fingered notes.

Reads a JSON file of fingered notes and predicts hand positions.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from model import HandPositionTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BLACK_KEY_NOTES = [1, 4, 6, 9, 11]


def is_black_key(midi: int) -> float:
    if midi < 0:
        return -1.0
    return 1.0 if ((midi - 21) % 12) in BLACK_KEY_NOTES else 0.0


def load_model(hand: str):
    """Load trained hand position transformer."""
    model_path = PROJECT_ROOT / "checkpoints" / f"hand_position_{hand}.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    model = HandPositionTransformer(
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        dim_feedforward=checkpoint.get('dim_feedforward', 128)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def group_notes_by_time(notes: list[dict], threshold_ms: float = 1.0) -> list[list[dict]]:
    """Group simultaneous notes into chords."""
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: n['time'])
    groups = []
    current_group = [notes[0]]

    for note in notes[1:]:
        if note['time'] - current_group[0]['time'] <= threshold_ms:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]

    groups.append(current_group)
    return groups


def build_tokens(prev_hand: dict, current_notes: list[dict], lookahead: list[dict]) -> np.ndarray:
    """
    Build 21x4 input tokens for hand position prediction.

    Args:
        prev_hand: dict with 'wrist': (x,y) and 'fingertips': [(x,y), ...]
        current_notes: list of notes with 'note' (midi) and 'finger' (1-5)
        lookahead: list of future notes with 'note', 'finger', and 'time_until'
    """
    tokens = np.zeros((21, 4), dtype=np.float32)

    # --- Previous hand tokens (0-5) ---
    tokens[0, 0] = prev_hand['wrist'][0]
    tokens[0, 1] = prev_hand['wrist'][1]
    tokens[0, 2] = 0.0  # part_id
    tokens[0, 3] = 0.0  # token_type

    for f in range(5):
        tokens[1 + f, 0] = prev_hand['fingertips'][f][0]
        tokens[1 + f, 1] = prev_hand['fingertips'][f][1]
        tokens[1 + f, 2] = (f + 1) / 5.0
        tokens[1 + f, 3] = 0.0

    # --- Current note tokens (6-10): one per finger slot ---
    finger_to_midi = {}
    for note in current_notes:
        finger_to_midi[note['finger'] - 1] = note['note']

    for f in range(5):
        if f in finger_to_midi:
            midi = finger_to_midi[f]
            tokens[6 + f, 0] = (midi - 21) / 87.0
            tokens[6 + f, 1] = is_black_key(midi)
            tokens[6 + f, 2] = 1.0  # is_active
        else:
            tokens[6 + f, 0] = -1.0
            tokens[6 + f, 1] = -1.0
            tokens[6 + f, 2] = -1.0
        tokens[6 + f, 3] = 0.33

    # --- Lookahead tokens (11-20) ---
    for k in range(10):
        if k < len(lookahead):
            note = lookahead[k]
            tokens[11 + k, 0] = (note['note'] - 21) / 87.0
            tokens[11 + k, 1] = is_black_key(note['note'])
            tokens[11 + k, 2] = (note['finger'] - 1) / 4.0
            tokens[11 + k, 3] = min(note['time_until'], 10.0) / 10.0
        else:
            tokens[11 + k, :] = -1.0

    return tokens


def predict(model, prev_hand: dict, current_notes: list[dict], lookahead: list[dict]) -> dict:
    """Predict hand position for current notes."""
    tokens = build_tokens(prev_hand, current_notes, lookahead)

    with torch.no_grad():
        tokens_tensor = torch.tensor([tokens], dtype=torch.float32)
        output = model(tokens_tensor).numpy()[0]

    return {
        'wrist': (float(output[0]), float(output[1])),
        'fingertips': [
            (float(output[2]), float(output[3])),
            (float(output[4]), float(output[5])),
            (float(output[6]), float(output[7])),
            (float(output[8]), float(output[9])),
            (float(output[10]), float(output[11])),
        ]
    }


def predict_hand_positions(notes: list[dict], hand: str, model) -> list[dict]:
    """Predict hand positions for all chords of one hand."""
    is_left = (hand == 'left')
    hand_notes = [n for n in notes if n['left'] == is_left]

    if not hand_notes:
        return []

    groups = group_notes_by_time(hand_notes)

    # Initialize hand position at center
    prev_hand = {
        'wrist': (0.5, 1.5),
        'fingertips': [
            (0.46, 0.8),
            (0.48, 0.6),
            (0.50, 0.5),
            (0.52, 0.6),
            (0.54, 0.8),
        ]
    }

    results = []

    for i, group in enumerate(groups):
        current_time = group[0]['time']

        # Build lookahead from future groups
        lookahead = []
        for future_group in groups[i + 1:]:
            future_time = future_group[0]['time']
            time_until = (future_time - current_time) / 1000.0

            for note in future_group:
                lookahead.append({
                    'note': note['note'],
                    'finger': note['finger'],
                    'time_until': time_until
                })
                if len(lookahead) >= 10:
                    break
            if len(lookahead) >= 10:
                break

        # Predict
        hand_pos = predict(model, prev_hand, group, lookahead)

        results.append({
            'time': current_time,
            'notes': [{'note': n['note'], 'finger': n['finger']} for n in group],
            'wrist': hand_pos['wrist'],
            'fingertips': hand_pos['fingertips']
        })

        prev_hand = hand_pos

    return results


def main():
    parser = argparse.ArgumentParser(description="Hand position inference from fingered notes")
    parser.add_argument("input", help="Input JSON file with fingered notes")
    parser.add_argument("-o", "--output", help="Output JSON file (default: input_positions.json)")
    parser.add_argument("--hand", "-H", choices=["left", "right", "both"], default="both",
                        help="Which hand to predict (default: both)")

    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        notes = json.load(f)

    print(f"Loaded {len(notes)} notes")

    results = {'left': [], 'right': []}

    hands = ['left', 'right'] if args.hand == 'both' else [args.hand]

    for hand in hands:
        model = load_model(hand)
        print(f"Loaded {hand} hand model")
        results[hand] = predict_hand_positions(notes, hand, model)
        print(f"Predicted {len(results[hand])} positions for {hand} hand")

    # Save output
    output_path = args.output or args.input.replace('.json', '_positions.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
