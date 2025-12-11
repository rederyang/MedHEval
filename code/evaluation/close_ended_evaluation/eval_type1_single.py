"""
Single Dataset Evaluation Script for MedHEval Close-ended Tasks

Reuses eval_closed() from type1_utils.py for consistency.

Usage:
    python eval_single.py \
        --question-file /path/to/qa_pairs.json \
        --prediction-file /path/to/model_predictions.jsonl \
        --output-file /path/to/results.txt
"""

import json
import argparse
from utils.type1_utils import eval_closed


def format_results(avg_acc, lens, om_accs, om_lens):
    """Format results as readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("MedHEval Single Dataset Evaluation Results")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall accuracy (weighted average across all types)
    total = sum(lens)
    overall_acc = sum(avg_acc * lens) / total if total > 0 else 0
    lines.append(f"Overall Accuracy: {overall_acc:.4f} (n={total})")
    lines.append("")
    
    lines.append("Accuracy by Hallucination Type:")
    lines.append("-" * 40)
    type_names = {
        0: ("type_1", "Anatomical Hallucination"),
        1: ("type_2", "Measurement Hallucination"), 
        2: ("type_3", "Symptom-Based Hallucination"),
        3: ("type_4", "Technique Hallucination")
    }
    for i, (t, name) in type_names.items():
        if lens[i] > 0:
            lines.append(f"  {t} ({name}): {avg_acc[i]:.4f} (n={int(lens[i])})")
    
    lines.append("")
    lines.append("Omission Detection Accuracy:")
    lines.append("-" * 40)
    om_type_names = {0: "type_1", 1: "type_3"}
    for i, t in om_type_names.items():
        if om_lens[i] > 0:
            lines.append(f"  {t}: {om_accs[i]:.4f} (n={int(om_lens[i])})")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate single dataset for MedHEval close-ended tasks.')
    parser.add_argument('--question-file', type=str, required=True,
                        help='Path to the original QA pairs JSON file (ground truth)')
    parser.add_argument('--prediction-file', type=str, required=True,
                        help='Path to the model predictions JSONL file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save evaluation results')
    args = parser.parse_args()
    
    # Load original data (ground truth)
    print(f"Loading question file: {args.question_file}")
    with open(args.question_file, 'r') as f:
        ori_data = json.load(f)
    
    # Build id to original data mapping
    id_to_ori = {item['qid']: item for item in ori_data}
    
    # Load predictions
    print(f"Loading prediction file: {args.prediction_file}")
    with open(args.prediction_file, 'r') as f:
        predictions = [json.loads(line) for line in f]
    
    print(f"Evaluating {len(predictions)} predictions against {len(ori_data)} questions...")
    
    # Evaluate using existing eval_closed function
    avg_acc, lens, om_accs, om_lens = eval_closed(ori_data, id_to_ori, predictions)
    
    # Format and save
    output_text = format_results(avg_acc, lens, om_accs, om_lens)
    print(output_text)
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write(output_text)
        f.write("\n\nRaw Results:\n")
        f.write(f"avg_acc (type_1, type_2, type_3, type_4): {avg_acc.tolist()}\n")
        f.write(f"lens: {lens.tolist()}\n")
        f.write(f"om_accs (type_1, type_3): {om_accs.tolist()}\n")
        f.write(f"om_lens: {om_lens.tolist()}\n")
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
