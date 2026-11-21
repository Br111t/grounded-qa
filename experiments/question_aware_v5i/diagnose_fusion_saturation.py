"""Read-only V5I fusion decomposition and tanh-saturation diagnostic.

Uses only registered TRAIN/DEV and the existing V5I checkpoint.
Does not train, alter the model, or access reserved EVAL.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5i.diagnose_question_conditioning import forward_parts
from grounded_qa.models.span_model_v5i import load_v5i_checkpoint


def describe(t):
    t = t.detach().float().reshape(-1)
    return {
        "mean": float(t.mean()),
        "mean_abs": float(t.abs().mean()),
        "rms": float(t.square().mean().sqrt()),
        "max_abs": float(t.abs().max()),
    }


@torch.no_grad()
def measure_row(model, row):
    parts = forward_parts(model, row)
    h = parts["hidden"][0]
    q = parts["question"][0]
    mask = parts["mask"][0]  # Report evidence-token positions, not padding.
    w = model.question_fusion.weight
    d = model.config.d_model
    wh, wq = w[:, :d], w[:, d:]
    token_term = F.linear(h, wh)
    question_term = F.linear(q, wq)
    bias = model.question_fusion.bias
    z = token_term + question_term + bias
    a = torch.tanh(z)
    slope = 1 - a.square()
    te, ze, ae, se = (v[mask] for v in (token_term, z, a, slope))
    qe = question_term.unsqueeze(0).expand_as(h)[mask]
    # Exact counterfactual: remove ONLY W_q q, preserving the learned bias
    # and token features; not a valid real-world no-question prediction.
    without_q = torch.tanh(token_term + bias)
    q_impact = (a - without_q)[mask]
    start_w = model.start_head.weight[0]
    end_w = model.end_head.weight[0]
    # Question contribution AFTER tanh projected through each readout.
    start_impact = q_impact @ start_w
    end_impact = q_impact @ end_w
    return {
        "source_id": row["source_id"],
        "target_field": row["target_field"],
        "token_term": describe(te),
        "question_term": describe(qe),
        "preactivation": describe(ze),
        "activation": describe(ae),
        "slope": describe(se),
        "fraction_abs_z_ge_2": float((ze.abs() >= 2).float().mean()),
        "fraction_abs_z_ge_3": float((ze.abs() >= 3).float().mean()),
        "fraction_slope_lt_0_1": float((se < 0.1).float().mean()),
        "fraction_slope_lt_0_01": float((se < 0.01).float().mean()),
        "question_to_token_rms_ratio": float(
            qe.square().mean().sqrt() / te.square().mean().sqrt().clamp_min(1e-12)
        ),
        "tanh_question_impact": describe(q_impact),
        "start_score_question_impact": describe(start_impact),
        "end_score_question_impact": describe(end_impact),
    }


def mean_metric(rows, *path):
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(value)
    return sum(values) / len(values) if values else None


def run(checkpoint, output_dir):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Use a new output directory; overwriting forbidden")
    model, metadata = load_v5i_checkpoint(checkpoint)
    if metadata.get("experiment_variant") != "v5i_question_conditioned":
        raise ValueError("Expected trained V5I checkpoint")
    train_sha = hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest()
    if metadata.get("training_sha256") != train_sha:
        raise ValueError("Checkpoint's TRAIN SHA does not match registered data")
    model.eval()
    w = model.question_fusion.weight.detach()
    d = model.config.d_model
    weight_stats = {
        "token_weight": describe(w[:, :d]),
        "question_weight": describe(w[:, d:]),
        "bias": describe(model.question_fusion.bias.detach()),
        "token_weight_frobenius": float(torch.linalg.matrix_norm(w[:, :d])),
        "question_weight_frobenius": float(torch.linalg.matrix_norm(w[:, d:])),
    }
    results = {}
    for split in ("train", "dev"):
        rows = read_partition(split)
        measurements = [measure_row(model, row) for row in rows]
        report_keys = {
            "token_term_rms": ("token_term", "rms"),
            "question_term_rms": ("question_term", "rms"),
            "preactivation_mean_abs": ("preactivation", "mean_abs"),
            "preactivation_max_abs": ("preactivation", "max_abs"),
            "tanh_activation_mean_abs": ("activation", "mean_abs"),
            "tanh_derivative_mean": ("slope", "mean"),
            "fraction_abs_z_ge_2": ("fraction_abs_z_ge_2",),
            "fraction_abs_z_ge_3": ("fraction_abs_z_ge_3",),
            "fraction_derivative_below_0_1": ("fraction_slope_lt_0_1",),
            "fraction_derivative_below_0_01": ("fraction_slope_lt_0_01",),
            "question_to_token_rms_ratio": ("question_to_token_rms_ratio",),
            "question_effect_after_tanh_rms": ("tanh_question_impact", "rms"),
            "start_logit_question_effect_mean_abs": ("start_score_question_impact", "mean_abs"),
            "end_logit_question_effect_mean_abs": ("end_score_question_impact", "mean_abs"),
        }
        summary = {name: mean_metric(measurements, *keys)
                   for name, keys in report_keys.items()}
        results[split] = {
            "sha256": hashlib.sha256((DATA / f"{split}.jsonl").read_bytes()).hexdigest(),
            "rows": measurements,
            "summary": summary,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    document = {
        "diagnostic": "v5i_fusion_tanh_saturation",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "no_training": True,
        "reserved_eval_accessed": False,
        "weight_stats": weight_stats,
        "results": results,
        "notes": [
            "Metrics computed over evidence-token positions; each example contributes equally.",
            "Question term W_q q is constant across token positions within an example.",
            "Question effect after tanh removes W_q q while holding H and b fixed.",
            "This is a mathematical ablation, not a realistic blank-question input.",
            "Small activation derivatives can indicate saturation, but do not alone prove the cause of DEV failure.",
            "Readout effects on start/end logits need not change token ranking even if nonzero.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# V5I fusion and tanh diagnostic",
        "",
        "Read-only. TRAIN/DEV only. No training or reserved EVAL.",
        "",
        "## Learned fusion weights",
        "",
        f"- Token block Frobenius norm: {weight_stats['token_weight_frobenius']:.6f}",
        f"- Question block Frobenius norm: {weight_stats['question_weight_frobenius']:.6f}",
        "",
        "## Evidence-token activation diagnostics",
        "",
        "| Metric (mean across examples) | TRAIN | DEV |",
        "| --- | ---: | ---: |",
    ]
    for key in results["train"]["summary"]:
        lines.append(f"| {key} | {results['train']['summary'][key]:.6f} | {results['dev']['summary'][key]:.6f} |")
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- High fraction `abs(z) >= 2` or derivative below 0.1 suggests tanh saturation.",
        "- Low question/token RMS ratio suggests small question input relative to token input.",
        "- Small post-tanh question effect despite a sizable question term suggests compression, possibly saturation.",
        "- Question effects on logits do not necessarily imply a change in the selected field.",
        "- This diagnostic does not establish the causal origin of generalization failure.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
