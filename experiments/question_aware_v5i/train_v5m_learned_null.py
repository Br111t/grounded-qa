"""V5M: freeze V5J bilinear; learn a question-conditioned null-answer score.

Null(q,H) = linear([pooled question embedding, contextual BOS]).
Answer iff best candidate joint score exceeds Null. Only 129 new parameters.
TRAIN-only BCE optimization and feature normalization. No paired question or
gold requested field needed at inference. DEV exploratory; EVAL sealed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.field_boundary_decoder_v5e.field_decoder import complete_field_spans
from experiments.question_aware_v5g.data_design import DATA, read_partition
from experiments.question_aware_v5g.evaluate import enriched
from experiments.question_aware_v5i.diagnose_bilinear_answerability import load_bilinear_checkpoint
from experiments.question_aware_v5i.train_joint_a_prime import scorecard
from grounded_qa.models.span_model_v5 import prepare_span_input


@torch.no_grad()
def encode_one(model, row):
    """Extract contextual question/BOS vectors from frozen encoder output.

    The question vector uses the actual prompt question token mask. Field labels
    and answerability targets are not used to construct inference features.
    """
    ids, mask = prepare_span_input(row["information"], row["question"])
    captured = []
    def grab(_module, _inputs, output):
        captured.append(output.detach())
    handle = model.encoder.register_forward_hook(grab)
    try:
        start, end, _a = model(ids, mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one encoder output")
    hidden = captured[0]
    qmask = model.question_mask(ids, mask)
    q = (hidden * qmask.unsqueeze(-1)).sum(1) / qmask.sum(1, keepdim=True)
    context = torch.cat((q[0], hidden[0, 0]), dim=0).double().cpu()
    candidates = complete_field_spans(row["information"], model.config.max_span_bytes)
    if not candidates:
        raise ValueError("No candidate spans")
    scored = sorted(
        [(float(start[0,c["start"]] + end[0,c["end"]]), c) for c in candidates],
        key=lambda pair: pair[0], reverse=True,
    )
    return {"features": context, "max_joint": scored[0][0],
            "best_candidate": scored[0][1],
            "answerable": row["expected_value"] is not None}


def fit_null(train, *, steps=1500, lr=0.01):
    if steps < 1 or lr <= 0:
        raise ValueError("Invalid training settings")
    x = torch.stack([r["features"] for r in train])
    scores = torch.tensor([r["max_joint"] for r in train], dtype=torch.float64)
    labels = torch.tensor([float(r["answerable"]) for r in train], dtype=torch.float64)
    if x.shape != (48,128) or int(labels.sum()) != 36:
        raise ValueError("Unexpected registered TRAIN shapes")
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (x - mean) / std
    null_head = nn.Linear(x.shape[1], 1).double()
    nn.init.zeros_(null_head.weight)
    nn.init.zeros_(null_head.bias)
    optimizer = torch.optim.Adam(null_head.parameters(), lr=lr)
    losses = []
    for step in range(steps + 1):
        logits = scores - null_head(standardized).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if step in (0, steps):
            losses.append({"step": step, "train_bce": float(loss.detach())})
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in null_head.parameters()):
            raise ValueError("Nonfinite null-head parameters")
    null_head.eval()
    return null_head, mean, std, losses


@torch.no_grad()
def evaluate(rows, representations, null_head, mean, std):
    outcomes = []
    for row, record in zip(rows, representations):
        candidate = record["best_candidate"]
        feature = ((record["features"] - mean) / std).unsqueeze(0)
        null_score = float(null_head(feature).squeeze())
        logit = record["max_joint"] - null_score
        p = float(torch.sigmoid(torch.tensor(logit, dtype=torch.float64)))
        selected = candidate if p >= 0.5 else None
        prediction = {
            "predicted_field": selected["field"].lower() if selected else None,
            "predicted_value": selected["value"] if selected else None,
            "answerability_probability": p,
            "candidate_fields": [],
            "field_scores": {},
        }
        outcomes.append(enriched(row, prediction))
    return scorecard(outcomes)


def run(checkpoint, checkpoint_out, output_dir, *, steps=1500, lr=0.01):
    for p in (checkpoint_out, output_dir):
        if p.exists() or p.is_symlink():
            raise FileExistsError(f"Output already exists: {p}")
    torch.set_num_threads(1)
    model, original_meta = load_bilinear_checkpoint(checkpoint)
    model.requires_grad_(False)
    model.eval()
    train_rows, dev_rows = read_partition("train"), read_partition("dev")
    train = [encode_one(model, row) for row in train_rows]
    dev = [encode_one(model, row) for row in dev_rows]
    null_head, mean, std, losses = fit_null(train, steps=steps, lr=lr)
    metrics = {
        "train": evaluate(train_rows, train, null_head, mean, std),
        "dev": evaluate(dev_rows, dev, null_head, mean, std),
    }
    params = sum(p.numel() for p in null_head.parameters())
    if params != 129:
        raise ValueError("Expected a small 129-parameter null head")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "experiment": "v5m_learned_question_conditioned_null",
        "bilinear_checkpoint_sha256": sha,
        "train_sha256": hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256((DATA / "dev.jsonl").read_bytes()).hexdigest(),
        "null_head_parameters": params, "steps": steps, "learning_rate": lr,
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "null_head_state": {k: v.detach().cpu().clone()
                            for k,v in null_head.state_dict().items()},
        "training_loss": losses, "scores": metrics,
        "frozen_bilinear": True, "single_question_inference": True,
        "reserved_eval_accessed": False,
    }
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_out.open("xb") as f:
        torch.save(payload, f)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {k:v for k,v in payload.items() if k not in
              ("feature_mean","feature_std","null_head_state")}
    report["null_checkpoint_sha256"] = hashlib.sha256(checkpoint_out.read_bytes()).hexdigest()
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# V5M — Learned question-conditioned null answer", "",
        "Frozen V5J bilinear. A 129-parameter linear null head is TRAIN-fitted",
        "from pooled contextual question and BOS vectors. Single-question inference;",
        "no paired question, answer label, or requested field as an input.",
        "Decision: sigmoid(max candidate joint − learned null score) >= 0.5.",
        "TRAIN-only per-dimension feature normalization; fixed decision threshold.",
        "DEV exploratory, EVAL sealed.", "",
        f"Null-head parameters: {params}; TRAIN BCE: {losses[0]['train_bce']:.5f} → {losses[-1]['train_bce']:.5f}",
        "", "| Metric | V5K DEV reference | V5M DEV |",
        "| --- | ---: | ---: |",
    ]
    baseline = {
        "dev_exact":45, "answerable_correct":33,"unanswerable_correct":12,
        "unsupported_answer_errors":0,"false_abstentions":3,
        "paired_both_correct":21,"correct_switches":10,
    }
    for key, value in baseline.items():
        lines.append(f"| {key} | {value} | {metrics['dev'][key]} |")
    lines += [
        "", "## Interpretation", "",
        "- New null head adds parameters, not a full encoder retraining.",
        "- A high DEV score after repeated inspection does not establish generalization.",
        "- The frozen encoder was trained on both questions; V5M inference uses only the current question.",
        "- Do not change thresholds using DEV; inspect on a separately registered dataset after this.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--null-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.01)
    a = p.parse_args()
    run(a.checkpoint, a.null_checkpoint, a.output_dir, steps=a.steps, lr=a.lr)


if __name__ == "__main__":
    main()
