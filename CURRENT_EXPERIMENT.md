V5C — Span Boundary Probe
Most recent experiment

Branch: experiment/span-boundary-probe-v5c

Purpose: Investigate whether the model's span-end predictions generalize across name lengths and vowel/consonant endings.

Important design constraint: This is a read-only diagnostic. Its evaluation data must not be incorporated into training.

Core files:
- generate_span_probe_v5c.py
- evaluate_span_probe_v5c.py
- test_span_boundary_probe_v5c.py
- The existing experiment README.md
Latest results discussed: High exact-match accuracy on vowel-ending names, but 0/24 exact matches in each consonant-ending group of your expanded probe.