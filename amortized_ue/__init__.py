"""Amortized uncertainty estimation — Stage 1 offline dataset construction.

Stage 1 reuses the SEP repo's sampling, semantic-entropy, and hidden-state logic
(imported read-only from ../semantic_uncertainty) and writes one self-contained,
id-keyed record per prompt to local disk (optionally mirrored to W&B).

Nothing under semantic_uncertainty/ or semantic_entropy_probes/ is modified.
"""
