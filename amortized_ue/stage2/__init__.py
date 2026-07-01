"""Stage 2: amortized uncertainty proxy.

Trains a small decoder-only LM to predict a large LLM's continuous semantic
entropy from its stored hidden state (injected as soft tokens) plus optional
text, in a single forward pass. Consumes Stage-1 records read-only; edits nothing
under semantic_uncertainty/ or the Stage-1 builder.

Code is self-contained in this subpackage. See amortized_ue/CLAUDE.md and the
Stage-2 locked-decision list for the design.
"""
