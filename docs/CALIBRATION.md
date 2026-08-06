# Calibration and validation

Predicted-vs-published comparison tables, fitted constants, and error analysis.

**Rules (CLAUDE.md):** every number here comes from a citable published source
(MLPerf Inference results, vendor datasheets/blogs, peer-reviewed papers). Where no
reference exists, the entry reads "no reference point available" — never a plausible guess.

## Status

No calibration data yet. The validation harness and first reference points land in
Phase 1, Session 5 (see the approved plan). Planned sources:

- MLPerf Inference (Llama-2-70B on H100, MI300X)
- NVIDIA / AMD vendor benchmark blogs (Llama-3-8B, Mistral-7B)
- Community vLLM benchmark reports (secondary points)

## Fitted constants

None yet. Each fitted value in `backend/bwz/calibration.py` will be documented here
with the dataset it was fitted against and the fit residuals.
