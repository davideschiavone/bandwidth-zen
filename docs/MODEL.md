# The analytical model

Every formula the engine uses, with derivation and limits of validity.
Filled in milestone by milestone; the authoritative build spec is `PROMPT.md` §3.

## Status

| Section | Milestone | Status |
|---|---|---|
| Roofline (flat: compute ridge vs DRAM ridge) | M3 | not yet implemented |
| Roofline (hierarchical, multi-level) | M8, on demand | not yet implemented |
| FLOP/byte counts (transformer, CNN) | M2 | not yet implemented |
| Tiling and DRAM traffic | M3 | not yet implemented |
| Multi-chip sharding and collectives | M5 | not yet implemented |
| Memory capacity planning | M3 | not yet implemented |
| Power/energy | M7 | not yet implemented |
| Bottleneck classification | M3 | not yet implemented |

The v1 roofline is flat — one compute ridge and one DRAM ridge, with the on-chip SRAM entering
only as a real tile-buffer capacity that constrains tiling. The multi-level hierarchy is deferred
to M8 (see `docs/CORRECTIONS.md` D5).
