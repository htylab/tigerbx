# TigerBx Release Notes — v0.3.1

## Changes

- **Lower-memory registration**: `tigerbx.norm`, `tigerbx.reg('AD')` and the executable's `--norm` use
  `mprage_reg_v005_vfa`. It has the same weights as v004 and gives the same warps; the feature matching is computed
  one neighbour shift at a time, so the peak memory is much lower. Downloaded automatically on first use.
- Agent skill: `skills/tigerbx/SKILL.md` is one self-contained page that links to the docs.
- `doc/bx.md`: `run()` output table fixed (`a` -> `_aseg`, `t` -> `_tseg`).

Results are the same as 0.3.0; 0.3.0 and 0.3.1 outputs can be mixed in one study.
