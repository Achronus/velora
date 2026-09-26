# Type stubs

`mujoco` ships its `MjSpec` / `MjModel` / `MjData` bindings as compiled
pybind11 extensions (`_specs.pyd`, `_structs.pyd`, ...) with no `py.typed`
marker and no `.pyi` files, so editors resolve every member as `Unknown`.

These stubs are generated from the installed package's embedded pybind11
signatures.

## Regenerating

Pinned to `mujoco==3.10.0` (via the `mjlab` extra). After a `mujoco` upgrade:

```bash
uv run --no-sync --with pybind11-stubgen pybind11-stubgen mujoco -o typings/
```

The generator emits a handful of signatures that are not valid Python. Reapply
these fixes to `mujoco/_specs.pyi` afterwards:

| Symbol | Problem | Fix |
| --- | --- | --- |
| `MjSpec.resolve_orientation` | `sequence` defaults before non-defaulted `orientation` | drop the `= None` |
| `MjsActuator.set_to_muscle` | `tausmooth` has no default between defaulted args | add `= 0` |
| `MjVisual.global` | `global` is a reserved word | delete the line, keep `global_` |
| `MjSpec.compile` | bound as `-> object` | narrow to `mujoco._structs.MjModel` |

Verify with:

```bash
uv run --no-sync python -c "import ast,pathlib;[ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('typings').rglob('*.pyi')]"
```
