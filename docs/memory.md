[← back to the README](../README.md)

# Memory needs

Execution time and memory usage depend on the input Field Of View (FOV), not much on the input resolution
(the image is resampled to the model resolution). Values below are for a large FOV covering the neck (166x240x256 mm³).

| | model 3 (`-m 0`) | model 4 (`-m 4`) |
|---|---|---|
| GPU | > 12 GB of GPU memory | to be measured |
| CPU, `-nbworker 8` | ~25 min, ~20 GB of RAM | to be measured |
| CPU, `-nbworker 1` | ~2h20, ~17 GB of RAM | to be measured |

Use `-nbworker 1` (or 2) if you run into memory issues: each worker holds a whole case in RAM.
