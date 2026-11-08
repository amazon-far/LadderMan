#!/usr/bin/env bash
# Step 1: train one privileged teacher per expert (ladder inclination x rung spacing), sequentially.
#
#   bash scripts/train_teachers.sh                                  # 60.0 62.7 65.5 68.2 deg, default spacing
#   bash scripts/train_teachers.sh 60.0 65.5                        # a subset of inclinations
#   SPACINGS="0.22 0.25 0.28" bash scripts/train_teachers.sh        # inclinations x spacings
#   bash scripts/train_teachers.sh 65.5 -- wandb.mode=online        # extra Hydra overrides after "--"
#
# Teachers are trained in inclination-major order (outer loop over inclinations, inner over spacings),
# which is the order expected by `algo.teacher_checkpoint_path_list` of the student.
# To use several GPUs, run one teacher per GPU instead, e.g.
#   CUDA_VISIBLE_DEVICES=1 python scripts/train.py algo=teacher task=teacher \
#       task.command.ladder_config.inclination_deg=62.7 task.command.ladder_config.rung_spacing=0.25
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

INCLINATIONS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do INCLINATIONS+=("$1"); shift; done
[[ $# -gt 0 ]] && shift  # drop "--"
[[ ${#INCLINATIONS[@]} -eq 0 ]] && INCLINATIONS=(60.0 62.7 65.5 68.2)
read -r -a SPACING_LIST <<< "${SPACINGS:-}"

for incl in "${INCLINATIONS[@]}"; do
    if [[ ${#SPACING_LIST[@]} -eq 0 ]]; then
        echo "=== training teacher: inclination ${incl} deg, default rung spacing ==="
        python scripts/train.py algo=teacher task=teacher task.command.ladder_config.inclination_deg="${incl}" "$@"
    else
        for sp in "${SPACING_LIST[@]}"; do
            echo "=== training teacher: inclination ${incl} deg, rung spacing ${sp} m ==="
            python scripts/train.py algo=teacher task=teacher \
                task.command.ladder_config.inclination_deg="${incl}" task.command.ladder_config.rung_spacing="${sp}" "$@"
        done
    fi
done
