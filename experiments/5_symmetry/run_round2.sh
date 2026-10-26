#!/bin/bash
# Round 2: longer detector runs (20k steps), two at a time on the GPU, then
# test-set evaluation with the NLS clean-up.  Run from the repository root:
#   caffeinate -i bash experiments/5_symmetry/run_round2.sh
set -u
PY=${PY:-python}
D=experiments/5_symmetry
mkdir -p $D/logs
run() { $PY $D/sym_train.py "$@" > $D/logs/$1${TAGLOG}.log 2>&1; }
TAGLOG=_20k_s0 run eqdet8x64 --steps 20000 --tag _20k &
TAGLOG=_20k_s0 run lindet8x64 --steps 20000 --tag _20k &
wait
TAGLOG=_20k_s0 run eqdet16x64 --steps 20000 --tag _20k &
TAGLOG=_20k_s1 run eqdet8x64 --steps 20000 --tag _20k --seed 1 &
wait
for pair in eqdet8x64:eqdet8x64_20k_s0 lindet8x64:lindet8x64_20k_s0 eqdet16x64:eqdet16x64_20k_s0 \
            eqdet8x64:eqdet8x64_20k_s1 eqdet8x64:eqdet8x64_s0 lindet8x64:lindet8x64_s0 \
            eqdet8x64fixed:eqdet8x64fixed_s0; do
  m=${pair%%:*}; c=${pair##*:}
  [ -f $D/models/$c.pt ] && $PY $D/sym_eval_detector.py $D/models/$c.pt $m > $D/logs/eval_$c.log 2>&1
done
echo round2 done
