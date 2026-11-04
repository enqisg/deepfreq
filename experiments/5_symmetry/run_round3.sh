#!/bin/bash
# Round 3: 60k-step detectors with a count head (3 seeds) and the matched
# DeepFreq-style ablation, data-efficiency runs, then test-set and robustness
# evaluation.  Run from the repository root:
#   PY=.venv/bin/python caffeinate -i bash experiments/5_symmetry/run_round3.sh
set -u
PY=${PY:-python}
D=experiments/5_symmetry
mkdir -p $D/logs
train() { local log=$1; shift; [ -f $D/models/$log.pt ] || $PY $D/sym_train.py "$@" > $D/logs/$log.log 2>&1; }
evaluate() { [ -f $D/results/det_$2.json ] || $PY $D/sym_eval_detector.py $D/models/$2.pt $1 > $D/logs/eval_$2.log 2>&1; }

# A/B: long runs, two at a time
train eqdet16x64c_60k_s0 eqdet16x64c --steps 60000 --tag _60k --seed 0 &
train eqdet16x64c_60k_s1 eqdet16x64c --steps 60000 --tag _60k --seed 1 &
wait
train eqdet16x64c_60k_s2 eqdet16x64c --steps 60000 --tag _60k --seed 2 &
train lindet16x64c_60k_s0 lindet16x64c --steps 60000 --tag _60k --seed 0 &
wait
echo "long runs done"

# C: data efficiency, 10k steps, 3k-100k training signals
for n in 3000 10000 30000 100000; do
  train eqdet16x64c_n${n}_s0 eqdet16x64c --steps 10000 --ntrain $n --tag _n$n &
  train lindet16x64c_n${n}_s0 lindet16x64c --steps 10000 --ntrain $n --tag _n$n &
  wait
done
echo "data-efficiency runs done"

# evaluation (CPU)
for s in 0 1 2; do evaluate eqdet16x64c eqdet16x64c_60k_s$s; done
evaluate lindet16x64c lindet16x64c_60k_s0
rm -f $D/results/det_eqdet16x64_20k_s0.json; evaluate eqdet16x64 eqdet16x64_20k_s0
for n in 3000 10000 30000 100000; do
  evaluate eqdet16x64c eqdet16x64c_n${n}_s0
  evaluate lindet16x64c lindet16x64c_n${n}_s0
done
$PY $D/sym_eval_hard.py $D/models/eqdet16x64c_60k_s0.pt eqdet16x64c > $D/logs/hard_eqdet16x64c_60k_s0.log 2>&1
# NOMP and the round-1/2 detectors again with the final clean-up settings (100-step refinement,
# likelihood check); their earlier det_*.json files were made without the check
$PY $D/sym_eval_classical.py > $D/logs/eval_classical.log 2>&1
for pair in eqdet8x64:eqdet8x64_s0 lindet8x64:lindet8x64_s0 eqdet8x64:eqdet8x64_20k_s0 lindet8x64:lindet8x64_20k_s0; do
  m=${pair%%:*}; c=${pair##*:}
  $PY $D/sym_eval_detector.py $D/models/$c.pt $m > $D/logs/eval_$c.log 2>&1
done
$PY $D/sym_eval_hard.py $D/models/eqdet16x64_20k_s0.pt eqdet16x64 > $D/logs/hard_eqdet16x64_20k_s0.log 2>&1
echo round3 done
