#!/bin/bash
# Round 4: the symmetry ablation (phase symmetry only, shift symmetry only)
# and two newer learned baselines (cResFreq, SwinFreq), 60k steps each.
# Training starts when round 3's GPU work is finished; the evaluations run
# after round 3's own evaluations.  Run from the repository root:
#   PY=.venv/bin/python caffeinate -i bash experiments/5_symmetry/run_round4.sh
set -u
PY=${PY:-python}
D=experiments/5_symmetry
mkdir -p $D/logs
train() { local log=$1; shift; [ -f $D/models/$log.pt ] || $PY $D/sym_train.py "$@" > $D/logs/$log.log 2>&1; }
evaluate() { [ -f $D/results/det_$2.json ] || $PY $D/sym_eval_detector.py $D/models/$2.pt $1 > $D/logs/eval_$2.log 2>&1; }
evalfr() { [ -f $D/results/fr_$1.json ] || $PY $D/sym_eval_fr.py "${@:2}" > $D/logs/evalfr_$1.log 2>&1; }

until grep -q "data-efficiency runs done" $D/logs/round3.log 2>/dev/null; do sleep 600; done
echo "round 3 GPU work finished, starting round 4"

train cpxdet16x64c_60k_s0 cpxdet16x64c --steps 60000 --tag _60k --seed 0 &
train rawdet16x64c_60k_s0 rawdet16x64c --steps 60000 --tag _60k --seed 0 &
wait
train cresfreq10x64_60k_s0 cresfreq10x64 --steps 60000 --tag _60k --seed 0 &
train swinfreq_60k_s0 swinfreq --steps 60000 --tag _60k --seed 0 &
wait
echo "round4 training done"

until grep -q "round3 done" $D/logs/round3.log 2>/dev/null; do sleep 600; done
evaluate cpxdet16x64c cpxdet16x64c_60k_s0
evaluate rawdet16x64c rawdet16x64c_60k_s0
evalfr cresfreq10x64_60k_s0 $D/models/cresfreq10x64_60k_s0.pt cresfreq10x64
evalfr swinfreq_60k_s0 $D/models/swinfreq_60k_s0.pt swinfreq
# earlier FR models with the same count threshold, clean-up and likelihood check
evalfr deepfreq2019 deepfreq2019
evalfr deepfreq20x64_60k deepfreq20x64
evalfr resfreq10x64_60k resfreq10x64
echo round4 done
