#!/bin/bash
# Round 5: is data augmentation enough without built-in symmetry?  DeepFreq's
# front-end (lindet16x64c) trained with random frequency shifts on 3k, 10k and
# 100k signals, and ours with the same shifts on 3k (10k steps each, compare
# with round 3's data-efficiency runs).  Then a 60k-step detector trained with
# random noise colours, and a second seed of the 60k DeepFreq-front-end model.
# Training starts when round 4's GPU work is finished; the evaluations run
# after round 4's own evaluations.  Run from the repository root:
#   PY=.venv/bin/python caffeinate -i bash experiments/5_symmetry/run_round5.sh > experiments/5_symmetry/logs/round5.log 2>&1
set -u
PY=${PY:-python}
D=experiments/5_symmetry
mkdir -p $D/logs
train() { local log=$1; shift; [ -f $D/models/$log.pt ] || $PY $D/sym_train.py "$@" > $D/logs/$log.log 2>&1; }
evaluate() { [ -f $D/results/det_$2.json ] || $PY $D/sym_eval_detector.py $D/models/$2.pt $1 > $D/logs/eval_$2.log 2>&1; }
hard() { [ -f $D/results/hard_impulsive_$2.json ] || $PY $D/sym_eval_hard.py $D/models/$2.pt $1 > $D/logs/hard_$2.log 2>&1; }

until grep -q "round4 training done" $D/logs/round4.log 2>/dev/null; do sleep 600; done
echo "round 4 GPU work finished, starting round 5"

# A: shift augmentation, 10k steps
train lindet16x64c_n3000aug_s0 lindet16x64c --steps 10000 --ntrain 3000 --augment-shift --tag _n3000aug &
train lindet16x64c_n10000aug_s0 lindet16x64c --steps 10000 --ntrain 10000 --augment-shift --tag _n10000aug &
wait
train lindet16x64c_n100000aug_s0 lindet16x64c --steps 10000 --ntrain 100000 --augment-shift --tag _n100000aug &
train eqdet16x64c_n3000aug_s0 eqdet16x64c --steps 10000 --ntrain 3000 --augment-shift --tag _n3000aug &
wait
echo "augmentation runs done"

# B: 60k steps, colour-trained detector and a second seed of the DeepFreq-front-end model
train eqdet16x64c_colour60k_s0 eqdet16x64c --steps 60000 --augment-colour --tag _colour60k --seed 0 &
train lindet16x64c_60k_s1 lindet16x64c --steps 60000 --tag _60k --seed 1 &
wait
echo "round5 training done"

until grep -q "round4 done" $D/logs/round4.log 2>/dev/null; do sleep 600; done
for n in 3000 10000 100000; do evaluate lindet16x64c lindet16x64c_n${n}aug_s0; done
evaluate eqdet16x64c eqdet16x64c_n3000aug_s0
evaluate eqdet16x64c eqdet16x64c_colour60k_s0
evaluate lindet16x64c lindet16x64c_60k_s1
hard eqdet16x64c eqdet16x64c_colour60k_s0
echo round5 done
