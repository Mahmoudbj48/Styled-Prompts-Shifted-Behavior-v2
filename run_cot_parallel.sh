#!/bin/bash
MODEL="L3.1-8B"
DATASET="gsm8k"
SAMPLES=128
BATCH=4

tmux new-session -d -s cot_spacing -x 220 -y 50
tmux send-keys -t cot_spacing "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style spacing --batch_size $BATCH --strengths 0 20 50 100 --places global" Enter

tmux new-session -d -s cot_punct -x 220 -y 50
tmux send-keys -t cot_punct "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style punctuation --batch_size $BATCH --strengths 0 3 10 20 --places global" Enter

tmux new-session -d -s cot_case -x 220 -y 50
tmux send-keys -t cot_case "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style letter_case --batch_size $BATCH --strengths 0 25 50 100 --places global" Enter

tmux new-session -d -s cot_polit -x 220 -y 50
tmux send-keys -t cot_polit "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style politeness --batch_size $BATCH --strengths -8 -4 0 4 8 --places global" Enter

echo "All 4 experiments started!"
echo "Monitor with: tmux attach -t cot_spacing"
