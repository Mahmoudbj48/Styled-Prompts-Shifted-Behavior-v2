#!/bin/bash
MODEL="G-7B"
DATASET="gsm8k"
SAMPLES=128
BATCH=4

tmux new-session -d -s cot_spacing_G7B -x 220 -y 50
tmux send-keys -t cot_spacing_G7B "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style spacing --batch_size $BATCH --strengths 0 20 50 100 --places global" Enter

tmux new-session -d -s cot_punct_G7B -x 220 -y 50
tmux send-keys -t cot_punct_G7B "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style punctuation --batch_size $BATCH --strengths 0 3 10 20 --places global" Enter

tmux new-session -d -s cot_case_G7B -x 220 -y 50
tmux send-keys -t cot_case_G7B "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style letter_case --batch_size $BATCH --strengths 0 25 50 100 --places global" Enter

tmux new-session -d -s cot_polit_G7B -x 220 -y 50
tmux send-keys -t cot_polit_G7B "source ~/.venv/bin/activate && cd ~/Styled-Prompts-Shifted-Behavior-v2 && CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python experiments/cot_reasoning_generate.py --models $MODEL --dataset $DATASET --sample_size $SAMPLES --style politeness --batch_size $BATCH --strengths -8 -4 0 4 8 --places global" Enter

echo "All 4 G-7B experiments started on lambda1!"
