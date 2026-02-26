run_experiment() {
    local script=$1
    local model=$2
    echo "Running: $script --models $model"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python $script --models $model --sample_size 128 --batch_size 4
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()"
    echo "Done: $script --models $model"
    echo ""
}





run_experiment experiments/punctuation.py Q2.5-1.5B
run_experiment experiments/spacing.py L3.2-3B
run_experiment experiments/spacing.py Q2.5-1.5B

echo "ALL DONE"
