"""
Evaluation metric utilities.

Functions:
    - compute_bleu: BLEU score for word-level similarity
    - compute_bertscore: BERTScore for semantic similarity
    - compute_confidence: Confidence metrics (Delta LogProb, Entropy Shift, JSD)
"""

import torch
import torch.nn.functional as F
from bert_score import score as bert_score
from sacrebleu import corpus_bleu


def compute_bleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate text.
    
    Args:
        reference (str): Reference (gold) text
        candidate (str): Generated text to evaluate
    
    Returns:
        float: BLEU score (0-100)
    """
    # SacreBLEU expects list of candidates and list of references
    bleu = corpus_bleu([candidate], [[reference]])
    return bleu.score


def compute_bertscore(reference, candidate, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute BERTScore (F1) between reference and candidate.
    
    Args:
        reference (str): Reference (gold) text
        candidate (str): Generated text to evaluate
        device (str): Device for computation
    
    Returns:
        float: BERTScore F1 (0-1)
    """
    # BERTScore handles its own model loading internally
    P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False, device=device)
    return F1.item()


def compute_confidence(model, tokenizer, prompt_orig, prompt_pert, response_orig):
    """
    Compute confidence metrics comparing original and perturbed prompts.
    
    This function measures how the model's internal confidence changes when
    we perturb the prompt, even if the output text looks similar.
    
    Metrics:
        - Delta LogProb: How much log probability of the original response decreased
        - Entropy Shift: Change in prediction uncertainty (higher = more confused)
        - JSD Drift: Jensen-Shannon Divergence between probability distributions
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_orig (str): Original clean prompt
        prompt_pert (str): Perturbed prompt (with spacing/punctuation/etc)
        response_orig (str): Response generated from original prompt
    
    Returns:
        dict: Contains 'delta_log_prob', 'entropy_shift', 'jsd_drift'
    """
    device = model.device
    
    def get_response_logits(prompt, response):
        """
        Helper function to get logits for response tokens.
        Uses teacher forcing: feeds prompt+response, extracts logits for response part.
        """
        # Encode separately to find split point
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        resp_ids = tokenizer.encode(response, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids + resp_ids]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            all_logits = outputs.logits[0]  # [seq_len, vocab]
        
        # Extract logits that predict the response tokens
        # The model predicts token i+1 from position i
        start_idx = len(prompt_ids) - 1
        end_idx = len(prompt_ids) + len(resp_ids) - 1
        target_logits = all_logits[start_idx:end_idx]
        targets = torch.tensor(resp_ids).to(device)
        
        return target_logits, targets
    
    # Get logits for both prompts
    logits_orig, targets_orig = get_response_logits(prompt_orig, response_orig)
    logits_pert, targets_pert = get_response_logits(prompt_pert, response_orig)
    
    # Align lengths (truncate to shorter in case tokenization differs)
    min_len = min(logits_orig.size(0), logits_pert.size(0))
    logits_orig = logits_orig[:min_len]
    logits_pert = logits_pert[:min_len]
    targets = targets_orig[:min_len]
    
    # Compute probabilities
    log_probs_orig = F.log_softmax(logits_orig, dim=-1)
    log_probs_pert = F.log_softmax(logits_pert, dim=-1)
    probs_orig = F.softmax(logits_orig, dim=-1)
    probs_pert = F.softmax(logits_pert, dim=-1)
    
    # --- Metric 1: Delta Log Probability ---
    # Measures how much the model's confidence in the original response decreased
    target_log_prob_orig = log_probs_orig.gather(1, targets.unsqueeze(1)).squeeze()
    target_log_prob_pert = log_probs_pert.gather(1, targets.unsqueeze(1)).squeeze()
    seq_log_prob_orig = target_log_prob_orig.sum().item()
    seq_log_prob_pert = target_log_prob_pert.sum().item()
    delta_log_prob = seq_log_prob_orig - seq_log_prob_pert
    
    # --- Metric 2: Entropy Shift ---
    # Measures change in prediction uncertainty
    # Higher entropy = flatter distribution = more confused
    entropy_orig = -(probs_orig * log_probs_orig).sum(dim=-1).mean().item()
    entropy_pert = -(probs_pert * log_probs_pert).sum(dim=-1).mean().item()
    entropy_shift = entropy_pert - entropy_orig
    
    # --- Metric 3: Jensen-Shannon Divergence ---
    # Measures overall distribution drift between orig and pert
    # JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    m = 0.5 * (probs_orig + probs_pert)
    log_m = torch.log(m + 1e-10)  # epsilon for numerical stability
    kl_p_m = (probs_orig * (log_probs_orig - log_m)).sum(dim=-1).mean()
    kl_q_m = (probs_pert * (log_probs_pert - log_m)).sum(dim=-1).mean()
    jsd = 0.5 * (kl_p_m + kl_q_m).item()
    
    return {
        "delta_log_prob": delta_log_prob,
        "entropy_shift": entropy_shift,
        "jsd_drift": jsd
    }