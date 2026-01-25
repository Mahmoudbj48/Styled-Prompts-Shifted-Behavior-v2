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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


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
    
    # --- Metric 3: Jensen-Shannon Divergence (JSD) ---
    # JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    m = 0.5 * (probs_orig + probs_pert)
    log_m = torch.log(m + 1e-10)  # epsilon for stability

    # KL Divergence = sum(p * (log_p - log_m))
    kl_p_m = (probs_orig * (log_probs_orig - log_m)).sum(dim=-1).mean()
    kl_q_m = (probs_pert * (log_probs_pert - log_m)).sum(dim=-1).mean()

    jsd = 0.5 * (kl_p_m + kl_q_m).item()
    
    return {
        "delta_log_prob": delta_log_prob,
        "entropy_shift": entropy_shift,
        "jsd_drift": jsd
    }


"""
Activation Analysis Metrics
Based on "Silent Tokens, Loud Effects: Padding in LLMs" methodology
"""



def get_layer_activations(model, tokenizer, prompt, layer_idx=-1):
    """
    Extract activations from a specific layer of the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt (str): Input prompt
        layer_idx (int): Layer index to extract (-1 for last layer)
    
    Returns:
        torch.Tensor: Activations of shape [seq_len, hidden_dim]
    """
    # Tokenize input
    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    except:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    attention_mask = torch.ones_like(inputs)
    
    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    # Extract hidden states from specified layer
    # hidden_states is a tuple: (embedding_output, layer_1, layer_2, ..., layer_N)
    hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
    
    # Take last token position (where the model "understands" the full prompt)
    last_token_activation = hidden_states[0, -1, :]  # [hidden_dim]
    
    return last_token_activation


def compute_activation_similarity(model, tokenizer, prompt_orig, prompt_pert, layer_idx=-1):
    """
    Compute cosine similarity between activations of original and perturbed prompts.
    
    Measures how much the internal representation changes due to style perturbation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_orig (str): Original prompt
        prompt_pert (str): Perturbed prompt
        layer_idx (int): Layer index (-1 for last layer)
    
    Returns:
        float: Cosine similarity between activations (0-1, higher = more similar)
    """
    # Get activations for both prompts
    act_orig = get_layer_activations(model, tokenizer, prompt_orig, layer_idx)
    act_pert = get_layer_activations(model, tokenizer, prompt_pert, layer_idx)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(act_orig.unsqueeze(0), act_pert.unsqueeze(0)).item()
    
    return similarity


def compute_activation_similarity_all_layers(model, tokenizer, prompt_orig, prompt_pert):
    """
    Compute activation similarity across all layers.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_orig (str): Original prompt
        prompt_pert (str): Perturbed prompt
    
    Returns:
        dict: Contains 'mean_similarity', 'per_layer_similarity', 'last_layer_similarity'
    """
    # Tokenize inputs
    try:
        inputs_orig = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_orig}],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
        inputs_pert = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_pert}],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)
    except:
        inputs_orig = tokenizer(prompt_orig, return_tensors="pt").input_ids.to(model.device)
        inputs_pert = tokenizer(prompt_pert, return_tensors="pt").input_ids.to(model.device)
    
    attention_mask_orig = torch.ones_like(inputs_orig)
    attention_mask_pert = torch.ones_like(inputs_pert)
    
    # Forward passes
    with torch.no_grad():
        outputs_orig = model(
            input_ids=inputs_orig,
            attention_mask=attention_mask_orig,
            output_hidden_states=True
        )
        outputs_pert = model(
            input_ids=inputs_pert,
            attention_mask=attention_mask_pert,
            output_hidden_states=True
        )
    
    # Compute similarity for each layer
    per_layer_similarities = []
    num_layers = len(outputs_orig.hidden_states)
    
    for layer_idx in range(num_layers):
        act_orig = outputs_orig.hidden_states[layer_idx][0, -1, :]  # Last token
        act_pert = outputs_pert.hidden_states[layer_idx][0, -1, :]
        
        similarity = F.cosine_similarity(act_orig.unsqueeze(0), act_pert.unsqueeze(0)).item()
        per_layer_similarities.append(similarity)
    
    return {
        'mean_similarity': float(np.mean(per_layer_similarities)),
        'last_layer_similarity': per_layer_similarities[-1],
        'per_layer_similarity': per_layer_similarities
    }


def reduce_activations_2d(activations_list, method='pca', seed=42):
    """
    Reduce high-dimensional activations to 2D for visualization.
    
    Args:
        activations_list (list): List of torch tensors or numpy arrays [n_samples, hidden_dim]
        method (str): 'pca' or 'tsne'
        seed (int): Random seed
    
    Returns:
        numpy.ndarray: 2D coordinates [n_samples, 2]
    """
    # Stack all activations
    if isinstance(activations_list[0], torch.Tensor):
        activations = torch.stack(activations_list).cpu().numpy()
    else:
        activations = np.array(activations_list)
    
    # Reduce to 2D
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=seed)
    elif method.lower() == 'tsne':
        # Dynamic perplexity based on number of samples
        n_samples = activations.shape[0]
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    reduced = reducer.fit_transform(activations)
    return reduced


def collect_activations_for_prompts(model, tokenizer, prompts, layer_idx=-1):
    """
    Collect activations for a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts (list): List of prompt strings
        layer_idx (int): Layer index
    
    Returns:
        list: List of activation tensors
    """
    activations = []
    for prompt in prompts:
        act = get_layer_activations(model, tokenizer, prompt, layer_idx)
        activations.append(act)
    return activations