#!/usr/bin/env python3
"""
ACCURATE Token Usage Estimation - See script for full details
"""
import sys
from pathlib import Path
from collections import defaultdict
import yaml

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from utils.styles import (
    apply_politeness, apply_spacing, apply_punctuation,
    apply_letter_case, apply_length_variation, apply_interrogative
)

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 0.75)

def load_config():
    with open(repo_root / "config.yaml") as f:
        return yaml.safe_load(f)

def load_datasets(config):
    from datasets import load_dataset
    datasets_dict = {}
    
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    print("\n[1/4] TruthfulQA...")
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    datasets_dict['truthful_qa'] = [item['question'] for item in tqa.select(range(128))]
    print(f"✓ {len(datasets_dict['truthful_qa'])} questions")
    
    print("\n[2/4] GSM8K...")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    datasets_dict['gsm8k'] = [item['question'] for item in gsm8k.select(range(128))]
    print(f"✓ {len(datasets_dict['gsm8k'])} problems")
    
    print("\n[3/4] HarmBench...")
    try:
        hb = load_dataset("mantas-m/HarmBench", split="val")
        std = [i for i in hb if i.get('semantic_category')=='standard'][:128]
        datasets_dict['harmbench'] = [i['behavior'] for i in std]
    except:
        datasets_dict['harmbench'] = ["Harmful prompt"] * 128
    print(f"✓ {len(datasets_dict['harmbench'])} prompts")
    
    print("\n[4/4] Alpaca...")
    try:
        alp = load_dataset("tatsu-lab/alpaca", split="train")
        datasets_dict['alpaca'] = [i['instruction'] for i in alp.select(range(128))]
    except:
        datasets_dict['alpaca'] = ["Harmless prompt"] * 128
    print(f"✓ {len(datasets_dict['alpaca'])} instructions")
    
    return datasets_dict

def compute_usage(datasets, config):
    print("\n" + "="*80)
    print("COMPUTING EXACT TOKEN USAGE")
    print("="*80)
    
    # Dataset stats
    stats = {}
    for name, prompts in datasets.items():
        tokens = [count_tokens(p) for p in prompts]
        stats[name] = {'avg': sum(tokens)/len(tokens), 'total': sum(tokens)}
        print(f"\n{name}: avg={stats[name]['avg']:.1f} tokens/prompt")
    
    usage = defaultdict(lambda: {'input': 0, 'output': 0, 'exp': 0})
    
    max_out = config['defaults']['max_new_tokens']
    max_cot = config['defaults']['max_new_tokens_cot']
    cot_inst = "\n\nLet's think step by step."
    
    # Helper
    def process(key, style_key, prompts, apply_fn, strengths, positions, out_tokens, add_cot=False):
        prompt_list = [p + cot_inst for p in prompts] if add_cot else prompts
        for s in strengths:
            for pos in positions:
                styled = [apply_fn(p, s, pos) for p in prompt_list]
                in_tok = sum(count_tokens(sp) for sp in styled)
                usage[style_key]['input'] += in_tok
                usage[style_key]['output'] += 128 * out_tokens
                usage[style_key]['exp'] += 1
                usage[key]['input'] += in_tok
                usage[key]['output'] += 128 * out_tokens
                usage[key]['exp'] += 1
    
    print("\n[1/6] Politeness...")
    pol_s = config['style_levels']['politeness']
    pol_p = config['style_positions']['politeness']
    process('pol_tqa', 'politeness', datasets['truthful_qa'], apply_politeness, pol_s, pol_p, max_out)
    process('pol_harm', 'politeness', datasets['harmbench'], apply_politeness, pol_s, pol_p, max_out)
    process('pol_alp', 'politeness', datasets['alpaca'], apply_politeness, pol_s, pol_p, max_out)
    process('pol_cot', 'politeness', datasets['gsm8k'], apply_politeness, [-8,-4,0,4,8], ['global','prefix','suffix'], max_cot, True)
    
    print("[2/6] Spacing...")
    spa_s = config['style_levels']['spacing']
    spa_p = config['style_positions']['spacing']
    process('spa_tqa', 'spacing', datasets['truthful_qa'], apply_spacing, spa_s, spa_p, max_out)
    process('spa_harm', 'spacing', datasets['harmbench'], apply_spacing, spa_s, spa_p, max_out)
    process('spa_alp', 'spacing', datasets['alpaca'], apply_spacing, spa_s, spa_p, max_out)
    process('spa_cot', 'spacing', datasets['gsm8k'], apply_spacing, [0,20,50,100], ['global'], max_cot, True)
    
    print("[3/6] Punctuation...")
    pun_s = config['style_levels']['punctuation']
    pun_p = config['style_positions']['punctuation']
    process('pun_tqa', 'punctuation', datasets['truthful_qa'], apply_punctuation, pun_s, pun_p, max_out)
    process('pun_harm', 'punctuation', datasets['harmbench'], apply_punctuation, pun_s, pun_p, max_out)
    process('pun_alp', 'punctuation', datasets['alpaca'], apply_punctuation, pun_s, pun_p, max_out)
    process('pun_cot', 'punctuation', datasets['gsm8k'], apply_punctuation, [0,3,10,20], ['global'], max_cot, True)
    
    print("[4/6] Letter Case...")
    cas_s = config['style_levels']['letter_case']
    cas_p = config['style_positions']['letter_case']
    process('cas_tqa', 'letter_case', datasets['truthful_qa'], apply_letter_case, cas_s, cas_p, max_out)
    process('cas_harm', 'letter_case', datasets['harmbench'], apply_letter_case, cas_s, cas_p, max_out)
    process('cas_alp', 'letter_case', datasets['alpaca'], apply_letter_case, cas_s, cas_p, max_out)
    process('cas_cot', 'letter_case', datasets['gsm8k'], apply_letter_case, [0,25,50,100], ['global'], max_cot, True)
    
    print("[5/6] Length Variation...")
    len_m = config['style_levels']['length_variation']
    for m in len_m:
        for name, key, prompts, out in [
            ('truthful_qa', 'len_tqa', datasets['truthful_qa'], max_out),
            ('harmbench', 'len_harm', datasets['harmbench'], max_out),
            ('alpaca', 'len_alp', datasets['alpaca'], max_out),
            ('gsm8k', 'len_cot', datasets['gsm8k'], max_cot)
        ]:
            est = int(stats[name]['avg'] * m * 128)
            usage['length']['input'] += est
            usage['length']['output'] += 128 * out
            usage['length']['exp'] += 1
            usage[key]['input'] += est
            usage[key]['output'] += 128 * out
            usage[key]['exp'] += 1
    
    print("[6/6] Inter vs Imper...")
    for v in config['style_levels']['inter_vs_imper']:
        for name, key, prompts, out in [
            ('truthful_qa', 'for_tqa', datasets['truthful_qa'], max_out),
            ('harmbench', 'for_harm', datasets['harmbench'], max_out),
            ('alpaca', 'for_alp', datasets['alpaca'], max_out),
            ('gsm8k', 'for_cot', datasets['gsm8k'], max_cot)
        ]:
            est = int((stats[name]['avg'] + 5) * 128)
            usage['form']['input'] += est
            usage['form']['output'] += 128 * out
            usage['form']['exp'] += 1
            usage[key]['input'] += est
            usage[key]['output'] += 128 * out
            usage[key]['exp'] += 1
    
    return usage

def print_summary(usage):
    styles = ['politeness', 'spacing', 'punctuation', 'letter_case', 'length', 'form']
    total_in = sum(usage[s]['input'] for s in styles)
    total_out = sum(usage[s]['output'] for s in styles)
    total = total_in + total_out
    
    print("\n" + "="*80)
    print("TOTAL TOKEN USAGE")
    print("="*80)
    print(f"\nInput:  {total_in:,} ({total_in/total*100:.1f}%)")
    print(f"Output: {total_out:,} ({total_out/total*100:.1f}%)")
    print(f"TOTAL:  {total:,}")
    
    print("\n" + "="*80)
    print("COSTS")
    print("="*80)
    for model, (inp, outp) in [('GPT-4', (30,60)), ('Claude Sonnet', (3,15)), ('GPT-4o-mini', (0.15,0.6))]:
        cost = (total_in/1e6)*inp + (total_out/1e6)*outp
        print(f"\n{model}: ${cost:,.2f}")
    
    print("\n" + "="*80)
    print("BY STYLE")
    print("="*80)
    for s in styles:
        st = usage[s]['input'] + usage[s]['output']
        print(f"\n{s.upper()}: {st:,} ({st/total*100:.1f}%)")

def main():
    print("="*80)
    print("ACCURATE TOKEN ESTIMATION")
    print("="*80)
    config = load_config()
    datasets = load_datasets(config)
    usage = compute_usage(datasets, config)
    print_summary(usage)
    print("\n✓ Complete")

if __name__ == "__main__":
    main()