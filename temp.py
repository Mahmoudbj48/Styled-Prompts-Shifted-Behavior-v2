from utils.data import load_gsm8k


def main():
    # Load some problems
    problems = load_gsm8k(sample_size=5, seed=42)

    for i, p in enumerate(problems):
        print(f"\n=== Problem {i+1} ===")
        print(f"Question: {p['question']}")
        print(f"Answer: {p['best_answer']}")
        print(f"Solution preview: {p['meta']['solution'][:100]}...")


if __name__ == "__main__":
    main()