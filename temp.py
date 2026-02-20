from utils.styles import apply_spacing, apply_punctuation, apply_letter_case

def run():
    print("Running experiment...")
    text = "What is the capital of France?"

    # Global (default - original behavior)
    print(apply_spacing(text, 5))
    print(apply_spacing(text, 5, place="global"))
    print(apply_spacing(text, 5, place="prefix"))
    print(apply_spacing(text, 5, place="suffix"))

    # Punctuation examples
    print(apply_punctuation(text, 3, place="prefix"))
    print(apply_punctuation(text, 3, place="suffix"))
    print(apply_punctuation(text, 3, place="global"))

    # Letter case examples
    print(apply_letter_case(text, 30))
    print(apply_letter_case(text, 30, place="global"))
    print(apply_letter_case(text, 50, place="prefix"))
    print(apply_letter_case(text, 50, place="suffix"))
    print(apply_letter_case(text, 100))

if __name__ == "__main__":
    run()