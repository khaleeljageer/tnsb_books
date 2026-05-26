import os
import re
import csv
import unicodedata
from pathlib import Path
from tqdm import tqdm


# ==========================================
# CONFIG
# ==========================================

INPUT_DIR = "output"
OUTPUT_DIR = "word_corpus"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# REGEX
# ==========================================

# Tamil Unicode range
TAMIL_PATTERN = re.compile(r'^[\u0B80-\u0BFF]+$')

# Remove non-Tamil chars
NON_TAMIL_PATTERN = re.compile(r'[^\u0B80-\u0BFF\s]')


# ==========================================
# CLEAN WORD
# ==========================================

def extract_tamil_words(text):

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove English/numbers/symbols
    text = NON_TAMIL_PATTERN.sub(" ", text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    cleaned_words = []

    for word in words:

        word = word.strip()

        # Skip tiny words
        if len(word) < 2:
            continue

        # Ensure full Tamil word
        if not TAMIL_PATTERN.fullmatch(word):
            continue

        # Remove OCR garbage
        if len(set(word)) <= 1:
            continue

        cleaned_words.append(word)

    return cleaned_words


# ==========================================
# PROCESS CLASS
# ==========================================

def process_class_folder(class_dir):

    print(f"\nProcessing: {class_dir.name}")

    unique_words = set()

    txt_files = list(class_dir.rglob("*.txt"))

    for txt_file in tqdm(txt_files):

        try:

            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            words = extract_tamil_words(text)

            unique_words.update(words)

        except Exception as e:

            print(f"[ERROR] {txt_file} -> {e}")

    # Sort words
    sorted_words = sorted(unique_words)

    # Output folder
    output_path = Path(OUTPUT_DIR) / class_dir.name

    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "word_corpus.csv"

    # Save CSV
    with open(csv_file, "w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["word"])

        for word in sorted_words:
            writer.writerow([word])

    print(f"Saved: {csv_file}")
    print(f"Unique Words: {len(sorted_words)}")


# ==========================================
# MAIN
# ==========================================

class_folders = [
    d for d in Path(INPUT_DIR).iterdir()
    if d.is_dir()
]

print(f"Found {len(class_folders)} class folders to process.")

for class_dir in sorted(class_folders):
    print(f"Processing Class: {class_dir.name}\n")

    process_class_folder(class_dir)

print("\nWord corpus generation completed.")