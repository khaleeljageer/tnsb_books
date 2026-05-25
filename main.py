import os
import re
from pathlib import Path

import cv2
from pdf2image import convert_from_path
from google.cloud import vision
from tqdm import tqdm
from google.oauth2 import service_account
from google.cloud import vision


# =========================
# CONFIG
# =========================

INPUT_DIR = "input/Class_03/Tamil/Term_1"
OUTPUT_DIR = "output/Class_03/Term_1"

DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

credentials = service_account.Credentials.from_service_account_file(
    "tamil_ocr_credentials.json"
)

client = vision.ImageAnnotatorClient(credentials=credentials)


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess_image(image_path):

    img = cv2.imread(str(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


# =========================
# OCR USING GOOGLE VISION
# =========================

def google_ocr(image):

    _, encoded = cv2.imencode(".png", image)

    content = encoded.tobytes()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    return response.full_text_annotation.text


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove page numbers
        if re.fullmatch(r"\d+", line):
            continue

        # Remove excessive symbols
        line = re.sub(r"[ ]{2,}", " ", line)

        cleaned.append(line)

    return "\n".join(cleaned)


# =========================
# PDF PROCESSING
# =========================

def process_pdf(pdf_path):

    print(f"\nProcessing: {pdf_path}")

    pages = convert_from_path(
        pdf_path,
        dpi=DPI
    )

    all_text = []

    for idx, page in enumerate(tqdm(pages)):

        temp_img = f"temp_{idx}.png"

        page.save(temp_img)

        processed = preprocess_image(temp_img)

        text = google_ocr(processed)

        text = clean_text(text)

        all_text.append(text)

        os.remove(temp_img)

    return "\n".join(all_text)


# =========================
# MAIN
# =========================

pdf_files = list(Path(INPUT_DIR).rglob("*.pdf"))

for pdf_file in pdf_files:

    text = process_pdf(str(pdf_file))

    out_file = Path(OUTPUT_DIR) / (pdf_file.stem + ".txt")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved: {out_file}")