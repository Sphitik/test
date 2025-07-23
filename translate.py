import os
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator

# 🧠 Set Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✍️ Font for English text
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

def detect_speech_bubbles(image):
    """Detects comic-style speech bubbles (white regions with black outlines)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Find contours of white regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue  # ignore small specks

        x, y, w, h = cv2.boundingRect(cnt)
        bubbles.append((x, y, w, h))

    return bubbles

def translate_bubble_text(image, bubble, translator, target_lang='en'):
    x, y, w, h = bubble
    cropped = image[y:y+h, x:x+w]

    # Convert to PIL for Tesseract
    pil_crop = Image.fromarray(cropped)

    # Use Tesseract for vertical/block Chinese
    text = pytesseract.image_to_string(
        pil_crop,
        lang='chi_sim_vert',
        config='--psm 6'
    ).strip()

    if not text:
        return "", ""

    # Translate using Google Translate
    try:
        translated = translator.translate(text, src='zh-cn', dest=target_lang).text
    except Exception as e:
        print("Translation error:", e)
        translated = "[Translation failed]"

    return text, translated

def overlay_translated_text(image, bubble_box, translated_text):
    draw = ImageDraw.Draw(image)

    # Optional: Remove background fill or reduce opacity
    x, y, w, h = bubble_box
    draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255, 0))  # Transparent background

    # Set up font and wrapping
    font = ImageFont.truetype("arial.ttf", size=24)
    max_width = w - 10
    wrapped = textwrap.fill(translated_text, width=int(max_width / 12))  # Estimate width

    draw.text((x + 5, y + 5), wrapped, font=font, fill=(0, 0, 0))



def detect_text_bubbles(image):
    """Detects text bubbles or blocks in the image using contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to connect text regions vertically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 40:  # tweak for your case
            text_bubbles.append((x, y, w, h))

    # Optional: Sort top to bottom, left to right
    text_bubbles.sort(key=lambda b: (b[1], b[0]))

    return text_bubbles


def translate_comic_image(image_path, output_path, target_lang="en"):
    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    bubbles = detect_text_bubbles(image)  # assumes this returns list of bounding boxes

    translator = Translator()

    for bubble in bubbles:
        x, y, w, h = bubble
        cropped = image[y:y+h, x:x+w]

        # OCR: Read text from bubble
        text = pytesseract.image_to_string(cropped, lang='chi_sim_vert')  # Make sure 'chi_sim' is installed
        print(f"[OCR TEXT]: {text.strip()}")

        if not text.strip():
            print("Skipped: Empty or undetectable text.")
            continue

        # Translate text
        try:
            translated = translator.translate(text, dest=target_lang).text
            print(f"[TRANSLATED]: {translated.strip()}")
        except Exception as e:
            print(f"[TRANSLATION FAILED]: {e}")
            continue

        # Overlay translated text
        if translated.strip():
            overlay_translated_text(pil_image, bubble, translated)
        else:
            print("Skipped: Translation returned empty text.")

    pil_image.save(output_path)
    print(f"[SAVED]: Translated image saved to {output_path}")


def translate_folder(folder_path, target_lang="en"):
    out_dir = folder_path + " (translated)"
    os.makedirs(out_dir, exist_ok=True)

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            in_path = os.path.join(folder_path, filename)
            out_path = os.path.join(out_dir, filename)
            translate_comic_image(in_path, out_path, target_lang)

# ✅ RUN THIS
translate_folder(r"C:\Users\h_daf\Downloads\Documents\test", target_lang="en")
