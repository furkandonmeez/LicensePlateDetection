import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image_folder_path = 'BIM472_Project1_Images'  


image_paths = []

for i in range(1, 51):  
    image_path = os.path.join(image_folder_path, f"{i}.JPG")
    if os.path.exists(image_path): 
        image_paths.append(image_path)

def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 100, 200)

    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return dilated

def find_candidates(img):

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h

        if 800 < area < 15000 and 1.5 < aspect_ratio < 5:
            rect_area = w * h
            fill_factor = area / rect_area

            if fill_factor > 0.3:
                candidates.append((x, y, w, h))
    return candidates

def draw_candidates(img, candidates):
    for (x, y, w, h) in candidates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def extract_text_from_candidates(img, candidates):
    plate_texts = []
    for (x, y, w, h) in candidates:
        roi = img[y:y + h, x:x + w]

        text = pytesseract.image_to_string(roi, config='--psm 8')  # config parametresi, plaka gibi kısa metinleri tanımak için kullanılır
        plate_texts.append(text.strip())
    return plate_texts

total_marked_candidates = 0
all_plate_texts = []


for img_path in image_paths:

    img = cv2.imread(img_path)
    if img is not None:
        preprocessed = preprocess_image(img)
        candidates = find_candidates(preprocessed)
        draw_candidates(img, candidates)
        plate_texts = extract_text_from_candidates(img, candidates)
        all_plate_texts.extend(plate_texts)
        total_marked_candidates += len(candidates)
        print(f"Number of candidates in {img_path}: {len(candidates)}")
        for text in plate_texts:
            print(f"Detected text: {text}")
    else:
        print(f"Image not found or cannot be loaded: {img_path}")

# Toplam işaretlenen yerleri konsola yazdır
print(f"Total number of marked candidates: {total_marked_candidates}")
print("All detected plate texts:", all_plate_texts)