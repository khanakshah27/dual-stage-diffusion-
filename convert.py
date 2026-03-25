import os
import csv

CAPTION_FILE = "results_20130124.token"  
IMAGE_FOLDER = "flickr30k-images"        
OUTPUT_CSV = "captions.csv"

MIN_WORDS = 5
MAX_WORDS = 50

rows = []
missing_images = 0
filtered_captions = 0

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")

        if len(parts) != 2:
            continue

        image_part, caption = parts

        image_name = image_part.split("#")[0]

        caption = caption.lower().strip()

        words = caption.split()

        if len(words) < MIN_WORDS:
            filtered_captions += 1
            continue

        if len(words) > MAX_WORDS:
            caption = " ".join(words[:MAX_WORDS]).rstrip(",:;-")


        image_path = os.path.join(IMAGE_FOLDER, image_name)
        if not os.path.exists(image_path):
            missing_images += 1
            continue

        rows.append([image_name, caption])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "caption"])
    writer.writerows(rows)

print("===================================")
print(f"Total valid pairs: {len(rows)}")
print(f"Filtered captions: {filtered_captions}")
print(f"Missing images: {missing_images}")
print("Saved to:", OUTPUT_CSV)
print("===================================")
