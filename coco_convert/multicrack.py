import os
import json
import numpy as np # type: ignore
import cv2 # type: ignore

# Dosya yolları
input_dir = 'train'  # Orjinal resimlerin bulunduğu klasör
output_dir = f'output_{input_dir}'  # Çıktı klasörü
images_dir = os.path.join(output_dir, 'image')  # Çıktı resimler klasörü
masks_dir = os.path.join(output_dir, 'mask')  # Çıktı maskeler klasörü

# Klasörleri oluştur
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# JSON dosyasını oku
with open(f'{input_dir}/_annotations.coco.json') as file:
    data = json.load(file)

# Görüntü bilgilerini bir sözlükte sakla
images_info = {item['id']: item for item in data['images']}
annotations_per_image = {}

# Her annotasyon için maske oluştur
for annotation in data['annotations']:
    image_id = annotation['image_id']
    if image_id not in annotations_per_image:
        annotations_per_image[image_id] = []
    annotations_per_image[image_id].append(annotation)

# Her resim için tüm segmentasyonları birleştir
for image_id, annotations in annotations_per_image.items():
    image_info = images_info[image_id]
    image_path = os.path.join(input_dir, image_info['file_name'])
    mask_path = os.path.join(masks_dir, os.path.splitext(image_info['file_name'])[0] + '.png')

    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Maske için boş bir görüntü oluştur
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Tüm segmentasyonları maske üzerinde çiz
    for annotation in annotations:
        for seg in annotation['segmentation']:
            poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)

    # Görüntü ve maskeyi yeniden boyutlandır
    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Maskeyi ve görüntüyü kaydet
    cv2.imwrite(mask_path, resized_mask)
    cv2.imwrite(os.path.join(images_dir, os.path.splitext(image_info['file_name'])[0] + '.png'), resized_image)

print("Resimler ve maskeler 512x512 boyutunda oluşturuldu ve .png formatında kaydedildi.")
