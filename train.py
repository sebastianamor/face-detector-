import os
import sys
import pickle

import numpy as np
import face_recognition
from PIL import Image, ImageOps

DATASET_PATH = "dataset"
OUTPUT_PATH = "known_faces.pkl"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_SIDE = 1000  # redimensiona imágenes grandes para ir más rápido


def load_image(path):
    """Carga la imagen corrigiendo la rotación EXIF (fotos de celular) y la reduce si es muy grande."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(img)


def main():
    print("Iniciando entrenamiento...")

    if not os.path.isdir(DATASET_PATH):
        print(f"Error: no existe la carpeta '{DATASET_PATH}'.")
        print("Crea una carpeta por persona, por ejemplo: dataset/Juan_Perez/foto1.jpg")
        sys.exit(1)

    known_encodings = []
    known_names = []

    for name in sorted(os.listdir(DATASET_PATH)):
        person_dir = os.path.join(DATASET_PATH, name)
        if not os.path.isdir(person_dir):
            continue

        display_name = name.replace("_", " ")
        count = 0

        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            image_path = os.path.join(person_dir, filename)
            try:
                image = load_image(image_path)
            except Exception as e:
                print(f"Advertencia: no se pudo abrir {image_path} ({e})")
                continue

            locations = face_recognition.face_locations(image)

            if len(locations) == 0:
                print(f"Advertencia: no se encontró rostro en {image_path}")
                continue
            if len(locations) > 1:
                print(f"Advertencia: {len(locations)} rostros en {image_path}, se omite (usa fotos con una sola persona)")
                continue

            encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
            known_encodings.append(encoding)
            known_names.append(display_name)
            count += 1
            print(f"Procesado: {display_name} - {filename}")

        if count == 0:
            print(f"Advertencia: ninguna foto válida para '{display_name}'")

    if not known_encodings:
        print("Error: no se generó ningún encoding. No se guardó el archivo.")
        sys.exit(1)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    personas = len(set(known_names))
    print(f"¡Entrenamiento completado! {len(known_encodings)} fotos de {personas} persona(s) guardadas en '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()