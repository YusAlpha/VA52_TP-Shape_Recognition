import cv2
import numpy as np
import os

# Dossier contenant les images transformées
transformed_images_folder = "transformed_images"
current_directory = os.getcwd()
folder_path = os.path.join(current_directory, transformed_images_folder)

# Dossier où vous souhaitez enregistrer les images extraites
output_folder = "extracted_characters"
os.makedirs(output_folder, exist_ok=True)

# Emplacement du fichier pour enregistrer les coordonnées
output_txt_file = "coordinates.txt"

# Ouvrir le fichier de sortie en mode écriture
with open(output_txt_file, "w") as coordinates_file:
    # Liste des noms de fichiers d'images transformées
    image_files = [filename for filename in os.listdir(os.path.join(current_directory, transformed_images_folder)) if filename.endswith(".jpg") and filename != ".DS_Store" and filename != "Thumbs.db"]

    # Boucle sur toutes les images
    for image_file in image_files:
        # Charger l'image binaire résultante après le traitement
        
        image_path = os.path.join(transformed_images_folder, image_file)
        print(f"Chargement de l'image depuis : {image_path}")

        # Charger l'image en niveau de gris
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        result = {}
        # Parcourir chaque colonne de l'image et compter les pixels blancs
        for col in range(70):
            # Extraire la colonne courante
            col_data = image[:, col]
            
            # Compter le nombre de pixels blancs (supérieurs à un certain seuil)
            white_pixels = np.sum(col_data > 200)  # Vous pouvez ajuster le seuil (200) selon vos besoins
            
            # Stocker le nombre de pixels blancs dans le tableau de résultats
            result[col] = white_pixels

        # Identifier les passages 0 → x et x → 0 pour la projection verticale
        start_positions = []
        end_positions = []
        in_character = False

        for i, value in enumerate(result.values()):
            if value > 0 and not in_character:
                start_positions.append(i)
                in_character = True
            elif value == 0 and in_character:
                end_positions.append(i)
                in_character = False

        if in_character:
            # Si la dernière colonne est blanche, marquez-la comme fin
            end_positions.append(len(result) - 1)

        print(f"Nombre de caractères détectés : {len(start_positions)}")
        
        # 2. Extraire et enregistrer les images des caractères
        for character_index, (start_x, end_x) in enumerate(zip(start_positions, end_positions)):
            # Extraire le caractère individuel
            character_image = image[:, start_x:end_x + 1]
            
            # Calculer l'histogramme horizontal pour le caractère
            horizontal_projection = np.sum(character_image, axis=1)
            
            # Identifier les passages 0 → x et x → 0 pour la projection horizontale
            top_position = 0
            bottom_position = len(horizontal_projection) - 1
            
            while top_position < len(horizontal_projection) and horizontal_projection[top_position] == 0:
                top_position += 1
            
            while bottom_position >= 0 and horizontal_projection[bottom_position] == 0:
                bottom_position -= 1

            # Enregistrer l'image extraite
            character_filename = f"{image_file.replace('.jpg', '')}_character_{character_index}.jpg"
            output_path = os.path.join(output_folder, character_filename)
            cv2.imwrite(output_path, character_image[top_position:bottom_position + 1, :])

            # Enregistrer les coordonnées dans le fichier de sortie
            coordinates_file.write(f"{image_file}, character_{character_index}: Top Left ({start_x}, {top_position}), Bottom Right ({end_x}, {bottom_position})\n")

print("Extraction terminée. Les images de caractères sont enregistrées dans le dossier 'extracted_characters'.")
print("Les coordonnées des boîtes englobantes sont enregistrées dans le fichier 'coordinates.txt'.")
