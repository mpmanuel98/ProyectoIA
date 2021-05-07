from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Directorio donde se almacenan las imagenes de cada persona
dataset_path = "dataset"

# Directorio para el fichero serializado de cada encoding
encodings_path = "encodings.pickle"

# Lista el directorio de las imagenes de cada persona
print("Listando directorio...")
image_Paths = list(paths.list_images(dataset_path))

known_Encodings = []
known_Names = []

for (i, image_Path) in enumerate(image_Paths):
    # Extrae el nombre de la persona del nombre del directorio
    print("Procesando la imagen {}/{}".format(i + 1, len(image_Paths)))
    name = image_Path.split(os.path.sep)[-2]

    # Abre la imagen
    image = face_recognition.load_image_file(image_Path)

    # Obtiene las coordenadas que delimitan cada cara en la imagen
    # Se especifica que se usa una CNN
    boxes = face_recognition.face_locations(image)

    print("Caras detectadas: " + str(len(boxes)))

    # Obtiene los encodings para cada cara delimitada
    encodings = face_recognition.face_encodings(image, boxes)

    for encoding in encodings:
        # Añade cada encoding generado junto con el nombre de la persona asociada
        known_Encodings.append(encoding)
        known_Names.append(name)

# Guarda en disco los encodings con sus correspondientes nombres
print("Serializando encodings...")
data = {"encodings": known_Encodings, "names": known_Names}
f = open(encodings_path, "wb")
f.write(pickle.dumps(data))
f.close()
