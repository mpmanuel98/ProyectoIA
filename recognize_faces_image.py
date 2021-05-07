import face_recognition
import pickle
import cv2
import dlib
from PIL import Image
import io
import numpy as np
from collections import Counter

# Permite la computacion usando CUDA para acelerar el procesamiento
dlib.DLIB_USE_CUDA = True
dlib.USE_AVX_INSTRUCTIONS = True

# Directorio para el fichero serializado de encodings de caras conocidas
encodings_path = "encodings.pickle"

# Carga los encodings con sus correspondientes nombres
print("Cargando encodings de caras conocidas...")
data = pickle.loads(open(encodings_path, "rb").read())
# Obtiene el numero de encodings que tiene cada cara conocida
name_counter = Counter(data['names'])

cap = cv2.VideoCapture()
cap.open("rtsp://admin:AmgCam18*@192.168.1.51:554/Streaming/Channels/1")
_, frame = cap.read()
cv2.imwrite("imagen_original.jpeg",frame)

# Abrimos la imagen
image = face_recognition.load_image_file("imagen_original.jpeg")

# Obtiene las coordenadas que delimitan cada cara en el frame capturado
# Obtiene los encodings para cada cara delimitada
print("Obteniendo coordenadas de la cara...")
boxes = face_recognition.face_locations(image)
print("Obteniendo encondigns de la cara...")
encodings = face_recognition.face_encodings(image, boxes)
print("Terminado!")
names = []
for encoding in encodings:
    # Relaciona el encoding de cada cara conocida con el encoding generado para cada cada capturada
    matches = face_recognition.compare_faces(data["encodings"], encoding)

    name = "Desconocido"

    # Comprobamos si existe alguna coincidencia
    if True in matches:
        # Guarda el indice de cada una de las caras conocidas que hayan tenido una coincidencia
        matchedIds = []
        for (index, boolean_value) in enumerate(matches):
            if(boolean_value):
                matchedIds.append(index)

        counts = {}

        # Itera por los indices con coincidencia y lleva la cuenta con cada cara reconocida
        for i in matchedIds:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # Itera por los nombres identificados y calcula la confianza para cada cara reconocida
        for name in counts:
            counts[name] = counts.get(name, 0) / name_counter[name]

        # Se queda con el nombre con mayor confianza
        name = max(counts, key=counts.get)

    # Actualiza la lista de nombres
    names.append(name)

index = 0
for face in encodings:
    cv2.rectangle(image, ((boxes[index])[3], (boxes[index])[0]), ((boxes[index])[1], (boxes[index])[2]), (0, 255, 0), 5)
    cv2.putText(image, names[index], ((boxes[index])[3], (boxes[index])[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 255, 0), 6)
    index += 1

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("imagen_procesada.jpeg", image_rgb)

print("En la imagen capturada aparecen las siguientes personas: ")
print(names)

print("La imagen se ha guardado en disco con el nombre 'imagen_procesada.jpeg'")
