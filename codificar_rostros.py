import cv2
import numpy
import face_recognition
import pickle
import os
from imutils import paths

def entrenar(rutaDataset, archivoCodificado, modeloDeteccion):

    print("Obteniendo imágenes de dataset...")

    # Se obtienen las rutas de las imágenes del dataset
    rutaImagenes = list(paths.list_images(rutaDataset))

    # Inicializan las variables donde se almacenarán los rostros codificados y el nombre correspondiente
    rostrosCodificados = []
    nombres = []

    # Se recorren imagenes del dataset
    for (i, rutaImagen) in enumerate(rutaImagenes):
        # Se obtiene el nombre de la persona de la carpeta de las imágenes
        nombre = rutaImagen.split(os.path.sep)[-2].replace("_", " ")

        print("Procesando imagen de " + nombre + ". Ruta: " + rutaImagen + ". Imagen {}/{}".format(i + 1, len(rutaImagenes)))

        # Cambiando BGR a RGB debido a que dlib trabaja con RGB
        file = open(rutaImagen, 'rb')
        bytes = bytearray(file.read())
        arrayTemp = numpy.asarray(bytes, dtype=numpy.uint8)
        imageTemp = cv2.imdecode(arrayTemp, cv2.IMREAD_UNCHANGED)

        # Cambiando tamaño de imagen para facilitar procesamiento
        imageResize = cv2.resize(imageTemp, (int(imageTemp.shape[1]*350/imageTemp.shape[0]), 350), interpolation = cv2.INTER_AREA)

        imageRGB = cv2.cvtColor(imageResize, cv2.COLOR_BGR2RGB)

        # Obtener coordenadas de cada una de los rostros de la imagen
        roi = face_recognition.face_locations(imageRGB, model=modeloDeteccion)

        # Codificar los rostros
        codificados = face_recognition.face_encodings(imageRGB, roi)

        # Guardar los rostros codificados y los nombres relacionados por index
        for codificado in codificados:
            rostrosCodificados.append(codificado)
            nombres.append(nombre)

    # Almacenando rostros codificados en un archivo
    print("Almacenando rostros codificados...")
    data = {"codificados": rostrosCodificados, "nombres": nombres}
    f = open(archivoCodificado, "wb")
    f.write(pickle.dumps(data))
    f.close()

entrenar('dataset', 'codificados.pickle', 'cnn')