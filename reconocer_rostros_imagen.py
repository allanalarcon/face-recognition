import face_recognition
import pickle
import cv2

def reconocerRostroImg(archivoCodificado, modeloDeteccion, rutaImagen, altura):

    print("\nCargando rostros codificados...")
    data = pickle.loads(open(archivoCodificado, "rb").read())

    # Se obtiene la imagen
    image = cv2.imread(rutaImagen)

    # Cambiando tamaño de imagen para poder visualizar
    imageResize = cv2.resize(image, (int(image.shape[1]*700/image.shape[0]), 700), interpolation = cv2.INTER_AREA)

    # Cambiando tamaño de imagen para facilitar procesamiento
    if altura > image.shape[0]:
        # Cambiando BGR a RGB debido a que dlib trabaja con RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r = imageResize.shape[1] / float(image.shape[1])
    else:
        imageResizeMore = cv2.resize(image, (int(image.shape[1]*altura/image.shape[0]), altura), interpolation = cv2.INTER_AREA)
        # Cambiando BGR a RGB debido a que dlib trabaja con RGB
        imageRGB = cv2.cvtColor(imageResizeMore, cv2.COLOR_BGR2RGB)
        r = imageResize.shape[1] / float(imageResizeMore.shape[1])

    print("Reconociendo rostros de la imagen...")
    # Obtener coordenadas de cada una de los rostros de la imagen
    roi = face_recognition.face_locations(imageRGB, model=modeloDeteccion)

    # Codificar los rostros
    codificados = face_recognition.face_encodings(imageRGB, roi)

    # Inicializa variable donde se almacenarán los nombres de las personas reconocidas
    nombres = []
    veces = {}

    # Se recorren los rostros codificados para reconocerlos
    for codificado in codificados:
        # Arreglo booleano para indicar coincidencia con los codificados en el entrenamiento
        coincidencias = face_recognition.compare_faces(data["codificados"], codificado)
        nombre = "Desconocido"

        # Valida existencia de coincidencia
        if True in coincidencias:
            # Indices de las coincidencias
            coincidenciasID = [i for (i, b) in enumerate(coincidencias) if b]

            # Inicializa diccionario donde almacena los nombres y la cantidad de coindicencias
            contador = {}

            # Recorre los indices
            for i in coincidenciasID:
                nombre = data["nombres"][i]
                contador[nombre] = contador.get(nombre, 0) + 1

            # Elige el nombre que tenga el mayor número de coincidencias
            nombre = max(contador, key=contador.get)

        # Agrega el nombre del rostro al arreglo de nombres
        nombres.append(nombre)
        veces[nombre] = veces.get(nombre, 0) + 1

    # Recorre las coordenadas de los rostros para dibujar un roi y colocar el nombre
    for ((top, right, bottom, left), nombre) in zip(roi, nombres):
        # Se vuelve al tamaño del primer resize
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # Dibuja un rectángulo en el rostro
        cv2.rectangle(imageResize, (left, top), (right, bottom), (142, 85, 20), 1)

        # Coloca el nombre de la persona
        cv2.rectangle(imageResize, (left, bottom), (right, bottom + 30), (142, 85, 20), cv2.FILLED)
        cv2.putText(imageResize, nombre, (left + 3, bottom + 11), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
        cv2.putText(imageResize, 'Veces: ' + str(veces[nombre]), (left + 3, bottom + 26), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
    
    cv2.imshow("Imagen", imageResize)
    cv2.waitKey(0)

#reconocerRostro('codificados.pickle', 'cnn', "2.jpg")