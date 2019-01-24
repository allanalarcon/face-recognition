import face_recognition
import pickle
import cv2

def reconocerRostro(archivoCodificado, modeloDeteccion, rutaImagen):

    print("Cargando rostros codificados...")
    data = pickle.loads(open(archivoCodificado, "rb").read())

    # Se obtiene la imagen
    image = cv2.imread(rutaImagen)

    # Cambiando tamaño de imagen para poder visualizar
    # ESTE TAMAÑO ES MUY GRANDE PARA LA LAPTOP, PERO CON ALTO DE 400 UNA PRUEBA SE DEMORÓ 70s
    imageResize = cv2.resize(image, (int(image.shape[1]*500/image.shape[0]), 500), interpolation = cv2.INTER_AREA)

    # Cambiando tamaño de imagen para facilitar procesamiento
    # CON ESTA SEGUNDA REDUCCIÓN CON LA MISMA IMAGEN DE LA PRUEBA ANTERIOR DEMORÓ 35s
    imageResizeMore = cv2.resize(imageResize, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

    # Cambiando BGR a RGB debido a que dlib trabaja con RGB
    imageRGB = cv2.cvtColor(imageResizeMore, cv2.COLOR_BGR2RGB)
    print(imageResize.shape)

    print("Reconociendo rostros de la imagen...")
    # Obtener coordenadas de cada una de los rostros de la imagen
    roi = face_recognition.face_locations(imageRGB, model=modeloDeteccion)

    # Codificar los rostros
    codificados = face_recognition.face_encodings(imageRGB, roi)

    # Inicializa variable donde se almacenarán los nombres de las personas reconocidas
    nombres = []

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

    # Recorre las coordenadas de los rostros para dibujar un roi y colocar el nombre
    for ((top, right, bottom, left), nombre) in zip(roi, nombres):
        # Se vuelve al tamaño del primer resize
        top = int(top * 2)
        right = int(right * 2)
        bottom = int(bottom * 2)
        left = int(left * 2)

        # Dibuja un rectángulo en el rostro
        cv2.rectangle(imageResize, (left, top), (right, bottom), (255, 0, 0), 2)

        # Coloca el nombre de la persona
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(imageResize, nombre, (left, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Imagen", imageResize)
    cv2.waitKey(0)

reconocerRostro('codificados.pickle', 'cnn', "2.jpg")