import cv2
import face_recognition
import pickle
import time

def reconocerRostroVid(archivoCodificado, modeloDeteccion, rutaVideo, factor):

    print("\nCargando rostros codificados...")
    data = pickle.loads(open(archivoCodificado, "rb").read())

    # Inicia transmisión en vivo de la ruta
    print("Iniciando transmisión...")
    video = cv2.VideoCapture(rutaVideo)
    time.sleep(2.0)

    # Loop para leer todos los frames
    while True:
        # Se obtiene el frame
        ret, frame = video.read()

        # Cambiando tamaño de imagen para facilitar procesamiento
        imageResize = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        r = frame.shape[1] / float(imageResize.shape[1])

        # Cambiando BGR a RGB debido a que dlib trabaja con RGB
        imageRGB = cv2.cvtColor(imageResize, cv2.COLOR_BGR2RGB)

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
            # Se vuelve al tamaño original del frame
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # Dibuja un rectángulo en el rostro
            cv2.rectangle(frame, (left, top), (right, bottom), (157, 93, 21), 1)

            # Coloca el nombre de la persona
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), (157, 93, 21), cv2.FILLED)
            cv2.putText(frame, nombre, (left + 3, bottom + 11), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, 'Veces: ' + str(veces[nombre]), (left + 3, bottom + 26), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)

        cv2.imshow("Video", frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

#reconocerRostro('codificados.pickle', 'cnn', 0)