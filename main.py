from reconocer_rostros_cam import *
from reconocer_rostros_imagen import *
from codificar_rostros import *
import os

while True:	
	print("""\n1. Entrenar modelo
2. Reconocer una imagen
3. Reconocer imágenes
4. Reconocer video
5. Salir\n""")

	op = input("Elija una opción: ")

	if op not in ['1', '2', '3', '4', '5']:
		print("\nOpción incorrecta")
		continue

	# OPCION 1
	if op == "1":
		print("\nENTRENAR MODELO")
		print("(Ingresar -1 en cualquier entrada para regresar)")
		print("""\n---El directorio que funcionará como dataset debe tener la siguiente estructura:---
\n---Debe contener carpetas con los nombres de las personas, separados por un subguión.---
---Dentro de la carpeta deben haber imágenes en las que aparezca solo dicha persona.---
---(No hay problema si aparece más de una vez en la imagen)---

	Estructura de ejemplo:\n
	Juan_Perez
	|___1.jpg
	|___a.jpg
	Maria_Ramirez
	|___1.jpg
	|___2.jpg""")

		while True:
			rutaDataset = input("\nRuta del dataset: ")

			if os.path.isdir(rutaDataset) or rutaDataset == "-1":
				break
			else:
				print("\nDirectorio erróneo")

		if rutaDataset == "-1":
			continue

		while True:
			print("\nModelo de detección (Recomendado Hog)\n1. cnn (Mayor precisión)\n2. hog (Mayor rapidez)")
			opm = input("\nElija una opción del modelo (1 o 2): ")

			if opm in ['1', '2', '-1']:
				break
			else:
				print("\nOpción incorrecta")

		if opm == "-1":
			continue

		if opm == "1":
			modeloDeteccion = "cnn"
			altura = 250
		else:
			modeloDeteccion = "hog"
			altura = 2500

		if opm == "-1":
			continue

		entrenar(rutaDataset, 'codificados'+modeloDeteccion+'.pickle', modeloDeteccion, altura)

		print("\nModelo entrenado")

	# OPCION 2
	if op == "2":
		print("\nRECONOCER IMAGEN")
		print("(Ingresar -1 en cualquier entrada para regresar)")

		while True:
			rutaImagen = input("\nRuta de la imagen: ")

			if os.path.exists(rutaImagen) or rutaImagen == "-1":
				break
			else:
				print("\nRuta errónea")

		if rutaImagen == "-1":
			continue

		while True:
			print("\nModelo de detección\n1. cnn (Mayor precisión)\n2. hog (Mayor rapidez)")
			opm = input("\nElija una opción del modelo (1 o 2): ")

			if opm == "1":
				modeloDeteccion = "cnn"
				altura = 350
			else:
				modeloDeteccion = "hog"
				altura = 2500

			if not os.path.exists('codificados'+modeloDeteccion+'.pickle'):
				print("\nNo se ha entrenado con ese modelo")
				continue

			if opm in ['1', '2', '-1']:
				break
			else:
				print("\nOpción incorrecta")

		if opm == "-1":
			continue

		reconocerRostroImg('codificados'+modeloDeteccion+'.pickle', modeloDeteccion, rutaImagen, altura)

		print("\nRostros identificados")

	# OPCION 3
	if op == "3":
		print("\nRECONOCER IMÁGENES")
		print("(Ingresar -1 en cualquier entrada para regresar)")

		while True:
			rutaDirectorio = input("\nRuta del directorio: ")

			if os.path.isdir(rutaDirectorio) or rutaDirectorio == "-1":
				break
			else:
				print("\nDirectorio erróneo")

		if rutaDirectorio == "-1":
			continue

		while True:
			print("\nModelo de detección\n1. cnn (Mayor precisión)\n2. hog (Mayor rapidez)")
			opm = input("\nElija una opción del modelo (1 o 2): ")

			if opm == "1":
				modeloDeteccion = "cnn"
				altura = 350
			else:
				modeloDeteccion = "hog"
				altura = 2500

			if not os.path.exists('codificados'+modeloDeteccion+'.pickle'):
				print("\nNo se ha entrenado con ese modelo")
				continue

			if opm in ['1', '2', '-1']:
				break
			else:
				print("\nOpción incorrecta")

		if opm == "-1":
			continue

		rutaImagenes = list(paths.list_images(rutaDirectorio))

		for (i, rutaImagen) in enumerate(rutaImagenes):
			reconocerRostroImg('codificados'+modeloDeteccion+'.pickle', modeloDeteccion, rutaImagen, altura)
			
			op = input("\nIngrese q para detener la detección o cualquiera para continuar: ")
			
			if op == "q":
				break
			else:
				continue

		print("\nRostros identificados")

	# OPCION 4
	if op == "4":
		print("\nRECONOCER VIDEO")
		print("(Ingresar -1 en cualquier entrada para regresar)")

		while True:
			rutaVideo = input("\nRuta del video (Ingresar 0 para webcam): ")

			if os.path.exists(rutaVideo) or rutaVideo in ['0', '-1']:
				break
			else:
				print("\nRuta errónea")

		if rutaVideo == "-1":
			continue

		if rutaVideo == "0":
			rutaVideo = int(rutaVideo)

		while True:
			print("\nModelo de detección\n1. cnn (Mayor precisión)\n2. hog (Mayor rapidez)")
			opm = input("\nElija una opción del modelo (1 o 2): ")

			if opm == "1":
				modeloDeteccion = "cnn"
				factor = 0.25
			else:
				modeloDeteccion = "hog"
				factor = 0.30

			if not os.path.exists('codificados'+modeloDeteccion+'.pickle'):
				print("\nNo se ha entrenado con ese modelo")
				continue

			if opm in ['1', '2', '-1']:
				break
			else:
				print("\nOpción incorrecta")

		if opm == "-1":
			continue

		reconocerRostroVid('codificados'+modeloDeteccion+'.pickle', modeloDeteccion, rutaVideo, factor)

		print("\nRostros identificados")

	# OPCION 5
	if op == "5":
		print("\nFinalizado")
		break
