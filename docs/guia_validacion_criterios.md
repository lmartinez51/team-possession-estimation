# Guía de Validación y Criterios de Evaluación — Reto "More than a Hack 2027"

Este documento resume de manera sencilla cómo los jueces evaluarán nuestras predicciones de posesión, qué validaciones debemos revisar al correr nuestro script local y cuáles son los archivos entregables que debemos preparar.

---

## 1. ¿Cómo evaluarán los jueces nuestros resultados? (El "Ground Truth")
Los organizadores (FC Barcelona y Metrica Sports) tienen una **etiqueta real oculta** para cada fotograma del partido, creada por analistas humanos que vieron el video cuadro por cuadro. Nuestro script entrega la columna de posesión vacía ya resuelta y la plataforma de los jueces la compara contra su respuesta secreta evaluando tres pilares:

1. **Precisión Cuadro por Cuadro (Accuracy / Macro-F1):**
   Mide qué tan bien atinamos si el balón lo tenía el Local (`H`) o el Visitante (`A`) en cada milisegundo. Aquí es donde ganamos ventaja gracias a nuestros **puentes de pase** y el **decodificador de Viterbi**, ya que evitamos que la posesión parpadee falsamente cuando la pelota viaja por el aire cerca de un defensa rival.
2. **Calibración de Probabilidades (Log-Loss):**
   Evalúa qué tan seguros estamos de nuestras predicciones. En jugadas divididas o rebotes confusos, nuestro sistema utiliza **Jev AI (System One)** para asignar probabilidades equilibradas (ej. 55% / 45%) en lugar de adivinar a ciegas, lo que evita penalizaciones matemáticas severas.
3. **Coherencia Táctica y Tasa de Parpadeo (*Flicker Rate*):**
   En el fútbol real la posesión no cambia 40 veces por minuto. Los clubes profesionales revisan que la secuencia tenga sentido y que nuestro archivo no presente errores de formato que puedan romper su evaluador automático.

---

## 2. ¿Qué debemos revisar en la terminal al correr `auto_process_competition.py`?
Antes de entregar cualquier archivo, al ejecutar el script localmente debemos confirmar lo siguiente en la consola:

* **`0 NaNs, contract-verified`:** Es la señal vital más importante. Nos garantiza que el archivo de entrega no tiene celdas vacías ni errores de estructura que provoquen una descalificación automática por tecnicismos.
* **Porcentaje de posesión global (Home vs. Away Share):** Debe reflejar un partido de fútbol realista (ej. 55%-45%, 60%-40%). Si saliera un porcentaje absurdo como 99%-1%, sabríamos que hay un error en la entrada de datos.
* **Inspección rápida en el Visor 3D (`viewer.html`):** Abrir el visor 20 segundos y comprobar visualmente que, durante los pases largos, el color y la baliza se mantienen firmes en el equipo poseedor hasta que ocurre una recuperación real.

---

## 3. ¿Cuál es el documento y archivo entregable?
Dependiendo de lo que pida la plataforma de la competencia en esta primera fase, los entregables son:

1. **El archivo de predicciones de datos (`submission_*.csv` o Parquet):**
   * Es el entregable principal. Contiene la tabla fotograma por fotograma con la estimación exacta de la posesión y las probabilidades calculadas por nuestro pipeline para cada partido evaluado.
2. **El repositorio de código / script ejecutable:**
   * El orquestador unificado (`auto_process_competition.py`) junto con los módulos de modelos y decodificación, listo para correrse de manera autónoma en el entorno de evaluación de los jueces.
3. **Soporte visual para la defensa (Opcional pero potente):**
   * El visor 3D interactivo (`viewer.html` y `match_payload.js`), el cual podemos proyectar o mostrar para explicar de forma visual e intuitiva por qué nuestra inteligencia artificial interpreta los partidos mejor que los modelos tradicionales basados solo en cercanía.