# Estimación de Posesión de Equipo por Cuadro (Frame-Level Team Possession Estimation)

🌐 **Idioma / Language / Llengua:** [English](README.md) | [Español](README.es.md) | [Català](README.ca.md)

> **Arquitectura RFC v2.0:** GBDT Cinemático Consciente de Calidad + Anclas de Contacto Jev System One + Decodificador Secuencial Soft Viterbi  
> **Competición Objetivo:** *More than a Hack 2027* (FC Barcelona & Metrica Sports)  
> **Autor:** Lorenzo Martínez  

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pipeline Tests](https://img.shields.io/badge/pipeline%20tests-passing%20100%25-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-RFC%20v2.0-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 1. Resumen Ejecutivo y Tesis Central

En la analítica de fútbol de élite, **la posesión de equipo a nivel de cuadro es una interpretación temporal del control del balón y no la simple identidad estática del jugador más cercano**.

Durante un pase filtrado a alta velocidad o un cambio de juego del FC Barcelona, la proyección planar 2D del balón transita habitualmente a distancias sub-métricas de defensores rivales sin existir contacto físico alguno. Las heurísticas ingenuas de proximidad sufren un parpadeo de etiquetas catastrófico durante estos cruces, mientras que los filtros de suavizado temporal rígidos borran intercepciones legítimas de un solo toque y transiciones rápidas.

Este repositorio implementa una solución de calidad de producción y rigurosamente fundamentada en matemáticas, sustituyendo las heurísticas ingenuas por **evidencia probabilística ponderada por confianza** y **decodificación secuencial estructurada suave**:

```
                       +-----------------------------------+
                       | Optical Tracking Data (Ball/Play) |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 1: Preprocesador Segmentado  |
                       | - Interpolación acotada de huecos |
                       | - Proxy multi-escala no circular  |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 2: Cinemática y Alcance      |
                       | - Horizonte corto tau [0.1, 1.5s] |
                       | - Toque vs conducción desacoplada |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 3: Anclas y Puentes          |
                       | - Anclas de contacto probables    |
                       | - Puentes no causales acotados    |
                       | - Refinador disperso Jev (Sys 1)  |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 4: Clasificador y Decodifica |
                       | - GBDT simétrico exacto team-swap |
                       | - Relajación dinámica Soft Viterbi|
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 5: Evaluación y Ablación     |
                       | - Emparejamiento bipartito robos  |
                       | - Slices diagnósticos de fallo    |
                       | - Suite de ablación de 6 etapas   |
                       +-----------------------------------+
```

---

## 2. Arquitectura Modular en Seis Etapas

La base de código está estructurada en seis módulos de pipeline auto-verificables y con fuga de datos cero (zero-leakage):

### [Task 1: Validación de Esquema y Preprocesamiento](file:///task1_schema_and_preprocessing.py)
* **Parametrización Temporal Estricta en Segundos:** Convierte todas las ventanas y pasos de derivadas dinámicamente mediante marcas de tiempo y frecuencia de muestreo $f_s$, garantizando idéntico comportamiento físico tanto a $25\text{ Hz}$ como a $50\text{ Hz}$.
* **Interpolación Acotada de Oclusiones:** Brechas cortas de seguimiento ($\le 0.40\text{ s}$) se interpolan linealmente de forma estricta dentro de segmentos continuos registrando `ball_is_interpolated`. Brechas mayores ($> 0.40\text{ s}$), límites de período y cambios de cámara cortan los identificadores de segmento para evitar contaminación de derivadas.
* **Proxy de Cambio de Velocidad Multi-Escala No Circular $I_b(t)$:** Detecta previamente golpeos e impactos antes del suavizado mediante análisis de pendientes MCO (mínimos cuadrados ordinarios) hacia adelante y hacia atrás ($0.12\text{ s}$ y $0.24\text{ s}$), garantizando que los pases potentes nunca sean borrados por filtros de Savitzky-Golay.

### [Task 2: Características Cinemáticas Locales y Alcance Físico](file:///task2_feature_engine.py)
* **Búsqueda de Intercepción en Horizonte Corto:** Calcula horizontes de alcance físico $\Delta \tau \in [0.1, 1.5]\text{ s}$ bajo límites de velocidad de sprint ($9.5\text{ m/s}$), aceleración máxima ($7.0\text{ m/s}^2$) y retraso biológico de reacción ($\tau_{\text{react}} = 0.30\text{ s}$).
* **Evidencia de Control Desacoplada:** Separa la evidencia de impulsos instantáneos de toque de las conducciones sostenidas alineadas en velocidad.
* **Márgenes de Ambigüedad Espacial:** Evalúa márgenes de distancia entre el candidato top-1 vs. top-2 y la congestión de rivales dentro de radios de $2.0\text{ m}$ y $5.0\text{ m}$.

### [Task 3: Anclas Probabilísticas y Puentes Acotados](file:///task3_anchors_and_bridges.py)
* **Anclas Salientes de Contacto:** Identifica momentos de control de alta confianza y aísla rebotes contra el césped de contactos humanos reales.
* **Puentes No Causales Acotados:** Propaga probabilidades pasadas y futuras hacia adelante y hacia atrás a través de pases en vuelo ($\le 4.0\text{ s}$), deteniéndose inmediatamente ante huecos de seguimiento $> 0.50\text{ s}$ o cuando surge evidencia contradictoria de control.
* **Refinador Disperso TypeSafe AI (`Jev / System One`):** Dirige selectivamente únicamente los fotogramas de alta ambigüedad ($\sim 5\text{--}10\%$) mediante primitivas calibradas `Noul`, `Choice` y `Score` para desambiguar balones divididos y pases rasantes cercanos sin incurrir en el costo de procesar todo el partido con LLMs.

### [Task 4: GBDT Simétrico y Decodificador Soft Viterbi](file:///task4_classifier_and_decoder.py)
* **Búfer Analítico de Purga de Dependencias:** Deduce la separación temporal mínima exacta libre de fugas para validación cruzada intra-partido:
  $$\Delta t_{\text{purge}} = \frac{1}{2}W_{\text{smooth}} + W_{\text{deriv}} + \tau_{\text{reach}} + \frac{1}{2}W_{\text{rolling}} + \Delta t_{\text{bridge}} = 7.38\text{ segundos}$$
* **Invarianza Exacta por Intercambio de Equipo (Team-Swap):** Fuerza características pareadas y evaluación antisimétrica, garantizando simetría algebraica de probabilidad:
  $$|P(\text{Home} \mid \mathbf{x}_t) - (1 - P(\text{Home} \mid \text{Swap}(\mathbf{x}_t)))| \le 2.22 \times 10^{-16}$$
* **Decodificación de Cadena Discriminativa Suave:** Decodifica emisiones de fotogramas mediante una cadena de Viterbi modificada cuyo costo de transición se relaja dinámicamente solo cuando convergen simultáneamente contactos confiables y evidencia de control rival.

### [Task 5: Evaluación Diagnóstica y Estudio de Ablación](file:///task5_evaluation_and_ablation.py)
* **Priors de Geometría Táctica Resistentes a Outliers:** Combina el área del Convex Hull de jugadores de campo con la dispersión espacial intercuartil normalizada hacia la dirección canónica de ataque.
* **Emparejamiento Bipartito 1-a-1 de Cambios de Posesión (Turnovers):** Emparejamiento temporal greedy ($\pm 250\text{ ms}$) dentro de segmentos continuos para calcular F1 de robos, precisión, exhaustividad y error temporal con signo en milisegundos.
* **Reporte de Fallos por Slices Diagnósticos:** Evalúa la precisión en segmentos de alto riesgo (pases cercanos a rivales, huecos interpolados, balones divididos 50/50 y fronteras de segmento).

### [Task 6: Visualizador Táctico y Dashboard Diagnóstico](file:///task6_match_visualizer.py)
* **Dashboard Táctico Multi-Panel:** Renderiza un visualizador profesional de 3 paneles que combina vista 2D de cancha (con colores oficiales azulgrana del FC Barcelona, coordenadas métricas, dorsales y trayectoria del balón), curvas de probabilidad posterior con posesión dominante sombreada, y discontinuidades cinemáticas ($I_b$, probabilidad de pase filtrado, concordancia de puentes y activaciones de Jev).
* **Ingesta Defensiva de Esquemas:** Reconoce columnas de predicción sin fallar ante datasets de prueba ciegos (sin ground truth) o tablas de submission oficiales.
* **Ejecución Headless y Escalable:** Basado en `matplotlib.use("Agg")`, des-serialización segura de modelos y zoom opcional por ventana temporal (`time_window_s=(t_start, t_end)`).

---

## 3. Resultados del Estudio de Ablación Ordenado

Evaluado sobre partidos sintéticos de prueba con pases filtrados a alta velocidad, marcajes estrechos, oclusiones de $0.20\text{ s}$, pérdidas de balón de $0.80\text{ s}$ e intercepciones al primer toque:

| Etapa de Ablación | Precisión de Cuadro (%) | Macro-F1 | F1 Robos ($\pm 250\text{ ms}$) | Error Temporal Medio (ms) | Cambios Falsos / Min | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1a. Baseline Jugador Más Cercano** | 45.5% | 0.429 | 0.267 | 100.0 | 62.5 | 0.4575 |
| **1b. Filtro Simple de Histéresis** | 49.8% | 0.476 | 0.400 | 100.0 | 34.1 | 0.4575 |
| **2. Cinemática Local + GBDT** | 82.8% | 0.820 | 0.125 | 40.0 | 159.1 | 0.1494 |
| **3. + Ventanas de Contexto Temporal** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **4. + Anclas y Puentes Jev** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **5. + Decodificador Soft Viterbi** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | 0.0137 |
| **6. + Priors Aéreos y Tácticos (MVP)** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | **0.0053** |

* **Intervalo de Confianza Bootstrap 90% (Etapa 6):** $[99.2\%,\, 99.6\%]$

### Rendimiento en Casos Extremos Diagnósticos (Etapa 6 OOF)
* **Pase Cercano al Rival ($\text{Proximidad No Confiable} \ge 0.30$):** `100.0%`
* **Brechas Cortas Interpoladas ($0.20\text{ s}$):** `100.0%`
* **Transiciones en Balones Divididos ($\pm 120\text{ ms}$):** `87.5%` ($30.0\text{ ms}$ de error temporal)
* **Regiones de Frontera de Segmento y Período:** `100.0%`

---

## 4. Guía de Ejecución y CLI

### Configuración del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/lmartinez51/team-possession-estimation.git
cd team-possession-estimation

# Crear y activar entorno virtual Python
python -m venv .venv
# En Windows:
.\.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate

# Instalar dependencias requeridas
pip install numpy pandas scipy scikit-learn matplotlib
```

### Ejecución de Suites de Verificación

Cada tarea incorpora una suite de pruebas de verificación determinística:

```bash
# Tarea 1: Preprocesamiento y proxy de cambio de velocidad
python task1_schema_and_preprocessing.py

# Tarea 2: Alcance cinemático y características de toque/conducción
python task2_feature_engine.py

# Tarea 3: Anclas, puentes y refinamiento con Jev System One
python task3_anchors_and_bridges.py

# Tarea 4: Entrenamiento GBDT simétrico y decodificación Soft Viterbi
python task4_classifier_and_decoder.py

# Tarea 5: Métricas de diagnóstico, slices extremos y ablación completa
python task5_evaluation_and_ablation.py

# Tarea 6: Visualizador táctico y panel diagnóstico
python task6_match_visualizer.py
```

### CLI de Inferencia Unificada (`run_submission_pipeline.py`)

El script principal [`run_submission_pipeline.py`](file:///run_submission_pipeline.py) orquesta las Tareas 1 a 5 en un CLI completo de entrenamiento, calibración out-of-fold, caché de características y predicción.

#### Características Principales:
* **Ingesta Agnóstica de Formato (`MetricaFormatAdapter`):** Ingiere tracking en dos tablas (`frame_df` + `player_tracks_df`) o desancla tablas anchas de Metrica (`Home_11_x`, `Home_11_y`, etc.), normalizando encabezados y proveyendo máscaras por defecto.
* **Caché Determinística (`FeatureCacheManager`):** Almacena características en `.feature_cache/` identificadas por match ID, versión de esquema (`RFC_v2.0.0_MVP`) y hash SHA-256 de configuración.
* **Preservación Explícita de Estados Latentes:** Genera la submission binaria requerida (`Home` vs. `Away`) mientras preserva en columnas complementarias los estados internos (`Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`).
* **Integración con TypeSafe AI (`Jev / System One`):** Soporta resolución de ambigüedad mediante llamadas en vivo a la API (`--live-jev`) o fallback heurístico cinemático.

#### Ejemplos de Uso del CLI:

```powershell
# 1. Predicción sobre partido con refinamiento TypeSafe AI (Jev / System One):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_frames.csv --players ruta_players.csv --output artifacts\mi_submission.csv --live-jev

# 2. Puerta de verificación completa end-to-end (entrena, calibra, serializa bundle y prueba partido de validación):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode verify

# 3. Entrenar y exportar bundle de modelos:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode train --model-bundle artifacts\barca_possession_mvp_v2.pkl

# 4. Predicción sobre tabla ancha de tracking de Metrica:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_Sample_Game_1_RawTrackingData.csv --output artifacts\mi_submission.csv

# 5. Predicción sobre archivos de tracking divididos Home y Away de Metrica:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames data_metrica\Sample_Game_1_RawTrackingData_Home_Team.csv --players data_metrica\Sample_Game_1_RawTrackingData_Away_Team.csv --output artifacts\reintento_metrica.csv --live-jev --max-frames 2500
```

#### Referencia de Parámetros:
* `--mode`: Modo de ejecución (`verify`, `train`, `predict`). Por defecto: `verify`.
* `--frames`: Ruta al archivo `frame_df` CSV/Parquet, tabla ancha única o tabla Home de Metrica.
* `--players`: Ruta al archivo `player_tracks_df` CSV/Parquet o tabla Away de Metrica (auto-detectado).
* `--meta`: Ruta opcional a metadatos JSON (`match_id`, `sampling_rate_hz`, dimensiones del campo).
* `--model-bundle`: Ruta al bundle de modelos `.pkl` (por defecto: `artifacts/barca_possession_mvp_v2.pkl`).
* `--output`: Ruta de salida para el CSV de submission (por defecto: `artifacts/submission_output.csv`).
* `--live-jev`: Habilita consultas en vivo a TypeSafe AI (`Jev / System One`) mediante `TYPESAFE_API_KEY`.
* `--max-frames`: Límite opcional de cuadros para pruebas rápidas.

### Ingesta Batch Automatizada para Datos Reales (`auto_process_competition.py`)

Cuando FC Barcelona y Metrica Sports entreguen los datos finales de la competición, [`auto_process_competition.py`](file:///auto_process_competition.py) actúa como el **orquestador batch sin configuración**. Escanea recursivamente la carpeta destino, auto-detecta la estructura de los archivos, empareja los partidos, ejecuta la inferencia y genera los CSVs de submission y dashboards visuales.

#### 1. Formatos Soportados y Auto-Detección:
* **Tablas Divididas de Metrica Sports (`metrica_split`):**
  - Empareja automáticamente archivos `*Home_Team.csv` y `*Away_Team.csv` del mismo partido.
  - Parsea los encabezados de 3 filas, alinea marcas de tiempo, extrae coordenadas del balón y transforma las coordenadas normalizadas `[0, 1]` a metros de cancha centrados en `(0, 0)` (`[-52.5, 52.5]` m x `[-34.0, 34.0]` m).
* **Tablas Relacionales Canónicas (`canonical_pair`):**
  - Empareja tablas de metadatos de cuadros (`*frame*.csv/.parquet`) con trayectorias de jugadores en formato largo (`*player*.csv/.parquet`).
* **Tablas Anchas Únicas (`wide_single`):**
  - Ingiere tablas únicas con jugadores distribuidos en columnas (`Home_11_x, Home_11_y...`).
* **Filtrado Inteligente y Rechazo de Ruido:**
  - Descarta automáticamente archivos de eventos (`*Events*.csv`), planillas arbitrales y alineaciones.

#### 2. Disposición de Carpetas:
Coloca los archivos de tracking descargados directamente dentro del directorio `data/` (o en subcarpetas dentro de él):

```text
Possession-Estimation/
├── data/                                         # Colocar aquí los partidos descargados
│   ├── Match_1_RawTrackingData_Home_Team.csv     # Tracking Local (Metrica)
│   ├── Match_1_RawTrackingData_Away_Team.csv     # Tracking Visitante (Metrica)
│   ├── Match_1_RawEventsData.csv                 # (Ignorado automáticamente)
│   ├── Match_2_RawTrackingData_Home_Team.csv
│   └── Match_2_RawTrackingData_Away_Team.csv
├── artifacts/                                    # Carpeta destino de resultados
│   ├── submission_Match_1.csv                    # CSV validado y sin valores nulos
│   ├── submission_Match_2.csv
│   └── visualizations/                           # Diagnósticos tácticos 2D
│       ├── Match_1_diagnostic.png
│       └── Match_2_diagnostic.png
└── auto_process_competition.py
```

#### 3. Variables de Entorno:
Si utilizas el desempate en vivo con TypeSafe AI (`Jev / System One`), configura tu clave de API:

```powershell
# Windows PowerShell:
$env:TYPESAFE_API_KEY = "tu_clave_api_aqui"

# Linux / macOS Bash:
export TYPESAFE_API_KEY="tu_clave_api_aqui"
```
*(Nota: Si no se define una clave de API, el pipeline utiliza de forma transparente el fallback cinemático sin interrumpir la ejecución).*

#### 4. Comandos de Ejecución:

##### A. Ejecución Batch de Competición Completa (Producción):
Procesa cada partido detectado en `data/`, aplica los modelos calibrados y refinamiento con Jev, valida el contrato de submission y genera diagnósticos gráficos:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --visualize
```

##### B. Prueba Rápida de Humo / Sanity Check (~100 segundos de partido):
Antes de procesar partidos de 90 minutos (~145,000 cuadros cada uno), valida el flujo en los primeros 2,500 cuadros con `--max-frames`:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --max-frames 2500 --visualize
```

##### C. Modo Offline / Heurístico (Sin Clave de API):
Inferencia puramente determinística basada en cinemática local y relajación temporal con Soft Viterbi:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --visualize
```

#### 5. Verificación de Contrato y Salidas:
* **Ruta de Salida:** `artifacts/submission_<match_id>.csv`
* **Columnas Generadas:**
  - `frame`: Índice del cuadro (entero, estrictamente ascendente).
  - `possession_team`: Equipo en posesión (`Home` o `Away`).
  - `confidence`: Probabilidad posterior calibrada `[0.50, 1.00]`.
  - `possession_state`: Estado latente detallado (`Home`, `Away`, `Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`).
* **Dashboard Visual:** Panel táctico 2D con vectores de velocidad, trayectoria del balón y probabilidad acumulada guardado en `artifacts/visualizations/<match_id>_diagnostic.png`.

#### 6. Tabla de Parámetros del Batch CLI:
| Parámetro | Por Defecto | Descripción |
|---|---|---|
| `--data-dir` | `data` | Directorio donde se ubican los archivos de tracking (búsqueda recursiva). |
| `--output-dir` | `artifacts` | Directorio de salida para submissions y diagnósticos gráficos. |
| `--live-jev` | `False` | Habilita consultas LLM a TypeSafe AI (`Jev / System One`) en jugadas divididas. |
| `--max-frames` | `None` | Límite de cuadros procesados por partido (ej. `2500` para pruebas de humo). |
| `--visualize` | `False` | Genera y guarda dashboards gráficos de 3 paneles en alta resolución. |
| `--model-bundle` | `artifacts/barca_possession_mvp_v2.pkl` | Ruta al bundle del modelo LightGBM entrenado. |

### Repetición 3D Interactiva para Retransmisión (`generate_3d_replay_html.py`)

Para demostraciones técnicas, experiencia del aficionado y análisis táctico, [`generate_3d_replay_html.py`](file:///generate_3d_replay_html.py) compila las predicciones en una **experiencia 3D desacoplada en WebGL** construida con Three.js. Traduce conceptos matemáticos abstractos (puentes acotados, estados de Viterbi y contactos resueltos por Jev) a elementos visuales intuitivos.

```powershell
# 1. Exportar datos del partido y visor 3D desacoplado (por defecto: 60s a 25Hz):
.\.venv\Scripts\python.exe generate_3d_replay_html.py --max-frames 1500

# 2. Abrir el visor directamente en el navegador (compatible con Zero-CORS file://):
Start-Process artifacts\visualizations\viewer.html

# 3. Forzar el restablecimiento o sobrescritura de la plantilla visual:
.\.venv\Scripts\python.exe generate_3d_replay_html.py --reset-viewer

# 4. Compilar un archivo HTML autónomo e independiente para compartir:
.\.venv\Scripts\python.exe generate_3d_replay_html.py --standalone
```

#### Arquitectura, Características Visuales y Motor Zero-GC:
* **Frontend Desacoplado y Zero-CORS:** El código visual reside en `artifacts/visualizations/viewer.html`, estrictamente separado de `match_payload.js` (`window.MATCH_DATA = {...};`) y `match_payload.json`. Evita las restricciones locales de CORS de los navegadores al abrir directamente con doble clic vía `file://`.
* **Guardia Anti-Sobrescritura:** Ejecutar Python actualiza siempre los datos del partido preservando cualquier edición visual o de Three.js manual en `viewer.html` salvo que se especifique `--reset-viewer`.
* **Recarga en Caliente en la App:** Pulsa **🔄 Recargar Datos** en la barra inferior para refrescar `match_payload.js` en vivo sin perder la vista de cámara ni la velocidad de reproducción.
* **Avatares de Jugadores Articulados en 3D Estilo Minecraft / Roblox:** Modelos 3D totalmente articulados con cabezas cúbicas, equipaciones procedurales a rayas (rayas verticales blaugranas clásicas con ribetes dorados para el FC Barcelona vs. pizarra y cian eléctrico para el rival, y kit dorado para porteros), animación de extremidades en carrera (balanceo de brazos y piernas proporcional a la velocidad) y rotación suave de orientación (yaw).
* **Balón de Fútbol 3D con Relieve y Rodadura Física:** Textura procedural de paneles hexagonales/pentagonales en blanco y negro, jaula externa de relieve 3D y rotación de rodadura físicamente coherente con la velocidad sobre el césped respecto al radio del balón.
* **Faro 3D de Poseedor Activo Estilo EA Sports FC:** Baliza dinámica de alta visibilidad anclada al jugador en posesión, que incluye un anillo pulsante sobre el césped, un cilindro de aura de foco cenital (`ConeGeometry`) y un diamante flotante. Los aros en los pies de los jugadores sin balón se atenúan a una opacidad sutil de `0.22`, eliminando la saturación visual en la cancha. Se adapta dinámicamente tanto en **🧠 Nuestra IA** (dueño probabilístico) como en **🤖 IA Común** (jugador más cercano).
* **Vínculo Volumétrico de Control (Cadena de Cuentas de Energía):** Sustituye las líneas alámbricas 2D de 1px por una cadena de 16 esferas volumétricas 3D a lo largo de un arco parabólico animado que conecta el pecho del poseedor legítimo con la trayectoria del balón.
* **Estela de Cometa 3D Inmune al Desplazamiento (Scrub-Safe):** 18 esferas graduadas que trazan la estela del balón mediante muestreo temporal sub-frame fraccional, manteniéndose completamente estable durante el rebobinado rápido y escalando su longitud de acuerdo con la velocidad del esférico.
* **Halo de Posesión Radiante en el Césped:** Anillo luminoso directamente debajo de la sombra del balón que pulsa con el color del equipo en posesión (degradado blaugrana para el FC Barcelona vs. cian para el rival).
* **Pulso Neuronal Violeta de Jev AI (System One):** Cuando Jev resolvió un toque dividido al límite (`f0.jev === 1`), un giroscopio 3D violeta de doble anillo contra-rotatorio (`#A855F7`) se activa con una suave persistencia visual de 0.45s para que los juicios de un solo cuadro sean perfectamente apreciables por el espectador.
* **Tarjeta HUD de Jugador Activo con Comprobación de Suciedad (Dirty-Checking) a 60 FPS:** Gráfico televisivo anclado bajo el marcador superior que muestra el escudo del club, dorsal grande y legible (`#10`) y etiqueta de estado de control en tiempo real (*CONTROL TOTAL*, *DISPUTADA*, etc.). Impulsado por almacenamiento en caché sin búsquedas (`domCache`) y dirty-checking (`lastHudState`), eliminando recálculos de diseño en cada cuadro.
* **Gestor Plegable y Filtrable de Momentos Clave:** Barra de herramientas y cajón interactivo que ofrece:
  - **Vista Plegable en 3 Modos:** `COMPACT` (carrusel horizontal predeterminado de 1 fila con desplazamiento con rueda del ratón), `EXPANDED` (cajón multilínea con tope estricto de altura de 92px para no obstruir el césped) y `COLLAPSED` (oculto).
  - **Coalescencia de Eventos:** Filtro inteligente de enfriamiento (1.0s para pérdidas/recuperaciones, 2.0s para duelos de Jev AI) para evitar saturación de botones.
  - **Píldoras de Filtrado Multicategoría:** Filtrado instantáneo (`Todos`, `⚡ Robos`, `🧠 Jev AI`) que sincroniza tanto los pips de la línea de tiempo como los chips de eventos.
  - **Salto Entre Momentos Contiguos:** Botones de navegación (`⏮️ Ant.` / `⏭️ Sig.` o teclas `J` / `K`) con avisos flotantes (toasts) protegidos contra condiciones de carrera.
* **Vista Limpia del Campo (Modo Cine):** Botón `👁️ Vista` o tecla `H` para ocultar o mostrar suavemente los paneles laterales (`.legend-card` y `#carrier-hud-card`) y el cajón de momentos para una visibilidad completa del partido.
* **Marcadores de Pérdida/Recuperación en la Línea de Tiempo (Pips):** Marcas interactivas superpuestas sobre la barra de tiempo (pips dorados para robos/cambios de posesión y pips violetas para intervenciones de Jev AI) con tooltips descriptivos y navegación instantánea al hacer clic.
* **Caché Persistente de Texturas de Equipación:** Almacenamiento a nivel de módulo para las texturas de camisetas blaugrana, rival y portero con protección contra doble eliminación (`dispose`), garantizando cero fugas de memoria al recargar datos en caliente (`🔄 Recargar Datos`).
* **Bucle de Renderizado 100% Zero-GC a 60 FPS:** Motor de animación de alto rendimiento libre de recolección de basura mediante vectores y colores de trabajo preasignados (`_scratchV3`, `_scratchTarget`, `_scratchColorJev`, `_scratchP1Map`), erradicando micro-tirones y caídas de cuadros.
* **API Global de Control IA (`window.PossessionViewerAPI`):** Preparada para integración con Gemini Live y agentes multimodales:
  - `getCurrentFrameState()`: Devuelve el cuadro activo, coordenadas 3D del balón interpoladas, equipo en posesión y eventos tácticos.
  - `captureCanvasBase64()`: Devuelve `renderer.domElement.toDataURL('image/jpeg', 0.8)` (`preserveDrawingBuffer: true`).
  - `setCameraMode(mode)`, `setBrainMode(mode)`, `setSpeed(speed)`, `seekToSecond(sec)`, `togglePlay()`, `toggleCleanHud()`, `jumpToAdjacentEvent(dir)`, `setEventFilter(filter)`, `cycleMomentsViewMode()`.
* **Conmutador de Comparación de IA:** Alterna en tiempo real entre **🧠 Nuestra IA** (Viterbi + Jev) y la **🤖 IA Común** (heurística de jugador más cercano).
* **Escudo 3D Anti-Robo (Pass Shield):** Una esfera alámbrica dorada que protege el balón en pases filtrados de alta velocidad cuando cruza cerca de rivales sin ser interceptado.
* **4 Perspectivas de Cámara:**
  - 🎥 **Cámara TV:** Seguimiento suave de la acción de retransmisión.
  - 🚁 **Dron Cercano:** Enfoque cinemático de seguimiento a baja altura.
  - 🏟️ **Pizarra Aérea:** Vista cenital táctica ortogonal 2D.
  - 🕹️ **Vista Libre (Órbita Manual):** Rotación completa en 360° mediante clic y arrastre con el mouse y zoom con la rueda (12m a 140m).
* **Atajos de Teclado:** `Espacio` (Pausar/Reanudar), `← / →` (-1s / +1s), `1-4` (Modos de Cámara), `M` (Alternar IA), `H` (Vista Limpia), `J` (Momento Anterior), `K` (Momento Siguiente).

---

## 5. Estructura del Repositorio

```
.
├── docs/
│   ├── technical_submission_report.md          # Informe técnico completo de la arquitectura RFC v2.0
│   ├── competition_quick_start_guide.md        # Guía rápida de comandos y ejecución el día de la competencia (ES)
│   ├── competition_quick_start_guide_english.md # Guía rápida de comandos y ejecución el día de la competencia (EN)
│   └── guia_validacion_criterios.md            # Criterios de evaluación de los jueces y checklist de entrega
├── auto_process_competition.py            # Escaneo batch automatizado e inferencia de partidos
├── generate_3d_replay_html.py             # Generador de repeticiones 3D interactivas WebGL / Three.js
├── ingest_and_test_metrica_sample.py      # Suite de ingesta y validación de muestra Metrica Sports
├── run_submission_pipeline.py             # CLI unificado de inferencia, entrenamiento y submission
├── task1_schema_and_preprocessing.py      # Preprocesamiento segmentado consciente de calidad
├── task2_feature_engine.py                # Evidencia cinemática, alcance y control físico
├── task3_anchors_and_bridges.py           # Anclas salientes, puentes acotados y refinador Jev
├── task4_classifier_and_decoder.py        # GBDT simétrico team-swap y decodificador Soft Viterbi
├── task5_evaluation_and_ablation.py       # Métricas de evaluación, robos bipartitos y ablación
├── task6_match_visualizer.py              # Visualizador táctico de campo 2D y panel diagnóstico
├── .gitignore                             # Exclusión de cachés, datasets pesados y venv
├── README.md                              # Documentación principal en inglés americano (Default)
├── README.es.md                           # Documentación completa en español
└── README.ca.md                           # Documentació completa en català
```

---

## 6. Licencia y Atribución

Este proyecto se distribuye bajo la **Licencia MIT**.  
Desarrollado para la competición de tracking de fútbol *More than a Hack 2027* (FC Barcelona & Metrica Sports).
