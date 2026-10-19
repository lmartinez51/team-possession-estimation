# Estimació de la Possessió d'Equip per Fotograma (Frame-Level Team Possession Estimation)

🌐 **Llengua / Language / Idioma:** [English](README.md) | [Español](README.es.md) | [Català](README.ca.md)

> **Arquitectura RFC v2.0:** GBDT Cinemàtic Conscient de Qualitat + Àncores de Contacte Jev System One + Descodificador Seqüencial Soft Viterbi  
> **Competició Objectiu:** *More than a Hack 2027* (FC Barcelona & Metrica Sports)  
> **Autor:** Lorenzo Martínez  

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pipeline Tests](https://img.shields.io/badge/pipeline%20tests-passing%20100%25-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-RFC%20v2.0-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

---

## 1. Resum Executiu i Tesi Central

En l'analítica de futbol d'elit, **la possessió d'equip a nivell de fotograma és una interpretació temporal del control de la pilota i no la simple identitat estàtica del jugador més proper**.

Durant una passada filtrada a alta velocitat o un canvi d'orientació del FC Barcelona, la projecció planar 2D de la pilota transita habitualment a distàncies submètriques dels defensors rivals sense que hi hagi cap contacte físic. Les heurístiques ingènues de proximitat pateixen d'un parpelleig catastròfic d'etiquetes durant aquests encreuaments, mentre que els filtres de suavitzat temporal rígids esborren intercepcions legítimes d'un sol toc i transicions ràpides.

Aquest repositori implementa una solució de qualitat de producció i rigorosament fonamentada en matemàtiques, substituint les heurístiques ingènues per **evidència probabilística ponderada per confiança** i **descodificació seqüencial estructurada suau**:

```
                       +-----------------------------------+
                       | Optical Tracking Data (Ball/Play) |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 1: Preprocessador Segmentat  |
                       | - Interpolació acotada d'angles   |
                       | - Proxy multi-escala no circular  |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 2: Cinemàtica i Abast Físic  |
                       | - Horitzó curt tau [0.1, 1.5s]    |
                       | - Toc vs conducció desacoblada    |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 3: Àncores i Ponts           |
                       | - Àncores de contacte probables   |
                       | - Ponts no causals acotats        |
                       | - Refinador dispers Jev (Sys 1)   |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 4: Classificador i Descodifica|
                       | - GBDT simètric exacte team-swap  |
                       | - Relaxació dinàmica Soft Viterbi |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Task 5: Avaluació i Ablació       |
                       | - Emparellament bipartit canvis   |
                       | - Slices diagnòstics d'error      |
                       | - Suite d'ablació de 6 etapes     |
                       +-----------------------------------+
```

---

## 2. Arquitectura Modular en Sis Etapes

La base de codi està estructurada en sis mòduls de pipeline auto-verificables i amb fuga de dades zero (zero-leakage):

### [Task 1: Validació d'Esquema i Preprocessament](file:///task1_schema_and_preprocessing.py)
* **Parametrització Temporal Estricta en Segons:** Converteix totes les finestres i passos de derivades dinàmicament mitjançant marques de temps i freqüència de mostreig $f_s$, garantint un comportament físic idèntic tant a $25\text{ Hz}$ com a $50\text{ Hz}$.
* **Interpolació Acotada d'Oclusions:** Els buits curts de seguiment ($\le 0.40\text{ s}$) s'interpolen linealment de manera estricta dins de segments continus registrant `ball_is_interpolated`. Els buits superiors ($> 0.40\text{ s}$), límits de període i canvis de càmera tallen els identificadors de segment per evitar contaminació de derivades.
* **Proxy de Canvi de Velocitat Multi-Escala No Circular $I_b(t)$:** Detecta prèviament xuts i impactes abans del suavitzat mitjançant anàlisi de pendents MCO (mínims quadrats ordinaris) cap endavant i cap enrere ($0.12\text{ s}$ i $0.24\text{ s}$), assegurant que les passades potents mai quedin desdibuixades per filtres de Savitzky-Golay.

### [Task 2: Característiques Cinemàtiques Locals i Abast Físic](file:///task2_feature_engine.py)
* **Cerca d'Intercepció en Horitzó Curt:** Calcula horitzons d'abast físic $\Delta \tau \in [0.1, 1.5]\text{ s}$ sota límits de velocitat d'esprint ($9.5\text{ m/s}$), acceleració màxima ($7.0\text{ m/s}^2$) i retard biològic de reacció ($\tau_{\text{react}} = 0.30\text{ s}$).
* **Evidència de Control Desacoblada:** Separa l'evidència d'impulsos instantanis de toc de les conduccions sostingudes alineades en velocitat.
* **Marges d'Ambigüitat Espacial:** Avalua marges de distància entre el candidat top-1 vs. top-2 i la congestió de rivals dins de radis de $2.0\text{ m}$ i $5.0\text{ m}$.

### [Task 3: Àncores Probabilístiques i Ponts Acotats](file:///task3_anchors_and_bridges.py)
* **Àncores Salients de Contacte:** Identifica moments de control d'alta confiança i aïlla rebots contra la gespa de contactes humans reals.
* **Ponts No Causals Acotats:** Propaga probabilitats passades i futures cap endavant i cap enrere a través de passades en vol ($\le 4.0\text{ s}$), aturant-se immediatament davant de buits de seguiment $> 0.50\text{ s}$ o quan sorgeix evidència contradictòria de control.
* **Refinador Dispers TypeSafe AI (`Jev / System One`):** Dirigeix selectivament únicament els fotogrames d'alta ambigüitat ($\sim 5\text{--}10\%$) mitjançant primitives calibrades `Noul`, `Choice` i `Score` per desambiguar pilotes dividides i passades properes sense incórrer en el cost de processar tot el partit amb LLMs.

### [Task 4: GBDT Simètric i Descodificador Soft Viterbi](file:///task4_classifier_and_decoder.py)
* **Búfer Analític de Purga de Dependències:** Dedueix la separació temporal mínima exacta lliure de fuites per a validació creuada intra-partit:
  $$\Delta t_{\text{purge}} = \frac{1}{2}W_{\text{smooth}} + W_{\text{deriv}} + \tau_{\text{reach}} + \frac{1}{2}W_{\text{rolling}} + \Delta t_{\text{bridge}} = 7.38\text{ segons}$$
* **Invariància Exacta per Intercanvi d'Equip (Team-Swap):** Imposa característiques aparellades i avaluació antisimètrica, garantint simetria algebraica de probabilitat:
  $$|P(\text{Home} \mid \mathbf{x}_t) - (1 - P(\text{Home} \mid \text{Swap}(\mathbf{x}_t)))| \le 2.22 \times 10^{-16}$$
* **Descodificació de Cadena Discriminativa Suau:** Descodifica emissions de fotogrames mitjançant una cadena de Viterbi modificada el cost de transició de la qual es relaxa dinàmicament només quan convergeixen simultàniament contactes fiables i evidència de control rival.

### [Task 5: Avaluació Diagnòstica i Estudi d'Ablació](file:///task5_evaluation_and_ablation.py)
* **Priors de Geometria Tàctica Resistents a Outliers:** Combina l'àrea del Convex Hull de jugadors de camp amb la dispersió espacial interquartílica normalitzada cap a la direcció canònica d'atac.
* **Emparellament Bipartit 1-a-1 de Canvis de Possessió:** Emparellament temporal greedy ($\pm 250\text{ ms}$) dins de segments continus per calcular F1 de recuperacions, precisió, exhaustivitat i error temporal amb signe en mil·lisegons.
* **Informe d'Errors per Slices Diagnòstics:** Avalua la precisió en segments d'alt risc (passades properes a rivals, buits interpolats, pilotes dividides 50/50 i fronteres de segment).

### [Task 6: Visualitzador Tàctic i Panell Diagnòstic](file:///task6_match_visualizer.py)
* **Panell Tàctic Multi-Panell:** Renderitza un visualitzador professional de 3 panells que combina vista 2D del camp (amb els colors blaugrana del FC Barcelona, coordenades mètriques, dorsals i seguiment de la pilota), corbes de probabilitat posterior amb possessió dominant ombrejada, i discontinuïtats cinemàtiques ($I_b$, probabilitat de passada a l'espai, concordança de ponts i activacions de Jev).
* **Ingesta Defensiva d'Esquemes:** Reconeix columnes de predicció sense fallar davant conjunts de prova cecs (sense ground truth) o taules de tramesa oficials.
* **Execució Headless i Escalable:** Basat en `matplotlib.use("Agg")`, des-serialització segura de models i zoom opcional per finestra temporal (`time_window_s=(t_start, t_end)`).

---

## 3. Resultats de l'Estudi d'Ablació Ordenat

Avaluat sobre partits sintètics de prova amb passades filtrades a alta velocitat, marcatges estrets, oclusions de $0.20\text{ s}$, pèrdues de pilota de $0.80\text{ s}$ i intercepcions al primer toc:

| Etapa d'Ablació | Precisió de Fotograma (%) | Macro-F1 | F1 Recuperacions ($\pm 250\text{ ms}$) | Error Temporal Mitjà (ms) | Falsos Canvis / Min | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1a. Baseline Jugador Més Proper** | 45.5% | 0.429 | 0.267 | 100.0 | 62.5 | 0.4575 |
| **1b. Filtre Simple d'Histèresi** | 49.8% | 0.476 | 0.400 | 100.0 | 34.1 | 0.4575 |
| **2. Cinemàtica Local + GBDT** | 82.8% | 0.820 | 0.125 | 40.0 | 159.1 | 0.1494 |
| **3. + Finestres de Context Temporal** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **4. + Àncores i Ponts Jev** | 89.2% | 0.889 | 0.186 | 90.0 | 99.4 | 0.1062 |
| **5. + Descodificador Soft Viterbi** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | 0.0137 |
| **6. + Priors Aeris i Tàctics (MVP)** | **99.4%** | **0.994** | **1.000** | **30.0** | **0.0** | **0.0053** |

* **Interval de Confiança Bootstrap 90% (Etapa 6):** $[99.2\%,\, 99.6\%]$

### Rendiment en Casos Extrems Diagnòstics (Etapa 6 OOF)
* **Passada Propera al Rival ($\text{Proximitat No Fiable} \ge 0.30$):** `100.0%`
* **Buits Curts Interpolats ($0.20\text{ s}$):** `100.0%`
* **Transicions en Pilotes Dividides ($\pm 120\text{ ms}$):** `87.5%` ($30.0\text{ ms}$ d'error temporal)
* **Regions de Frontera de Segment i Període:** `100.0%`

---

## 4. Guia d'Execució i CLI

### Configuració de l'Entorn

```bash
# Clonar el repositori
git clone https://github.com/lmartinez51/team-possession-estimation.git
cd team-possession-estimation

# Crear i activar entorn virtual Python
python -m venv .venv
# A Windows:
.\.venv\Scripts\activate
# A Linux/macOS:
# source .venv/bin/activate

# Instal·lar dependències requerides
pip install numpy pandas scipy scikit-learn matplotlib
```

### Execució de les Suites de Verificació

Cada tasca incorpora una suite de proves de verificació determinista:

```bash
# Tasca 1: Preprocessament i proxy de canvi de velocitat
python task1_schema_and_preprocessing.py

# Tasca 2: Abast cinemàtic i característiques de toc/conducció
python task2_feature_engine.py

# Tasca 3: Àncores, ponts i refinament amb Jev System One
python task3_anchors_and_bridges.py

# Tasca 4: Entrenament GBDT simètric i descodificació Soft Viterbi
python task4_classifier_and_decoder.py

# Tasca 5: Mètriques de diagnòstic, slices extrems i ablació completa
python task5_evaluation_and_ablation.py

# Tasca 6: Visualitzador tàctic i panell diagnòstic
python task6_match_visualizer.py
```

### CLI de Tramesa Unificada (`run_submission_pipeline.py`)

L'script principal [`run_submission_pipeline.py`](file:///run_submission_pipeline.py) orquestra les Tasques 1 a 5 en un CLI complet d'entrenament, calibratge out-of-fold, memòria cau de característiques i predicció.

#### Característiques Principals:
* **Ingesta Agnòstica de Format (`MetricaFormatAdapter`):** Ingressa tracking en dues taules (`frame_df` + `player_tracks_df`) o taules amples de Metrica (`Home_11_x`, `Home_11_y`, etc.), normalitzant capçaleres i proveint màscares per defecte.
* **Memòria Cau Determinista (`FeatureCacheManager`):** Emmagatzema característiques a `.feature_cache/` identificades per match ID, versió d'esquema (`RFC_v2.0.0_MVP`) i hash SHA-256 de configuració.
* **Preservació Explícita d'Estats Latents:** Genera la tramesa binària requerida (`Home` vs. `Away`) mentre preserva en columnes complementàries els estats interns (`Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`).
* **Integració amb TypeSafe AI (`Jev / System One`):** Suporta resolució d'ambigüitat mitjançant crides en viu a l'API (`--live-jev`) o fallback heurístic cinemàtic.

#### Exemples d'Ús del CLI:

```powershell
# 1. Predicció sobre partit amb refinament TypeSafe AI (Jev / System One):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_frames.csv --players ruta_players.csv --output artifacts\mi_submission.csv --live-jev

# 2. Porta de verificació completa end-to-end (entrena, calibra, serialitza bundle i prova partit de validació):
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode verify

# 3. Entrenar i exportar bundle de models:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode train --model-bundle artifacts\barca_possession_mvp_v2.pkl

# 4. Predicció sobre taula ampla de tracking de Metrica:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames ruta_Sample_Game_1_RawTrackingData.csv --output artifacts\mi_submission.csv

# 5. Predicció sobre fitxers de tracking dividits Home i Away de Metrica:
.\.venv\Scripts\python.exe run_submission_pipeline.py --mode predict --frames data_metrica\Sample_Game_1_RawTrackingData_Home_Team.csv --players data_metrica\Sample_Game_1_RawTrackingData_Away_Team.csv --output artifacts\reintento_metrica.csv --live-jev --max-frames 2500
```

#### Referència de Paràmetres:
* `--mode`: Mode d'execució (`verify`, `train`, `predict`). Per defecte: `verify`.
* `--frames`: Ruta al fitxer `frame_df` CSV/Parquet, taula ampla única o taula Home de Metrica.
* `--players`: Ruta al fitxer `player_tracks_df` CSV/Parquet o taula Away de Metrica (auto-detectat).
* `--meta`: Ruta opcional a metadades JSON (`match_id`, `sampling_rate_hz`, dimensions del camp).
* `--model-bundle`: Ruta al bundle de models `.pkl` (per defecte: `artifacts/barca_possession_mvp_v2.pkl`).
* `--output`: Ruta de sortida per al CSV de tramesa (per defecte: `artifacts/submission_output.csv`).
* `--live-jev`: Habilita consultes en directe a TypeSafe AI (`Jev / System One`) mitjançant `TYPESAFE_API_KEY`.
* `--max-frames`: Límit opcional de fotogrames per a proves ràpides.

### Ingesta Batch Automatitzada per a Dades Reals (`auto_process_competition.py`)

Quan el FC Barcelona i Metrica Sports publiquin les dades finals de la competició, [`auto_process_competition.py`](file:///auto_process_competition.py) actua com l'**orquestrador batch sense configuració**. Escaneja recursivament la carpeta de destinació, auto-detecta l'estructura dels fitxers, emparella els partits, executa la inferència i genera els CSVs de tramesa i panells visuals.

#### 1. Formats Suportats i Auto-Detecció:
* **Taules Dividides de Metrica Sports (`metrica_split`):**
  - Emparella automàticament fitxers `*Home_Team.csv` i `*Away_Team.csv` del mateix partit.
  - Parsea les capçaleres de 3 files, alinea marques de temps, extreu coordenades de la pilota i transforma les coordenades normalitzades `[0, 1]` a metres de terreny de joc centrats a `(0, 0)` (`[-52.5, 52.5]` m x `[-34.0, 34.0]` m).
* **Taules Relacionals Canòniques (`canonical_pair`):**
  - Emparella taules de metadades de fotogrames (`*frame*.csv/.parquet`) amb trajectòries de jugadors en format llarg (`*player*.csv/.parquet`).
* **Taules Amples Úniques (`wide_single`):**
  - Ingressa taules úniques amb jugadors distribuïts en columnes (`Home_11_x, Home_11_y...`).
* **Filtrat Intel·ligent i Rebuig de Soroll:**
  - Descarta automàticament fitxers d'esdeveniments (`*Events*.csv`), actes arbitrals i alineacions.

#### 2. Disposició de Carpetes:
Col·loca els fitxers de tracking descarregats directament dins del directori `data/` (o en subcarpetes dins d'ell):

```text
Possession-Estimation/
├── data/                                         # Col·locar aquí els partits descarregats
│   ├── Match_1_RawTrackingData_Home_Team.csv     # Tracking Local (Metrica)
│   ├── Match_1_RawTrackingData_Away_Team.csv     # Tracking Visitant (Metrica)
│   ├── Match_1_RawEventsData.csv                 # (Ignorat automàticament)
│   ├── Match_2_RawTrackingData_Home_Team.csv
│   └── Match_2_RawTrackingData_Away_Team.csv
├── artifacts/                                    # Carpeta destinació de resultats
│   ├── submission_Match_1.csv                    # CSV validat i sense valors nuls
│   ├── submission_Match_2.csv
│   └── visualizations/                           # Diagnòstics tàctics 2D
│       ├── Match_1_diagnostic.png
│       └── Match_2_diagnostic.png
└── auto_process_competition.py
```

#### 3. Variables d'Entorn:
Si utilitzes el desempat en viu amb TypeSafe AI (`Jev / System One`), configura la teva clau d'API:

```powershell
# Windows PowerShell:
$env:TYPESAFE_API_KEY = "la_teva_clau_api_aqui"

# Linux / macOS Bash:
export TYPESAFE_API_KEY="la_teva_clau_api_aqui"
```
*(Nota: Si no es defineix una clau d'API, el pipeline utilitza de forma transparent el fallback cinemàtic sense interrompre l'execució).*

#### 4. Comandaments d'Execució:

##### A. Execució Batch de Competició Completa (Producció):
Processa cada partit detectat a `data/`, aplica els models calibrats i refinament amb Jev, valida el contracte de tramesa i genera diagnòstics gràfics:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --visualize
```

##### B. Prova Ràpida de Fum / Sanity Check (~100 segons de partit):
Abans de processar partits de 90 minuts (~145,000 fotogrames cadascun), valida el flux en els primers 2,500 fotogrames amb `--max-frames`:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --live-jev --max-frames 2500 --visualize
```

##### C. Mode Offline / Heurístic (Sense Clau d'API):
Inferència purament determinista basada en cinemàtica local i relaxació temporal amb Soft Viterbi:

```powershell
.\.venv\Scripts\python.exe auto_process_competition.py --data-dir data --output-dir artifacts --visualize
```

#### 5. Verificació de Contracte i Sortides:
* **Ruta de Sortida:** `artifacts/submission_<match_id>.csv`
* **Columnes Generades:**
  - `frame`: Índex del fotograma (enter, estrictament ascendent).
  - `possession_team`: Equip en possessió (`Home` o `Away`).
  - `confidence`: Probabilitat posterior calibrada `[0.50, 1.00]`.
  - `possession_state`: Estat latent detallat (`Home`, `Away`, `Contested_Home`, `Contested_Away`, `Unknown_Missing_Ball`, `Dead_Ball`).
* **Panell Visual:** Panell tàctic 2D amb vectors de velocitat, trajectòria de la pilota i probabilitat acumulada desat a `artifacts/visualizations/<match_id>_diagnostic.png`.

#### 6. Taula de Paràmetres del Batch CLI:
| Paràmetre | Per Defecte | Descripció |
|---|---|---|
| `--data-dir` | `data` | Directori on s'ubiquen els fitxers de tracking (cerca recursiva). |
| `--output-dir` | `artifacts` | Directori de sortida per a trameses i diagnòstics gràfics. |
| `--live-jev` | `False` | Habilita consultes LLM a TypeSafe AI (`Jev / System One`) en pilotes dividides. |
| `--max-frames` | `None` | Límit de fotogrames processats per partit (ex. `2500` per a proves de fum). |
| `--visualize` | `False` | Genera i desa panells gràfics de 3 panells en alta resolució. |
| `--model-bundle` | `artifacts/barca_possession_mvp_v2.pkl` | Ruta al bundle del model LightGBM entrenat. |

### Repetició 3D Interactiva per a Retransmissió (`generate_3d_replay_html.py`)

Per a demostracions tècniques, experiència de l'aficionat i anàlisi tàctica, [`generate_3d_replay_html.py`](file:///generate_3d_replay_html.py) compila les prediccions en una **experiència 3D desacoblada en WebGL** construïda amb Three.js. Tradueix conceptes matemàtics abstractes (ponts acotats, estats de Viterbi i contactes resolts per Jev) a elements visuals intuïtius.

```powershell
# 1. Exportar dades del partit i visualitzador 3D desacoblat (per defecte: 60s a 25Hz):
.\.venv\Scripts\python.exe generate_3d_replay_html.py --max-frames 1500

# 2. Obrir el visualitzador directament al navegador (compatible amb Zero-CORS file://):
Start-Process artifacts\visualizations\viewer.html

# 3. Forçar el restabliment o sobreescriptura de la plantilla visual:
.\.venv\Scripts\python.exe generate_3d_replay_html.py --reset-viewer

# 4. Compilar un fitxer HTML autònom i independent per compartir:
.\.venv\Scripts\python.exe generate_3d_replay_html.py --standalone
```

#### Arquitectura i Recàrrega en Calent (Hot-Reload):
* **Frontend Desacoblat i Zero-CORS:** El codi visual resideix a `artifacts/visualizations/viewer.html`, estrictament separat de `match_payload.js` (`window.MATCH_DATA = {...};`) i `match_payload.json`. Evita les restriccions locals de CORS dels navegadors en obrir directament amb doble clic via `file://`.
* **Guàrdia Anti-Sobreescriptura:** Executar Python actualitza sempre les dades del partit preservant qualsevol edició visual o de Three.js manual a `viewer.html` tret que s'especifiqui `--reset-viewer`.
* **Recàrrega en Calent a l'App:** Prem **🔄 Recargar Datos** a la barra inferior per refrescar `match_payload.js` en viu sense perdre la vista de càmera ni la velocitat de reproducció.
* **API Global de Control IA (`window.PossessionViewerAPI`):** Preparada per a la integració amb Gemini Live i agents multimodals:
  - `getCurrentFrameState()`: Retorna el fotograma actiu, coordenades 3D de la pilota interpolades, equip en possessió i esdeveniments tàctics.
  - `captureCanvasBase64()`: Retorna `renderer.domElement.toDataURL('image/jpeg', 0.8)` (`preserveDrawingBuffer: true`).
  - `setCameraMode(mode)`, `setBrainMode(mode)`, `setSpeed(speed)`, `seekToSecond(sec)`, `togglePlay()`.
* **Commutador de Comparació d'IA:** Alterna en temps real entre **🧠 La Nostra IA** (Viterbi + Jev) i la **🤖 IA Comuna** (heurística de jugador més proper).
* **Escut 3D Anti-Robatori (Pass Shield):** Una esfera de filferro daurada que protegeix la pilota en passades filtrades d'alta velocitat quan creua a prop de rivals sense ser interceptada.
* **Fil Daurat de Control:** Corba de Bézier 3D que uneix en tot moment el posseïdor legítim amb la pilota.
* **4 Perspectives de Càmera:**
  - 🎥 **Càmera TV:** Seguiment suau de l'acció de retransmissió.
  - 🚁 **Dron Proper:** Enfocament cinemàtic de seguiment a baixa alçada.
  - 🏟️ **Pissarra Aèria:** Vista zenital tàctica ortogonal 2D.
  - 🕹️ **Vista Lliure (Òrbita Manual):** Rotació completa en 360° mitjançant clic i arrossegament amb el ratolí i zoom amb la roda (12m a 140m).
* **Dreceres de Teclat:** `Espai` (Pausar/Reprendre), `← / →` (-1s / +1s), `1-4` (Modes de Càmera), `M` (Alternar IA).

---

## 5. Estructura del Repositori

```
.
├── docs/
│   └── technical_submission_report.md     # Informe tècnic complet de l'arquitectura RFC v2.0
├── auto_process_competition.py            # Escaneig batch automatitzat i inferència de partits
├── generate_3d_replay_html.py             # Generador de repeticions 3D interactives WebGL / Three.js
├── ingest_and_test_metrica_sample.py      # Suite d'ingesta i validació de mostra Metrica Sports
├── run_submission_pipeline.py             # CLI unificat d'inferència, entrenament i tramesa
├── task1_schema_and_preprocessing.py      # Preprocessament segmentat conscient de qualitat
├── task2_feature_engine.py                # Evidència cinemàtica, abast i control físic
├── task3_anchors_and_bridges.py           # Àncores salients, ponts acotats i refinador Jev
├── task4_classifier_and_decoder.py        # GBDT simètric team-swap i descodificador Soft Viterbi
├── task5_evaluation_and_ablation.py       # Mètriques d'avaluació, recuperacions bipartites i ablació
├── task6_match_visualizer.py              # Visualitzador tàctic de camp 2D i panell diagnòstic
├── .gitignore                             # Exclusió de memòries cau, conjunts de dades pesats i venv
├── README.md                              # Documentació principal en anglès americà (Per defecte)
├── README.es.md                           # Documentació completa en espanyol
└── README.ca.md                           # Documentació completa en català
```

---

## 6. Llicència i Atribució

Aquest projecte es distribueix sota la **Llicència MIT**.  
Desenvolupat per a la competició de tracking de futbol *More than a Hack 2027* (FC Barcelona & Metrica Sports).
