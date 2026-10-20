# 🚀 Guía Rápida de Comandos — Día de la Competencia ("More than a Hack 2027")

Esta guía rápida está diseñada para que nuestro equipo pueda ejecutar el pipeline de estimación de posesión, correr inferencias con o sin Internet, y desplegar el visor 3D interactivo de manera inmediata el día del reto.

---

## 📂 1. Preparación Previa
Asegúrate de colocar los archivos oficiales del hackathon (ya sea en formato Metrica Sports dividido o tablas canónicas) dentro de la carpeta raíz del proyecto:
```powershell
.\data\
```

---

## ⚡ 2. Comandos Principales en la Terminal (PowerShell)

### A. Procesamiento Batch Automático (Modo Offline / Local por Defecto)
* **Cuándo usarlo:** Entornos cerrados (*air-gapped*), sin conexión a Internet, o cuando necesitamos velocidad instantánea. Utiliza el emulador local calibrado de System One.
* **Comando:**
  ```powershell
  .\.venv\Scripts\python.exe auto_process_competition.py
  ```
* **Resultado:** Escanea automáticamente la carpeta `data\`, procesa todos los partidos y genera los archivos finales `artifacts\submission_*.csv` limpios y validados (`0 NaNs`).

### B. Procesamiento Batch Avanzado con Jev AI en Vivo (Con Internet)
* **Cuándo usarlo:** Si la plataforma de competencia permite conexión exterior y queremos exprimir la máxima precisión probabilística en jugadas divididas complejas usando los servidores de TypeSafe AI.
* **Requisito previo:** Tener configurada la variable de entorno con la API Key:
  ```powershell
  $env:TYPESAFE_API_KEY = "tu_api_key_aqui"
  ```
* **Comando:**
  ```powershell
  .\.venv\Scripts\python.exe auto_process_competition.py --live-jev
  ```

### C. Actualizar y Generar el Visor 3D Interactivo para Espectadores
* **Cuándo usarlo:** Tras procesar los datos de la competencia, para actualizar la repetición 3D con la temática del FC Barcelona y la interfaz estilo transmisión de TV.
* **Comando:**
  ```powershell
  .\.venv\Scripts\python.exe generate_3d_replay_html.py --standalone
  ```
* **Resultado:** Actualiza `match_payload.js` con los datos reales del partido y genera el archivo portable `artifacts\visualizations\barca_3d_interactive_replay.html`.

---

## 🎮 3. Controles Clave dentro del Visor 3D (`viewer.html`)

Una vez que abras `artifacts\visualizations\viewer.html` en tu navegador, utiliza estos atajos y botones:

| Tecla / Botón | Acción en el Visor 3D |
| :--- | :--- |
| **`Espacio`** | Pausar o reanudar la reproducción de la repetición. |
| **`Teclas 1, 2, 3, 4`** | Alternar entre **Cámara TV**, **Dron Cercano**, **Pizarra Aérea** y **Cámara Libre** (Orbitar con mouse y zoom con rueda). |
| **`Tecla M`** | Comparar en vivo **`🧠 Nuestra IA`** vs. **`🤖 IA Común (Solo Cercanía)`** para evidenciar visualmente el parpadeo evitado. |
| **`Tecla H`** / **`👁️ Vista`** | Activar el **Modo Cine (Vista Limpia)** para ocultar las tarjetas laterales y despejar el 100% de la cancha. |
| **`Pips en la Línea de Tiempo`** | Hacer clic en las marcas de colores de la barra deslizadora para saltar directamente a los robos de balón o momentos de Jev AI. |

---

## ✅ 4. Checklist Rápido de Entrega
Antes de subir cualquier archivo a la plataforma de los jueces, verifica que:
1. La consola indique: `[PASS] Submission CSV written and contract-validated` con `0 NaNs`.
2. El porcentaje global de posesión (Home vs. Away) tenga coherencia futbolística realista.
3. Se hayan guardado los archivos `submission_*.csv` dentro de la carpeta `artifacts\`.