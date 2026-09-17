# prototipo-cherry-control

Prototipo de control de calidad de cerezas por visión: detecta cada fruta en una
foto, la clasifica por color en uno de tres tipos y estima su calibre.

Es el proyecto hermano de [cam-calibration-toolkit](../cam-calibration-toolkit),
que resuelve un problema distinto: calibrar la cámara y medir segmentos simples
con un tablero de referencia. Ahí se obtiene la escala mm/px; acá se usa.

## Estructura

```
scripts/       todo el código
fotos/         capturas y screenshots crudos, sin ninguna detección encima
detecciones/   todo lo que escriben los scripts: csv, vistas anotadas, máscaras
```

La regla es esa y no tiene excepciones: si una imagen tiene un contorno, un
número o una máscara dibujada encima, es salida y va en `detecciones/`.

Los scripts anclan sus rutas a la raíz del proyecto, así que da lo mismo si los
ejecutas desde la raíz o desde `scripts/`.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Usa `opencv-python` (con GUI, no la variante headless): varios scripts abren
ventanas con `cv2.imshow`. El venv es propio de este proyecto; el toolkit tiene
el suyo.

## Scripts

| Script | Para qué sirve |
|---|---|
| [scripts/capturar_fotos.py](scripts/capturar_fotos.py) | Captura frames de una cámara o URL a `fotos/`. ESPACIO guarda, ESC sale |
| [scripts/color_mask_interactivo.py](scripts/color_mask_interactivo.py) | Ajusta los umbrales HSV con sliders y ve el efecto en vivo. `[s]` guarda máscara + `.txt` de parámetros en `detecciones/` |
| [scripts/clasificar_cerezas.py](scripts/clasificar_cerezas.py) | El script principal: cuenta cerezas por tipo, mide el calibre de cada una y escribe el csv + la vista anotada |
| [scripts/medir_cerezas.py](scripts/medir_cerezas.py) | Diagnóstico del calibre: compara las tres formas de estimarlo y dibuja la medición sobre cada fruta |
| [scripts/color_mask.py](scripts/color_mask.py) | Versión mínima y fija de la máscara de color, sin sliders. Sirve de referencia |
| [scripts/csv_result.ipynb](scripts/csv_result.ipynb) | Explora un csv de `detecciones/`: distribución de tipos y de calibre |

Flujo habitual:

```
capturar_fotos.py  ->  color_mask_interactivo.py  ->  clasificar_cerezas.py  ->  csv_result.ipynb
                          (ajustar y guardar               (--params con ese
                           los umbrales con [s])            .txt)
```

```powershell
python scripts/clasificar_cerezas.py                        # fotos/claro_centro.png
python scripts/clasificar_cerezas.py fotos/oscuro.png
python scripts/clasificar_cerezas.py fotos/oscuro.png --mm-por-px 0.35
python scripts/clasificar_cerezas.py fotos/oscuro.png --params detecciones/mascara_..._parametros.txt
```

## Cómo se estima el calibre

El calibre es el **eje mayor de la elipse ajustada** a la silueta
(`cv2.fitEllipse`), no el Feret máximo. El Feret es la mayor distancia entre dos
puntos de la envolvente convexa, así que basta un vértice equivocado —un
pedicelo que agarró la máscara, una sombra, dos cerezas que se tocan— para
inflarlo. El ajuste por mínimos cuadrados usa todos los puntos del contorno y un
vértice suelto lo mueve poco.

`medir_cerezas.py` calcula y contrasta las tres aproximaciones (eje mayor, Feret
máximo, diámetro equivalente) y reporta cuánto se separan entre sí. Sobre
`fotos/claro_centro.png` la diferencia entre Feret y eje mayor es de 0,6% en
promedio, con máximo de 3,0%.

## Tres limitaciones que hay que tener presentes

**La escala.** Sin `--mm-por-px` el calibre sale en píxeles: una foto sola no
trae escala. Para milímetros hace falta una referencia de tamaño conocido en la
misma foto y al mismo plano — el tablero del toolkit vecino, o una regla.

**La luz.** Los centroides de color de `TIPOS` son valores de brillo absolutos,
medidos con la iluminación de `fotos/claro_centro.png`. Con otra luz las mismas
cerezas dan V más bajo y quedan fuera de rango; el script avisa cuando más del
30% de los objetos quedan sin clasificar. Ahí no basta con ajustar la detección:
hay que volver a medir los centroides con esa luz.

**La frontera entre tipo 2 y tipo 3.** Entre esos dos no hay hueco natural en la
nube de brillo: el corte en V ≈ 192 parte la nube por el medio, así que el
reparto es sensible a dónde quede. Para afirmarlo mejor hacen falta más cerezas
marcadas a mano de esos dos tipos, sobre todo en V 185-200.

Ninguna de las tres métricas de calibre está anclada todavía a una medición
física: falta comparar contra pie de metro sobre cerezas conocidas.
