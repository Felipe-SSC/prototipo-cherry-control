"""Captura fotos de la cámara y las guarda como PNG con la barra espaciadora.

A diferencia de capturar.py, no valida ningún patrón: guarda el frame tal cual.
Sirve para juntar imágenes de cerezas para ajustar la máscara de color.

Uso:
    python3 scripts/capturar_fotos.py         # webcam local (índice 0)
    python3 scripts/capturar_fotos.py 1
    python3 scripts/capturar_fotos.py http://10.59.210.39:8080/video

Teclas: [ESPACIO] guardar PNG, [ESC] o [q] salir.
"""

import os
import sys
import time

import cv2

# Rutas ancladas a la raíz del proyecto: el script funciona igual se ejecute
# desde la raíz (python scripts/...) o desde scripts/. Lo que captura este
# script es imagen cruda, sin detección encima, así que va a fotos/.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_FOTOS = os.path.join(RAIZ_PROYECTO, "fotos")

VENTANA = "Captura - ESPACIO para guardar, ESC para salir"

fuente = sys.argv[1] if len(sys.argv) > 1 else "0"
cap = cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la fuente: {fuente}")

os.makedirs(CARPETA_FOTOS, exist_ok=True)
cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
print(f"Fuente: {fuente}\nGuardando en: {CARPETA_FOTOS}")

contador = 0
ultimo_guardado = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Se perdió la señal de video.")
        break

    vista = frame.copy()
    texto = f"{contador} fotos | ESPACIO=guardar  ESC=salir"
    cv2.putText(vista, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vista, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Marco verde durante medio segundo como confirmación visual del guardado.
    if time.time() - ultimo_guardado < 0.5:
        cv2.rectangle(vista, (0, 0), (vista.shape[1] - 1, vista.shape[0] - 1), (0, 255, 0), 8)

    cv2.imshow(VENTANA, vista)

    tecla = cv2.waitKey(20) & 0xFF
    if tecla in (27, ord("q")):  # ESC
        break
    if tecla == ord(" "):
        # Marca de tiempo + correlativo: la marca evita pisar capturas de
        # sesiones anteriores y el correlativo, dos fotos en el mismo segundo.
        marca = time.strftime("%Y%m%d_%H%M%S")
        ruta = os.path.join(CARPETA_FOTOS, f"foto_{marca}_{contador:03d}.png")
        cv2.imwrite(ruta, frame)  # se guarda el frame limpio, sin el texto
        contador += 1
        ultimo_guardado = time.time()
        print(f"Guardado: {ruta}")

cap.release()
cv2.destroyAllWindows()
print(f"{contador} fotos guardadas en {CARPETA_FOTOS}")
