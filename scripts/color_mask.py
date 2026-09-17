import os

import cv2
import numpy as np

# Ruta anclada a la raíz del proyecto, no al directorio desde donde se ejecuta.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_IMAGEN = os.path.join(RAIZ_PROYECTO, "fotos", "foto.png")

img = cv2.imread(RUTA_IMAGEN)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(hsv)

# Rojo
mask = ((H < 10) | (H > 165)) & (S > 70) & (V > 35)

mask = (mask * 255).astype(np.uint8)

# Limpiar ruido
kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    np.ones((9, 9), np.uint8)
)

# Componentes conectados
num_labels, labels, stats, centroids = \
    cv2.connectedComponentsWithStats(mask)

# show in the same window until a key is pressed
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
