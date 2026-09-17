"""Ajuste interactivo del umbral HSV de color_mask.py.

Abre una ventana con sliders y muestra, en tiempo real, cómo cambian la máscara
y el resultado al mover cada parámetro. Sirve tanto sobre una imagen fija como
sobre una cámara/URL de video.

Orden del procesamiento:
    blur -> umbral HSV -> abrir -> cerrar -> erosionar -> dilatar
         -> filtro por area/forma -> relleno (huecos o envolvente convexa)

Sliders:
    H/S/V min-max   rango de color. Si H min > H max el rango es envolvente
                    (cruza el 0 del circulo de matiz), que es el caso del rojo.
    Blur            desenfoque gaussiano previo, kernel 2n+1.
    Abrir / Cerrar  morfologia: borra ruido suelto / tapa huecos y une trozos.
    Erosion/Dilatar ajuste fino del tamano del area.
    Relleno         0 = ninguno, 1 = rellena huecos cerrados (floodfill),
                    2 = envolvente convexa de cada blob (convex hull).
    Area min/max    limites de area del objeto, en unidades de 100 px.
                    Area max = 0 significa sin limite.
    Circular %      circularidad minima 4*pi*A/P^2 (100 = circulo perfecto).

Uso:
    python3 scripts/color_mask_interactivo.py                 # usa fotos/foto.png
    python3 scripts/color_mask_interactivo.py fotos/oscuro.png
    python3 scripts/color_mask_interactivo.py 0               # webcam local
    python3 scripts/color_mask_interactivo.py http://10.59.210.39:8080/video

Teclas: [s] guardar máscara + parámetros en detecciones/, [r] reiniciar valores,
[p] imprimir parámetros en consola, [ESC] o [q] salir.
"""

import os
import sys
import time

import cv2
import numpy as np

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_FOTOS = os.path.join(RAIZ_PROYECTO, "fotos")
CARPETA_DETECCIONES = os.path.join(RAIZ_PROYECTO, "detecciones")
RUTA_IMAGEN_POR_DEFECTO = os.path.join(CARPETA_FOTOS, "foto.png")

VENTANA_VISTA = "Original | Mascara | Resultado"
VENTANA_CONTROLES = "Controles"

ANCHO_PANEL = 460  # ancho de cada uno de los tres paneles del mosaico
ANCHO_MAXIMO_PROCESO = 1280  # se reduce la imagen antes de procesar, por fluidez

# Los sliders de área trabajan en unidades de 100 px para que el rango del
# trackbar (entero) cubra desde una mancha de ruido hasta una cereza completa.
UNIDAD_AREA = 100

SIN_RELLENO, RELLENAR_HUECOS, ENVOLVENTE_CONVEXA = 0, 1, 2
NOMBRE_RELLENO = {SIN_RELLENO: "no", RELLENAR_HUECOS: "huecos", ENVOLVENTE_CONVEXA: "hull"}

# Valores iniciales = los que estaban fijos en color_mask.py (rojo cereza).
# El tono parte "envolvente" (min > max) porque el rojo cruza el 0 del círculo
# de matiz: el rango real es H >= 165 o H <= 10.
# Los nombres son cortos a propósito: la ventana de trackbars los trunca.
PARAMETROS_INICIALES = {
    "H min": 165,
    "H max": 10,
    "S min": 70,
    "S max": 255,
    "V min": 35,
    "V max": 255,
    "Blur": 3,
    "Abrir": 0,
    "Cerrar": 9,
    "Erosion": 2,
    "Dilatar": 1,
    "Relleno": RELLENAR_HUECOS,
    "Area min": 1,
    "Area max": 300,
    "Circular %": 0,
}

MAXIMOS = {
    "H min": 179,
    "H max": 179,
    "S min": 255,
    "S max": 255,
    "V min": 255,
    "V max": 255,
    "Blur": 10,
    "Abrir": 25,
    "Cerrar": 25,
    "Erosion": 25,
    "Dilatar": 25,
    "Relleno": 2,
    "Area min": 1,
    "Area max": 300,  # 0 = sin límite superior
    "Circular %": 100,
}

# Píxel bajo el cursor, para leer su HSV mientras se ajustan los sliders.
cursor = {"x": -1, "y": -1, "panel": 0}


# cv2 exige un callback; los valores se leen en el bucle principal
def _sin_accion(_valor):
    pass


def crear_controles():
    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 460, 620)
    for nombre, inicial in PARAMETROS_INICIALES.items():
        cv2.createTrackbar(nombre, VENTANA_CONTROLES, inicial, MAXIMOS[nombre], _sin_accion)


def leer_controles():
    return {
        nombre: cv2.getTrackbarPos(nombre, VENTANA_CONTROLES)
        for nombre in PARAMETROS_INICIALES
    }


def reiniciar_controles():
    for nombre, inicial in PARAMETROS_INICIALES.items():
        cv2.setTrackbarPos(nombre, VENTANA_CONTROLES, inicial)


# máscara binaria cruda del rango HSV, sin limpieza morfológica
def umbralizar(imagen, p):
    if p["Blur"] > 0:
        lado = 2 * p["Blur"] + 1  # el kernel de blur debe ser impar
        imagen = cv2.GaussianBlur(imagen, (lado, lado), 0)

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    if p["H min"] <= p["H max"]:
        en_tono = (H >= p["H min"]) & (H <= p["H max"])
    else:
        # Rango envolvente (rojo): la banda cruza el 0 del círculo de matiz.
        en_tono = (H >= p["H min"]) | (H <= p["H max"])

    mascara = (
        en_tono
        & (S >= p["S min"]) & (S <= p["S max"])
        & (V >= p["V min"]) & (V <= p["V max"])
    )
    return (mascara * 255).astype(np.uint8)


# abrir borra el ruido suelto y cerrar tapa huecos y une trozos de un mismo objeto
def limpiar(mascara, p):
    operaciones = (
        (cv2.MORPH_OPEN, p["Abrir"]),
        (cv2.MORPH_CLOSE, p["Cerrar"]),
        # van al final porque acá ajustan el tamaño del área (encoger un blob que se
        # comió el fondo, recuperar el borde perdido por un umbral apretado), no limpian
        (cv2.MORPH_ERODE, p["Erosion"]),
        (cv2.MORPH_DILATE, p["Dilatar"]),
    )
    for operacion, lado in operaciones:
        if lado > 0:
            mascara = cv2.morphologyEx(mascara, operacion, np.ones((lado, lado), np.uint8))
    return mascara


# se inunda el fondo desde una esquina y lo que queda sin inundar son los huecos
# interiores (brillo especular del centro, zona oscura del pedicelo), que se suman
def rellenar_huecos(mascara):
    # borde de 1 px: deja el fondo alcanzable si un blob toca el borde de la imagen;
    # sin él ese blob encierra la esquina y se inunda la imagen entera
    con_borde = cv2.copyMakeBorder(mascara, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    semilla = np.zeros((con_borde.shape[0] + 2, con_borde.shape[1] + 2), np.uint8)
    cv2.floodFill(con_borde, semilla, (0, 0), 255)
    fondo_inundado = con_borde[1:-1, 1:-1]
    return cv2.bitwise_or(mascara, cv2.bitwise_not(fondo_inundado))


# se usan contornos externos y no connectedComponents porque así se obtiene el
# perímetro, necesario para la circularidad
def filtrar_objetos(mascara, p):
    area_minima = p["Area min"] * UNIDAD_AREA
    area_maxima = p["Area max"] * UNIDAD_AREA  # 0 = sin límite
    circularidad_minima = p["Circular %"] / 100.0
    usar_hull = p["Relleno"] == ENVOLVENTE_CONVEXA

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conservados = []
    for contorno in contornos:
        if usar_hull:
            # el hull cierra la silueta y rellena el interior sin aflojar el rango HSV,
            # que es lo que termina metiendo el fondo; área y circularidad se miden
            # sobre él, así lo que se ve en pantalla es lo filtrado y lo que se medirá
            contorno = cv2.convexHull(contorno)
        area = cv2.contourArea(contorno)
        if area <= 0 or area < area_minima:
            # Área 0 = restos de 1-2 px que deja una erosión fuerte, no objetos.
            continue
        if area_maxima > 0 and area > area_maxima:
            continue
        if circularidad_minima > 0:
            perimetro = cv2.arcLength(contorno, True)
            if perimetro == 0:
                continue
            # 4*pi*A/P^2 vale 1 en un círculo perfecto y baja con formas
            # alargadas o de borde irregular: separa cerezas de hojas, sombras
            # y pares de cerezas pegadas.
            if 4 * np.pi * area / (perimetro ** 2) < circularidad_minima:
                continue
        conservados.append(contorno)

    filtrada = np.zeros_like(mascara)
    cv2.drawContours(filtrada, conservados, -1, 255, cv2.FILLED)
    if not usar_hull:
        # drawContours rellena por definición: sin hull hay que recortar contra
        # la máscara original para conservar la forma exacta del blob. El
        # relleno de huecos, si se pidió, viene después y explícito.
        filtrada = cv2.bitwise_and(filtrada, mascara)

    return filtrada, conservados, len(contornos) - len(conservados)


# el relleno va DESPUÉS de filtrar por área: al revés, un 'Cerrar' grande une manchas
# del fondo en un blob enorme cuyo hull tapa media imagen y el filtro ya no lo descarta
def construir_mascara(imagen, p):
    mascara = limpiar(umbralizar(imagen, p), p)
    mascara, contornos, descartados = filtrar_objetos(mascara, p)
    if p["Relleno"] == RELLENAR_HUECOS:
        mascara = rellenar_huecos(mascara)
    return mascara, contornos, descartados


# doble trazo: el borde negro hace que el texto se lea sobre cualquier fondo
def escribir_texto(imagen, lineas, origen_y=22):
    for i, linea in enumerate(lineas):
        posicion = (10, origen_y + i * 20)
        cv2.putText(imagen, linea, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(imagen, linea, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# resumen de las áreas conservadas: guía para fijar los límites min/max
def texto_areas(contornos):
    if not contornos:
        return "areas: (sin objetos)"
    areas = np.array([cv2.contourArea(c) for c in contornos])
    return (
        f"areas px -> min {areas.min():.0f} | mediana {np.median(areas):.0f} | "
        f"max {areas.max():.0f}"
    )


# arma el mosaico original | máscara | resultado, con anotaciones
def componer_vista(imagen, mascara, contornos, descartados, p, escala):
    original = imagen.copy()
    cv2.drawContours(original, contornos, -1, (0, 255, 0), 2)

    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    paneles = []
    for panel, titulo in ((original, "Original"), (mascara_bgr, "Mascara"), (resultado, "Resultado")):
        panel = cv2.resize(panel, (ANCHO_PANEL, int(panel.shape[0] * ANCHO_PANEL / panel.shape[1])))
        escribir_texto(panel, [titulo])
        paneles.append(panel)

    mosaico = np.hstack(paneles)

    rango_tono = (
        f"H {p['H min']}-{p['H max']}"
        if p["H min"] <= p["H max"]
        else f"H {p['H min']}-179 + 0-{p['H max']} (envolvente)"
    )
    limite_superior = f"{p['Area max'] * UNIDAD_AREA}" if p["Area max"] > 0 else "sin limite"
    porcentaje = 100.0 * np.count_nonzero(mascara) / mascara.size
    lineas = [
        f"{rango_tono} | S {p['S min']}-{p['S max']} | V {p['V min']}-{p['V max']}",
        f"blur {p['Blur']} | abrir {p['Abrir']} | cerrar {p['Cerrar']} | "
        f"erosion {p['Erosion']} | dilatar {p['Dilatar']} | "
        f"relleno {p['Relleno']}={NOMBRE_RELLENO[p['Relleno']]}",
        f"area {p['Area min'] * UNIDAD_AREA} a {limite_superior} px | "
        f"circularidad min {p['Circular %']}%",
        f"objetos: {len(contornos)} (descartados {descartados}) | mascara: {porcentaje:.1f}% de la imagen",
        texto_areas(contornos),
        "cursor: (mueve el mouse sobre los paneles para leer su HSV)",
        "[s] guardar   [r] reiniciar   [p] imprimir parametros   [ESC] salir",
    ]

    # HSV del píxel apuntado: la referencia más útil para elegir los umbrales.
    x = int(cursor["x"] / escala)
    y = int(cursor["y"] / escala)
    if 0 <= x < imagen.shape[1] and 0 <= y < imagen.shape[0]:
        h, s, v = cv2.cvtColor(imagen[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
        dentro = "SI" if mascara[y, x] else "NO"
        detalle = ""
        for contorno in contornos:
            if cv2.pointPolygonTest(contorno, (float(x), float(y)), False) >= 0:
                detalle = f" | area del objeto: {cv2.contourArea(contorno):.0f} px"
                break
        lineas[5] = f"cursor ({x},{y}) HSV = {h},{s},{v} | en mascara: {dentro}{detalle}"

    # Barra de estado aparte, para no tapar la imagen con el texto.
    barra = np.zeros((20 * len(lineas) + 14, mosaico.shape[1], 3), np.uint8)
    escribir_texto(barra, lineas)
    return np.vstack([mosaico, barra])


def al_mover_mouse(evento, x, y, flags, param):
    if evento == cv2.EVENT_MOUSEMOVE:
        cursor["panel"] = x // ANCHO_PANEL
        cursor["x"] = x % ANCHO_PANEL  # coordenada dentro del panel apuntado
        cursor["y"] = y


def guardar(imagen, mascara, p):
    os.makedirs(CARPETA_DETECCIONES, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    ruta_mascara = os.path.join(CARPETA_DETECCIONES, f"mascara_{marca}.png")
    ruta_resultado = os.path.join(CARPETA_DETECCIONES, f"mascara_{marca}_resultado.png")
    ruta_parametros = os.path.join(CARPETA_DETECCIONES, f"mascara_{marca}_parametros.txt")

    cv2.imwrite(ruta_mascara, mascara)
    cv2.imwrite(ruta_resultado, cv2.bitwise_and(imagen, imagen, mask=mascara))
    with open(ruta_parametros, "w", encoding="utf-8") as archivo:
        for nombre, valor in p.items():
            archivo.write(f"{nombre} = {valor}\n")

    print(f"Guardado:\n  {ruta_mascara}\n  {ruta_resultado}\n  {ruta_parametros}")


def imprimir_parametros(p):
    print("\n# Parámetros actuales (para pegar en color_mask.py):")
    condicion_tono = (
        f"(H >= {p['H min']}) & (H <= {p['H max']})"
        if p["H min"] <= p["H max"]
        else f"((H >= {p['H min']}) | (H <= {p['H max']}))"
    )
    print(
        f"mask = {condicion_tono} & (S >= {p['S min']}) & (S <= {p['S max']})"
        f" & (V >= {p['V min']}) & (V <= {p['V max']})"
    )
    print(
        f"# blur={p['Blur']} abrir={p['Abrir']} cerrar={p['Cerrar']}"
        f" erosion={p['Erosion']} dilatar={p['Dilatar']}"
        f" relleno={NOMBRE_RELLENO[p['Relleno']]}"
    )
    print(
        f"# area_min={p['Area min'] * UNIDAD_AREA}"
        f" area_max={p['Area max'] * UNIDAD_AREA or 'sin limite'}"
        f" circularidad_min={p['Circular %'] / 100:.2f}\n"
    )


# devuelve (imagen_fija, captura); solo uno de los dos es distinto de None
def abrir_fuente(fuente):
    es_video = fuente.isdigit() or fuente.startswith(("http://", "https://", "rtsp://"))
    if es_video:
        cap = cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")
        return None, cap

    if not os.path.exists(fuente):
        raise RuntimeError(f"No existe la imagen {fuente}")
    imagen = cv2.imread(fuente)
    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen {fuente} (¿formato soportado?)")
    return imagen, None


# se procesa reducido por fluidez; la escala se devuelve para poder volver de una
# coordenada de la vista a la de la imagen original
def reducir(imagen):
    if imagen.shape[1] <= ANCHO_MAXIMO_PROCESO:
        return imagen, 1.0
    escala = ANCHO_MAXIMO_PROCESO / imagen.shape[1]
    return cv2.resize(imagen, None, fx=escala, fy=escala), escala


def main():
    fuente = sys.argv[1] if len(sys.argv) > 1 else RUTA_IMAGEN_POR_DEFECTO
    imagen_fija, cap = abrir_fuente(fuente)
    print(f"Fuente: {fuente}")

    crear_controles()
    cv2.namedWindow(VENTANA_VISTA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(VENTANA_VISTA, al_mover_mouse)

    if imagen_fija is not None:
        imagen, _ = reducir(imagen_fija)

    parametros = leer_controles()
    mascara = None

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("Se perdió la señal de video.")
                break
            imagen, _ = reducir(frame)

        parametros = leer_controles()
        mascara, contornos, descartados = construir_mascara(imagen, parametros)

        # Escala entre el panel mostrado y la imagen procesada, para mapear el cursor.
        escala_panel = ANCHO_PANEL / imagen.shape[1]
        cv2.imshow(
            VENTANA_VISTA,
            componer_vista(imagen, mascara, contornos, descartados, parametros, escala_panel),
        )

        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (27, ord("q")):  # ESC
            break
        if tecla == ord("s"):
            guardar(imagen, mascara, parametros)
        elif tecla == ord("r"):
            reiniciar_controles()
        elif tecla == ord("p"):
            imprimir_parametros(parametros)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    imprimir_parametros(parametros)


if __name__ == "__main__":
    main()
