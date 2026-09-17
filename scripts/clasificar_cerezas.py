import argparse
import os
import time

import cv2
import numpy as np

import color_mask_interactivo as cmi

# Rutas ancladas a la raíz del proyecto: los scripts funcionan igual se ejecuten
# desde la raíz (python scripts/...) o desde scripts/.
#   fotos/       capturas y screenshots crudos, sin ninguna detección encima
#   detecciones/ todo lo que escriben los scripts (csv, vistas anotadas, máscaras)
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_FOTOS = os.path.join(RAIZ_PROYECTO, "fotos")
CARPETA_DETECCIONES = os.path.join(RAIZ_PROYECTO, "detecciones")
RUTA_POR_DEFECTO = os.path.join(CARPETA_FOTOS, "claro_centro.png")

# Centroides HSV medidos sobre las cerezas marcadas en ejemplo.png.
# n = cuántas cerezas marcadas respaldan cada centroide.
# Los colores de dibujo son los mismos que usó Felipe en ejemplo.png.
TIPOS = {
    1: {"nombre": "tipo 1 (mas claro)", "H": 1.3, "S": 156.0, "V": 241.7, "n": 3, "color": (255, 0, 0)},
    2: {"nombre": "tipo 2 (intermedio)", "H": 1.4, "S": 134.8, "V": 205.3, "n": 2, "color": (0, 180, 0)},
    3: {"nombre": "tipo 3 (mas oscuro)", "H": 1.0, "S": 125.9, "V": 178.6, "n": 5, "color": (0, 0, 0)},
}

# Una cereza tiene que ser roja y saturada; con esto se descarta cualquier
# objeto que haya pasado la detección sin serlo (una mancha del piso, un reflejo).
S_MINIMO_CEREZA = 95
V_MINIMO_CEREZA = 90

COLOR_SIN_CLASIFICAR = (0, 0, 255)

# Detección pensada para fotos de 1920 px de ancho (las cerezas miden ~70 px de
# diámetro, unos 3.900 px de área). Si cambia la resolución hay que reescalar
# 'Area min' y 'Area max', que están en unidades de 100 px.
#PARAMETROS_DETECCION = {
#"H min": 160, "H max": 12, "S min": 60, "S max": 255, "V min": 0, "V max": 255,
#    "Blur": 0, "Abrir": 7, "Cerrar": 9, "Erosion": 0, "Dilatar": 0,
#    "Relleno": cmi.ENVOLVENTE_CONVEXA, "Area min": 25, "Area max": 100, "Circular %": 80,
#}


PARAMETROS_DETECCION = {
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
    "Relleno": cmi.ENVOLVENTE_CONVEXA,
    "Area min": 1,
    "Area max": 100,
    "Circular %": 0,
}


# devuelve [(id_tipo, v_minimo, v_maximo)], de más claro a más oscuro
# los límites salen de TIPOS y no fijados a mano: al recalibrar un centroide se mueven solos
def fronteras_de_brillo():
    orden = sorted(TIPOS, key=lambda t: -TIPOS[t]["V"])
    # cada corte es el punto medio del eje V entre dos centroides vecinos
    cortes = [
        (TIPOS[orden[i]]["V"] + TIPOS[orden[i + 1]]["V"]) / 2.0
        for i in range(len(orden) - 1)
    ]
    limites = []
    for i, tipo in enumerate(orden):
        techo = 256.0 if i == 0 else cortes[i - 1]
        piso = V_MINIMO_CEREZA if i == len(orden) - 1 else cortes[i]
        limites.append((tipo, piso, techo))
    return limites


# promedio de H circular porque el rojo cruza el 0/180 del círculo de matiz:
# promediar a secas daría ~90 (verde) para píxeles en 178 y en 2, que son el mismo rojo
def media_circular_tono(tonos):
    angulos = tonos.astype(np.float64) * 2.0 * np.pi / 180.0
    media = np.arctan2(np.sin(angulos).mean(), np.cos(angulos).mean())
    return float(np.degrees(media) / 2.0) % 180.0


def detectar_cerezas(imagen, parametros):
    # la piel es la segmentación SIN hull: son los píxeles de cereza de verdad y es
    # sobre esos que se promedia el color (el hull metería fondo en los bordes)
    piel = cmi.limpiar(cmi.umbralizar(imagen, parametros), parametros)
    # los contornos sí van con envolvente convexa: dan una silueta cerrada por cereza
    _, contornos, _ = cmi.filtrar_objetos(piel, parametros)
    return contornos, piel


# helper sin uso hoy: main escribe la piel directo con cv2.imwrite
def guardar_piel_mask(piel, ruta):
    cv2.imwrite(ruta, piel)
    print(f"Máscara de piel guardada en {ruta}")


# el promedio se toma solo sobre los píxeles de piel dentro del contorno, no sobre
# todo el contorno: el hull mete fondo en los bordes y correría el color
def color_promedio(hsv, piel, contorno):
    recorte = np.zeros(piel.shape, np.uint8)
    cv2.drawContours(recorte, [contorno], -1, 255, cv2.FILLED)
    pixeles = hsv[(recorte > 0) & (piel > 0)]
    if len(pixeles) == 0:
        return None
    return {
        "n": len(pixeles),
        "H": media_circular_tono(pixeles[:, 0]),
        "S": float(pixeles[:, 1].mean()),
        "V": float(pixeles[:, 2].mean()),
    }


# mayor distancia entre dos puntos de la silueta, el pie de metro abierto al máximo
# ya NO es el calibre (ver calibre): al ser un máximo, un vértice espurio del hull lo infla
# queda porque medir_cerezas.py lo compara y porque respalda cuando la elipse no ajusta
def feret_maximo(contorno):
    # los pares se buscan sobre el hull y no sobre el contorno completo: el punto más
    # lejano siempre cae en él, y los puntos bajan de cientos a unas decenas
    hull = cv2.convexHull(contorno).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 0.0, (None, None)
    distancias = np.linalg.norm(hull[:, None, :] - hull[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(distancias)), distancias.shape)
    return float(distancias[i, j]), (hull[i].astype(int), hull[j].astype(int))


# diámetro del círculo que tiene la misma área que la silueta
def diametro_equivalente(contorno):
    area = cv2.contourArea(contorno)
    return float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0


def eje_mayor_elipse(contorno):
    if len(contorno) < 5:  # fitEllipse exige al menos 5 puntos
        return 0.0, None
    elipse = cv2.fitEllipse(contorno)
    return float(max(elipse[1])), elipse


def calibre(contorno):
    # calibre es el eje mayor de la elipse ajustada
    eje, _ = eje_mayor_elipse(contorno)
    # si no se puede ajustar, se opta por feret maximo del convex hull
    # suceptible a inflar el calibre por vértices mal detectados
    return eje if eje > 0 else feret_maximo(contorno)[0]


def medir(contorno):
    feret, extremos = feret_maximo(contorno)
    eje, elipse = eje_mayor_elipse(contorno)
    return {
        "feret_max": feret,
        "diam_equiv": diametro_equivalente(contorno),
        "eje_mayor": eje,
        "extremos": extremos,
        "elipse": elipse,
    }


def clasificar(color, limites):
    if color["S"] < S_MINIMO_CEREZA or color["V"] < V_MINIMO_CEREZA:
        return None
    for tipo, piso, techo in limites:
        if piso <= color["V"] < techo:
            return tipo
    return None


# avisa cuando la foto no se parece a la que se usó para calibrar: los centroides de
# TIPOS son brillos absolutos medidos con la luz de claro_centro.png, y en una foto más
# oscura las mismas cerezas caen fuera de rango (hay que remedirlos, el script no se ajusta solo)
def advertir_si_muchos_sin_clasificar(resultados):
    if not resultados:
        return
    sin_tipo = sum(1 for r in resultados if r["tipo"] is None)
    if sin_tipo / len(resultados) < 0.3:
        return
    v = [r["V"] for r in resultados]
    print(f"\n  AVISO: {sin_tipo} de {len(resultados)} objetos quedaron sin clasificar "
          f"(V medio {np.mean(v):.0f}).")
    print(f"  Los tipos estan calibrados para V entre {TIPOS[3]['V']:.0f} y {TIPOS[1]['V']:.0f}. "
          "Si esta foto tiene otra iluminacion,")
    print("  hay que volver a medir los centroides de TIPOS sobre ella; no basta con "
          "ajustar la deteccion.")


# lee un .txt guardado por color_mask_interactivo.py con la tecla [s]
def cargar_parametros(ruta):
    parametros = dict(PARAMETROS_DETECCION)
    with open(ruta, encoding="utf-8") as archivo:
        for linea in archivo:
            if "=" not in linea:
                continue
            nombre, valor = linea.split("=", 1)
            parametros[nombre.strip()] = int(valor.strip())
    return parametros


def texto_centrado(vista, texto, centro, escala, color):
    grosor = max(1, int(round(escala * 2.5)))
    (ancho, alto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    origen = (centro[0] - ancho // 2, centro[1] + alto // 2)
    # el borde blanco es lo que hace legible el tipo 3 (negro) sobre la cereza
    for tono, trazo in ((255, 255, 255), grosor + 2), (color, grosor):
        cv2.putText(vista, texto, origen, cv2.FONT_HERSHEY_SIMPLEX, escala, tono, trazo, cv2.LINE_AA)


# dibuja cada cereza con el color de su tipo y el resumen del conteo
def anotar(imagen, resultados, limites, factor, unidad):
    vista = imagen.copy()
    for r in resultados:
        color = TIPOS[r["tipo"]]["color"] if r["tipo"] else COLOR_SIN_CLASIFICAR
        cv2.drawContours(vista, [r["contorno"]], -1, color, 3)
        # tres números por cereza, de arriba a abajo: id, tipo y calibre
        # el id es el del csv, y es lo que permite ir a revisar el HSV de una dudosa
        texto_centrado(vista, str(r["id"]), (r["x"], r["y"] - 24), 0.55, color)
        etiqueta = str(r["tipo"]) if r["tipo"] else "?"
        texto_centrado(vista, etiqueta, (r["x"], r["y"]), 0.8, color)
        texto_centrado(vista, f"{r['calibre'] * factor:.0f}", (r["x"], r["y"] + 24), 0.55, color)

    conteo = {t: sum(1 for r in resultados if r["tipo"] == t) for t in TIPOS}
    sin_clasificar = sum(1 for r in resultados if r["tipo"] is None)
    lineas = [f"{len(resultados)} objetos detectados   "
              f"(por cereza: id del csv / tipo / calibre en {unidad})"]
    for tipo, piso, techo in limites:
        calibres = [r["calibre"] * factor for r in resultados if r["tipo"] == tipo]
        detalle = (f"calibre medio {np.mean(calibres):5.1f} {unidad} "
                   f"({min(calibres):.0f}-{max(calibres):.0f})" if calibres else "")
        lineas.append(f"  tipo {tipo}: {conteo[tipo]:3d}   (V {piso:.0f}-{techo:.0f})   {detalle}")
    if sin_clasificar:
        lineas.append(f"  sin clasificar: {sin_clasificar}")

    alto_barra = 26 * len(lineas) + 20
    barra = np.zeros((alto_barra, vista.shape[1], 3), np.uint8)
    for i, linea in enumerate(lineas):
        cv2.putText(barra, linea, (14, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (tipo, _, _) in enumerate(limites):
        centro = (vista.shape[1] - 60, 34 + i * 26)
        # Aro blanco: sin él, el punto del tipo 3 (negro) se pierde en la barra.
        cv2.circle(barra, centro, 10, (255, 255, 255), -1)
        cv2.circle(barra, centro, 8, TIPOS[tipo]["color"], -1)
    return np.vstack([vista, barra])


def main():
    analizador = argparse.ArgumentParser(
        description="Cuenta cerezas por tipo de color y mide el calibre de cada una.")
    analizador.add_argument("imagen", nargs="?", default=RUTA_POR_DEFECTO)
    analizador.add_argument("--mm-por-px", type=float, default=None,
                            help="milimetros por pixel; sin esto el calibre va en px")
    analizador.add_argument("--params", default=None,
                            help="txt de parametros guardado por color_mask_interactivo.py")
    args = analizador.parse_args()

    ruta = args.imagen
    factor, unidad = (args.mm_por_px, "mm") if args.mm_por_px else (1.0, "px")

    parametros = dict(PARAMETROS_DETECCION)
    if args.params:
        parametros = cargar_parametros(args.params)
        print(f"Parametros de deteccion cargados de {args.params}")

    imagen = cv2.imread(ruta)
    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen {ruta}")
    print(f"Imagen: {ruta}  {imagen.shape[1]}x{imagen.shape[0]}")
    if unidad == "px":
        print("Sin --mm-por-px: el calibre va en pixeles (la foto no trae escala).")
    else:
        print(f"Escala: {factor} mm/px")

    limites = fronteras_de_brillo()
    print("\nRangos de brillo (V) por tipo, derivados de los centroides:")
    for tipo, piso, techo in limites:
        t = TIPOS[tipo]
        print(f"  tipo {tipo}: V de {piso:6.1f} a {techo:6.1f}   "
              f"centroide H={t['H']:.1f} S={t['S']:.1f} V={t['V']:.1f}  (n={t['n']} marcadas)")

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    contornos, piel = detectar_cerezas(imagen, parametros)

    resultados = []
    for contorno in contornos:
        color = color_promedio(hsv, piel, contorno)
        if color is None:
            continue
        momentos = cv2.moments(contorno)
        if momentos["m00"] == 0:
            continue
        resultados.append({
            "contorno": contorno,
            "x": int(momentos["m10"] / momentos["m00"]),
            "y": int(momentos["m01"] / momentos["m00"]),
            "area": float(cv2.contourArea(contorno)),
            "calibre": calibre(contorno),
            "tipo": clasificar(color, limites),
            **color,
        })

    # El id se asigna acá, y no al escribir el csv, para que la etiqueta de la
    # imagen y la fila del csv sean la misma cereza. El orden es de arriba a
    # abajo y de izquierda a derecha, que es como se recorre la foto a ojo.
    for i, r in enumerate(sorted(resultados, key=lambda r: (r["y"], r["x"])), 1):
        r["id"] = i

    conteo = {t: sum(1 for r in resultados if r["tipo"] == t) for t in TIPOS}
    sin_clasificar = [r for r in resultados if r["tipo"] is None]

    print(f"\n{len(resultados)} objetos detectados")
    print(f"\nCONTEO POR TIPO   (calibre = eje mayor de la elipse ajustada, en {unidad})")
    for tipo, _, _ in limites:
        de_este = [r for r in resultados if r["tipo"] == tipo]
        detalle = ""
        if de_este:
            v = [r["V"] for r in de_este]
            c = [r["calibre"] * factor for r in de_este]
            detalle = (f"V promedio {np.mean(v):5.1f} (min {min(v):5.1f}, max {max(v):5.1f})   "
                       f"calibre {np.mean(c):5.1f} {unidad} (min {min(c):5.1f}, max {max(c):5.1f})")
        print(f"  {TIPOS[tipo]['nombre']:22s} {conteo[tipo]:4d}   {detalle}")
    if sin_clasificar:
        print(f"  {'sin clasificar':22s} {len(sin_clasificar):4d}")
        for r in sin_clasificar[:10]:
            print(f"      id {r['id']:3d} en ({r['x']},{r['y']}) area={r['area']:.0f} "
                  f"H={r['H']:.1f} S={r['S']:.1f} V={r['V']:.1f}")
        if len(sin_clasificar) > 10:
            print(f"      ... y {len(sin_clasificar) - 10} mas (ver el csv)")
    advertir_si_muchos_sin_clasificar(resultados)

    # Cerezas a menos de 5 de V de una frontera: son las que conviene revisar a
    # ojo, porque un cambio chico de luz las pasa al tipo vecino.
    cortes = [piso for _, piso, _ in limites[:-1]]
    dudosas = [r for r in resultados if r["tipo"] and min(abs(r["V"] - c) for c in cortes) < 5]
    if dudosas:
        print(f"\n{len(dudosas)} cereza(s) a menos de 5 puntos de V de una frontera:")
        for r in sorted(dudosas, key=lambda r: r["V"]):
            print(f"      id {r['id']:3d} ({r['x']},{r['y']}) V={r['V']:.1f} -> tipo {r['tipo']}")

    os.makedirs(CARPETA_DETECCIONES, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    ruta_vista = os.path.join(CARPETA_DETECCIONES, f"clasificacion_{marca}.png")
    ruta_csv = os.path.join(CARPETA_DETECCIONES, f"clasificacion_{marca}.csv")
    cv2.imwrite(ruta_vista, anotar(imagen, resultados, limites, factor, unidad))
    cv2.imwrite(os.path.join(CARPETA_DETECCIONES, f"piel_{marca}.png"), piel)
    with open(ruta_csv, "w", encoding="utf-8") as archivo:
        archivo.write(f"id,x,y,area_px,pixeles,H,S,V,tipo,calibre_{unidad}\n")
        for r in sorted(resultados, key=lambda r: r["id"]):
            archivo.write(f"{r['id']},{r['x']},{r['y']},{r['area']:.0f},{r['n']},"
                          f"{r['H']:.1f},{r['S']:.1f},{r['V']:.1f},{r['tipo'] or ''},"
                          f"{r['calibre'] * factor:.2f}\n")
    print(f"\nGuardado:\n  {ruta_vista}\n  {ruta_csv}")


if __name__ == "__main__":
    main()
