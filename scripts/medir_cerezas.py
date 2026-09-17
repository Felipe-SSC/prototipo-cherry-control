"""Compara tres formas de estimar el calibre de cada cereza.

clasificar_cerezas.py ya entrega un calibre por cereza (el eje mayor de la
elipse ajustada). Este script es para cuando ese número no basta y hay que ver
de dónde sale: calcula las tres aproximaciones, las contrasta entre sí y dibuja
la medición encima de cada fruta. Las funciones de cálculo viven en
clasificar_cerezas.py (cz.medir), que es el módulo que las comparte.

Las tres aproximaciones del calibre:

    Eje mayor elipse   eje mayor de la elipse ajustada por mínimos cuadrados a
                       la silueta (cv2.fitEllipse). Es el calibre que entrega
                       el prototipo: al usar todos los puntos del contorno, un
                       vértice equivocado del hull lo mueve poco.
    Feret máximo       la mayor distancia entre dos puntos de la silueta (el
                       "pie de metro" abierto al máximo). Es lo más parecido a
                       cómo se mide el calibre a mano, pero al ser un máximo
                       basta un vértice espurio para estirarlo.
    Diámetro equiv.    diámetro del círculo de la misma área: sqrt(4*A/pi).
                       Promedia toda la silueta, así que es el más estable
                       frente a un borde sucio, pero subestima si la fruta no
                       es redonda.

Las tres se calculan sobre la silueta que entrega la detección. Con el relleno
en modo envolvente convexa (el de por defecto) esa silueta cierra las mordidas
que deja la segmentación en el borde, a costa de inflar un poco el calibre; al
final se imprime cuánto infla respecto de la piel sin convexificar.

Uso:
    python3 scripts/medir_cerezas.py                        # fotos/claro_centro.png
    python3 scripts/medir_cerezas.py fotos/oscuro.png
    python3 scripts/medir_cerezas.py fotos/oscuro.png --mm-por-px 0.35
    python3 scripts/medir_cerezas.py fotos/oscuro.png --params detecciones/mascara_..._parametros.txt

Sin --mm-por-px los resultados quedan en píxeles: la foto sola no tiene escala.
Para obtener mm/px se necesita una referencia de tamaño conocido en la misma
foto y al mismo plano (el tablero de calibrar.py, o una regla).
"""

import argparse
import os
import time

import cv2
import numpy as np

import clasificar_cerezas as cz

CARPETA_DETECCIONES = cz.CARPETA_DETECCIONES

# Las tres métricas, con el calibre del prototipo primero.
METRICAS = (("eje_mayor", "Eje mayor elip"),
            ("feret_max", "Feret maximo  "),
            ("diam_equiv", "Diam. equival."))


# una línea de estadísticas de una métrica, ya convertida a la unidad pedida
def resumir(valores, factor, unidad):
    if not valores:
        return "sin datos"
    v = np.array(valores) * factor
    return (f"media {v.mean():6.2f} {unidad} | sd {v.std():5.2f} | "
            f"min {v.min():6.2f} | max {v.max():6.2f}")


# dibuja sobre cada fruta la elipse, el segmento del Feret y el calibre
def anotar(imagen, resultados, factor, unidad):
    vista = imagen.copy()
    for r in resultados:
        color = cz.TIPOS[r["tipo"]]["color"] if r["tipo"] else cz.COLOR_SIN_CLASIFICAR
        if r["elipse"] is not None:
            cv2.ellipse(vista, r["elipse"], color, 2)
        a, b = r["extremos"]
        if a is not None:
            # el segmento del Feret queda al lado de la elipse para ver de un
            # vistazo cuándo se dispara por un vértice del hull
            cv2.line(vista, tuple(a), tuple(b), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(vista, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        # se etiqueta el eje mayor, no el Feret: ese es el calibre que entrega el prototipo
        etiqueta = f"{r['eje_mayor'] * factor:.0f}"
        posicion = (r["x"] - 18, r["y"] + 6)
        cv2.putText(vista, etiqueta, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vista, etiqueta, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cerezas = [r for r in resultados if r["tipo"] is not None]
    lineas = [f"{len(resultados)} objetos, {len(cerezas)} cerezas | "
              f"etiqueta = eje mayor de la elipse ({unidad})"]
    for tipo in sorted(cz.TIPOS):
        de_este = [r for r in resultados if r["tipo"] == tipo]
        if de_este:
            lineas.append(f"  tipo {tipo}: {len(de_este):3d}   Eje mayor "
                          + resumir([r["eje_mayor"] for r in de_este], factor, unidad))
    sin_tipo = sum(1 for r in resultados if r["tipo"] is None)
    if sin_tipo:
        lineas.append(f"  sin clasificar: {sin_tipo}")

    barra = np.zeros((26 * len(lineas) + 20, vista.shape[1], 3), np.uint8)
    for i, linea in enumerate(lineas):
        cv2.putText(barra, linea, (14, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([vista, barra])


def main():
    analizador = argparse.ArgumentParser(description="Calibre y tipo de color de cada cereza.")
    analizador.add_argument("imagen", nargs="?", default=cz.RUTA_POR_DEFECTO)
    analizador.add_argument("--mm-por-px", type=float, default=None,
                            help="milimetros por pixel; sin esto los calibres van en px")
    analizador.add_argument("--params", default=None,
                            help="txt de parametros guardado por color_mask_interactivo.py")
    args = analizador.parse_args()

    ruta = args.imagen
    factor, unidad = (args.mm_por_px, "mm") if args.mm_por_px else (1.0, "px")

    parametros = dict(cz.PARAMETROS_DETECCION)
    if args.params:
        parametros = cz.cargar_parametros(args.params)
        print(f"Parametros de deteccion cargados de {args.params}")

    imagen = cv2.imread(ruta)
    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen {ruta}")
    print(f"Imagen: {ruta}  {imagen.shape[1]}x{imagen.shape[0]}")
    if unidad == "px":
        print("Sin --mm-por-px: los calibres van en pixeles (la foto no trae escala).")
    else:
        print(f"Escala: {factor} mm/px")

    limites = cz.fronteras_de_brillo()
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    contornos, piel = cz.detectar_cerezas(imagen, parametros)

    resultados = []
    for contorno in contornos:
        color = cz.color_promedio(hsv, piel, contorno)
        momentos = cv2.moments(contorno)
        if color is None or momentos["m00"] == 0:
            continue
        # Silueta de la piel sin convexificar, para saber cuánto infla el hull.
        recorte = np.zeros(piel.shape, np.uint8)
        cv2.drawContours(recorte, [contorno], -1, 255, cv2.FILLED)
        area_piel = float(np.count_nonzero((recorte > 0) & (piel > 0)))

        resultados.append({
            "contorno": contorno,
            "x": int(momentos["m10"] / momentos["m00"]),
            "y": int(momentos["m01"] / momentos["m00"]),
            "area": float(cv2.contourArea(contorno)),
            "area_piel": area_piel,
            "tipo": cz.clasificar(color, limites),
            **color,
            **cz.medir(contorno),
        })

    cerezas = [r for r in resultados if r["tipo"] is not None]
    print(f"\n{len(resultados)} objetos detectados, {len(cerezas)} clasificados como cereza")
    cz.advertir_si_muchos_sin_clasificar(resultados)

    print(f"\nCALIBRE POR TIPO ({unidad})")
    for tipo, _, _ in limites:
        de_este = [r for r in cerezas if r["tipo"] == tipo]
        print(f"\n  {cz.TIPOS[tipo]['nombre']}  (n={len(de_este)})")
        if not de_este:
            continue
        for clave, etiqueta in METRICAS:
            print(f"    {etiqueta}: {resumir([r[clave] for r in de_este], factor, unidad)}")

    print(f"\nTODAS ({len(cerezas)} cerezas, {unidad})")
    for clave, etiqueta in METRICAS:
        print(f"  {etiqueta}: {resumir([r[clave] for r in cerezas], factor, unidad)}")

    if cerezas:
        # Cuánto se separan las métricas entre sí: si el Feret supera mucho al
        # diámetro equivalente, la silueta no es redonda (cerezas pegadas, un
        # pedicelo agarrado por la máscara, una sombra incluida).
        razon = np.array([r["feret_max"] / r["diam_equiv"] for r in cerezas if r["diam_equiv"] > 0])
        print(f"\n  Feret/Diam.equiv: media {razon.mean():.3f} (1,000 = circulo perfecto), "
              f"max {razon.max():.3f}")
        # Cuánto se ahorra el calibre al ajustar la elipse en vez de tomar el
        # máximo: si el max es alto, hay siluetas donde el Feret se iba con un
        # vértice del hull y la elipse no lo siguió.
        vs_eje = np.array([r["feret_max"] / r["eje_mayor"] for r in cerezas if r["eje_mayor"] > 0])
        print(f"  Feret/Eje mayor : media {vs_eje.mean():.3f}, max {vs_eje.max():.3f} "
              f"(el Feret mide {100 * (vs_eje.mean() - 1):.1f}% mas que el calibre en promedio)")
        infla = np.array([np.sqrt(r["area"] / r["area_piel"]) for r in cerezas if r["area_piel"] > 0])
        print(f"  La envolvente convexa infla el diametro un {100 * (infla.mean() - 1):.1f}% "
              f"en promedio respecto de la piel segmentada (max {100 * (infla.max() - 1):.1f}%)")
        peores = sorted(cerezas, key=lambda r: -r["feret_max"] / max(r["diam_equiv"], 1e-9))[:3]
        print("  Siluetas menos circulares (revisar a ojo):")
        for r in peores:
            print(f"    ({r['x']},{r['y']}) Feret={r['feret_max'] * factor:.1f} "
                  f"equiv={r['diam_equiv'] * factor:.1f} razon={r['feret_max'] / r['diam_equiv']:.3f}")

    os.makedirs(CARPETA_DETECCIONES, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    ruta_vista = os.path.join(CARPETA_DETECCIONES, f"calibre_{marca}.png")
    ruta_csv = os.path.join(CARPETA_DETECCIONES, f"calibre_{marca}.csv")
    cv2.imwrite(ruta_vista, anotar(imagen, resultados, factor, unidad))
    with open(ruta_csv, "w", encoding="utf-8") as archivo:
        archivo.write(f"id,x,y,tipo,H,S,V,area_px,feret_max_{unidad},"
                      f"diam_equiv_{unidad},eje_mayor_{unidad}\n")
        for i, r in enumerate(sorted(resultados, key=lambda r: (r["y"], r["x"])), 1):
            archivo.write(f"{i},{r['x']},{r['y']},{r['tipo'] or ''},"
                          f"{r['H']:.1f},{r['S']:.1f},{r['V']:.1f},{r['area']:.0f},"
                          f"{r['feret_max'] * factor:.2f},{r['diam_equiv'] * factor:.2f},"
                          f"{r['eje_mayor'] * factor:.2f}\n")
    print(f"\nGuardado:\n  {ruta_vista}\n  {ruta_csv}")


if __name__ == "__main__":
    main()
