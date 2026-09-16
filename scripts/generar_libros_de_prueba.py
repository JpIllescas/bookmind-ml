"""Genera un PDF grande y un EPUB mínimo para probar el visor."""
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Uso: python scripts/generar_libros_de_prueba.py ../pdfs-de-prueba/libro.pdf ../pdfs-de-prueba/libro.epub
destino_pdf, destino_epub = sys.argv[1], sys.argv[2]

TEXTO = (
    "La celula es la unidad basica de los seres vivos. Las plantas realizan la "
    "fotosintesis para transformar la energia del sol en alimento. El experimento "
    "nos permite comprobar si nuestra hipotesis sobre la materia es correcta."
)

with PdfPages(destino_pdf) as pdf:
    for numero in range(1, 261):
        figura = plt.figure(figsize=(8.5, 11))
        figura.text(0.1, 0.9, f"Capitulo {numero // 10 + 1} - Pagina {numero}", size=16)
        for linea in range(8):
            figura.text(0.1, 0.82 - linea * 0.05, TEXTO[:90], size=9)
        # Una figura cada diez paginas: asi el PDF pesa y trae imagenes reales.
        if numero % 10 == 0:
            ejes = figura.add_axes([0.15, 0.15, 0.7, 0.4])
            ejes.plot(np.linspace(0, 10, 200), np.sin(np.linspace(0, 10, 200) * numero))
            ejes.set_title(f"Grafica {numero}")
        pdf.savefig(figura)
        plt.close(figura)

CAPITULOS = [
    (
        f"cap{numero}.xhtml",
        f"<?xml version='1.0' encoding='utf-8'?><html xmlns='http://www.w3.org/1999/xhtml'>"
        f"<head><title>Capitulo {numero}</title></head><body><h1>Capitulo {numero}</h1>"
        + "".join(f"<p>{TEXTO}</p>" for _ in range(6))
        + "</body></html>",
    )
    for numero in range(1, 9)
]

manifest = "".join(
    f"<item id='c{i}' href='{nombre}' media-type='application/xhtml+xml'/>"
    for i, (nombre, _) in enumerate(CAPITULOS)
)
spine = "".join(f"<itemref idref='c{i}'/>" for i in range(len(CAPITULOS)))

with zipfile.ZipFile(destino_epub, "w") as epub:
    epub.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
    epub.writestr(
        "META-INF/container.xml",
        "<?xml version='1.0'?><container version='1.0' "
        "xmlns='urn:oasis:names:tc:opendocument:xmlns:container'><rootfiles>"
        "<rootfile full-path='OEBPS/libro.opf' media-type='application/oebps-package+xml'/>"
        "</rootfiles></container>",
    )
    epub.writestr(
        "OEBPS/libro.opf",
        "<?xml version='1.0'?><package xmlns='http://www.idpf.org/2007/opf' version='3.0' "
        "unique-identifier='id'><metadata xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        "<dc:title>Libro de prueba</dc:title><dc:identifier id='id'>prueba</dc:identifier>"
        f"</metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>",
    )
    for nombre, contenido in CAPITULOS:
        epub.writestr(f"OEBPS/{nombre}", contenido)

print("listo")
