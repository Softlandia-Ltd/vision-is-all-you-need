import pypdfium2 as pdfium
import numpy as np


def images_from_pdf_bytes(pdf_bytes: bytes) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    return images_from_document(pdf)


def images_from_pdf_path(pdf_path: str) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(pdf_path)
    return images_from_document(pdf)


def images_from_document(document: pdfium.PdfDocument) -> list[np.ndarray]:
    images = []

    for i in range(len(document)):
        page = document[i]
        img = page.render(scale=2, rev_byteorder=True).to_numpy()
        images.append(img)

    return images
