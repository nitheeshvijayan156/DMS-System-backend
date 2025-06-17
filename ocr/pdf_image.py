import pypdfium2 as pdfium
from PIL import Image
from io import BytesIO

def convert_pdf_to_images(file_path):
    scale = 300/72
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    list_final_images = []

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        list_final_images.append(image_byte_array)
    
    return list_final_images
