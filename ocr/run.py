from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
from ocr.pdf_image import convert_pdf_to_images
import pypandoc

def convert_word_to_pdf(word_file, output_pdf='output.pdf'):
    pypandoc.convert_file(word_file, 'pdf', outputfile=output_pdf)
    return output_pdf

def extract_text_with_pytesseract(images):
    image_content = []
    for image_bytes in images:
        image = Image.open(BytesIO(image_bytes))
        raw_text = image_to_string(image)
        image_content.append(raw_text)
    return "\n".join(image_content)

def handle_docx(file):
    # Save the uploaded docx file to a temporary location to convert it
    with open("/tmp/temp_file.docx", "wb") as temp_file:
        temp_file.write(file)

    pdf_file = convert_word_to_pdf("/tmp/temp_file.docx")
    images = convert_pdf_to_images(pdf_file)
    return extract_text_with_pytesseract(images)

def handle_pdf(file):
    images = convert_pdf_to_images(BytesIO(file))
    return extract_text_with_pytesseract(images)

def handle_image(file):
    image = Image.open(BytesIO(file))
    raw_text = image_to_string(image)
    return raw_text

def process_file(file, content_type):

    if content_type == 'application/pdf':
        return handle_pdf(file)
    elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
        return handle_docx(file)
    elif content_type in ['image/jpeg', 'image/png']:
        return handle_image(file)
    else:
        raise ValueError(f"Unsupported file type: {content_type}")

