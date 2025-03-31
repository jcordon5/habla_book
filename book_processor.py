from PyPDF2 import PdfReader

def load_book(file):
    reader = PdfReader(file)
    pages = []
    for _, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages
