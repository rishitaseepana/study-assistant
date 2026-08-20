from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from models import Document

class DocumentLoader:

    def load(self, file_path):

        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            return self.load_pdf(file_path)

        elif suffix == ".docx":
            return self.load_docx(file_path)

        elif suffix == ".pptx":
            return self.load_ppt(file_path)

        elif suffix == ".txt":
            return self.load_txt(file_path)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def load_pdf(self, file_path):

        elements = partition_pdf(filename=file_path)
        pages = []
        current_page = []
        current_number = 1

        for element in elements:
            page = element.metadata.page_number or current_number
            if page != current_number:
                pages.append("\n".join(current_page))
                current_page = []
                current_number = page

            current_page.append(str(element))

        if current_page:
            pages.append("\n".join(current_page))

        return Document(
            filename=Path(file_path).stem,
            file_type="pdf",
            text="\n\n".join(pages),
            pages=pages
        )

    def load_docx(self, file_path):

        elements = partition_docx(filename=file_path)
        text = []

        for element in elements:
            text.append(str(element))

        return Document(
            filename=Path(file_path).stem,
            file_type="docx",
            text="\n".join(text),
            pages=["\n".join(text)]
        )

    def load_ppt(self, file_path):

        elements = partition_pptx(filename=str(file_path))
        slides = []
        current_slide = []

        current_number = 1

        for element in elements:
            page = getattr(element.metadata, "page_number", None)
            if page is None:
                page = current_number
            if page != current_number:
                slides.append("\n".join(current_slide))
                current_slide = []
                current_number = page

            current_slide.append(str(element))

        if current_slide:
            slides.append("\n".join(current_slide))

        return Document(
            filename=Path(file_path).stem,
            file_type="pptx",
            text="\n\n".join(slides),
            pages=slides
        )

    def load_txt(self, file_path):

        elements = partition_text(filename=file_path)
        text = []

        for element in elements:
            text.append(str(element))

        return Document(
            filename=Path(file_path).stem,
            file_type="txt",
            text="\n".join(text),
            pages=["\n".join(text)]
        )