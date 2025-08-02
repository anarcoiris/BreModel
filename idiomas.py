import json
import os
import argparse
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PDFUnicode(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, self.header_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(10)

    def section_title(self, title):
        self.set_font("DejaVu", "B", 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def section_body(self, items):
        self.set_font("DejaVu", "", 11)
        for item in items:
            self.cell(0, 6, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

def generar_pdf_desde_json(ruta_json, ruta_fuente):
    base_nombre = os.path.splitext(os.path.basename(ruta_json))[0]

    with open(ruta_json, encoding="utf-8") as f:
        vocab = json.load(f)

    pdf = PDFUnicode()
    pdf.add_font("DejaVu", "", ruta_fuente)
    pdf.add_font("DejaVu", "B", ruta_fuente)
    pdf.header_title = f"Vocabulario esencial de {base_nombre} - Español"
    pdf.add_page()

    if vocab.get("sustantivos"):
        pdf.section_title("Sustantivos comunes")
        pdf.section_body(vocab["sustantivos"])

    if vocab.get("verbos"):
        pdf.section_title("Verbos frecuentes")
        pdf.section_body(vocab["verbos"])

    if vocab.get("adjetivos"):
        pdf.section_title("Adjetivos esenciales")
        pdf.section_body(vocab["adjetivos"])

    if vocab.get("adverbios"):
        pdf.section_title("Adverbios útiles")
        pdf.section_body(vocab["adverbios"])

    if vocab.get("frases"):
        pdf.section_title("Frases de uso común")
        pdf.section_body(vocab["frases"])

    nombre_pdf = f"Vocabulario esencial de {base_nombre} - Español.pdf"
    pdf.output(nombre_pdf)
    print(f"PDF generado: {nombre_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de PDF de vocabulario ruso-español desde JSON.")
    parser.add_argument("json_file", help="Ruta al archivo JSON con vocabulario.")
    parser.add_argument("--font", default=r"C:\Users\soyko\Downloads\PriceAction\DejaVuSans.ttf", help="Ruta a la fuente .ttf (por defecto DejaVuSans).")
    args = parser.parse_args()

    generar_pdf_desde_json(args.json_file, args.font)
