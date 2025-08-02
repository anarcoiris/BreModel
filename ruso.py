from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Define la ruta a la fuente (usa raw string para evitar escapes)
font_path = r"C:\Users\soyko\Downloads\PriceAction\DejaVuSans.ttf"

# Datos de ejemplo (puedes reemplazar con tus listas)
sustantivos = [
    "слово – palabra", "человек – persona", "время – tiempo", "год – año",
    "день – día", "рука – mano", "работа – trabajo", "город – ciudad",
    "женщина – mujer", "ребёнок – niño", "дом – casa", "окно – ventana",
    "стол – mesa", "стул – silla", "дорога – camino", "машина – coche",
    "лес – bosque", "река – río", "гора – montaña", "солнце – sol",
    # ... añade más hasta 100
]

# 40 verbos rusos - español
verbos = [
    "говорить – hablar", "быть – ser, estar", "делать – hacer", "идти – ir",
    "знать – saber", "хотеть – querer", "видеть – ver", "думать – pensar",
    "работать – trabajar", "жить – vivir", "слушать – escuchar", "читать – leer",
    "писать – escribir", "любить – amar", "спать – dormir", "бегать – correr",
    "покупать – comprar", "продавать – vender", "играть – jugar", "учить – aprender",
    "пить – beber", "есть – comer", "плавать – nadar", "ждать – esperar",
    "помогать – ayudar", "смотреть – mirar", "стоять – estar de pie", "сидеть – sentarse",
    "летать – volar", "плакать – llorar", "смеяться – reír", "готовить – cocinar",
    "говорить – hablar", "читать – leer", "писать – escribir", "брать – tomar",
    "давать – dar", "звонить – llamar", "спрашивать – preguntar", "отвечать – responder",
]

# 40 adjetivos rusos - español
adjetivos = [
    "красивый – bonito", "большой – grande", "маленький – pequeño", "новый – nuevo",
    "старый – viejo", "хороший – bueno", "плохой – malo", "быстрый – rápido",
    "медленный – lento", "тёплый – cálido", "холодный – frío", "сильный – fuerte",
    "слабый – débil", "высокий – alto", "низкий – bajo", "толстый – gordo",
    "тонкий – delgado", "счастливый – feliz", "грустный – triste", "умный – inteligente",
    "глупый – tonto", "интересный – interesante", "скучный – aburrido", "чистый – limpio",
    "грязный – sucio", "дорогой – caro", "дешёвый – barato", "свежий – fresco",
    "старый – antiguo", "тяжёлый – pesado", "лёгкий – ligero", "светлый – claro",
    "тёмный – oscuro", "добрый – amable", "злой – malo", "тихий – silencioso",
    "громкий – ruidoso", "молодой – joven", "толстый – grueso", "узкий – estrecho",
]

# 20 adverbios rusos - español
adverbios = [
    "быстро – rápidamente", "медленно – lentamente", "всегда – siempre",
    "иногда – a veces", "здесь – aquí", "там – allí", "сейчас – ahora",
    "вчера – ayer", "сегодня – hoy", "завтра – mañana", "очень – muy",
    "почти – casi", "только – sólo", "уже – ya", "еще – todavía",
    "далеко – lejos", "рядом – cerca", "возможно – posiblemente",
    "никогда – nunca", "совсем – completamente",
]

# 20 frases comunes rusas - español
frases = [
    "Как тебя зовут? – ¿Cómo te llamas?",
    "Меня зовут ... – Me llamo ...",
    "Привет! – ¡Hola!",
    "Доброе утро – Buenos días",
    "Добрый вечер – Buenas tardes/noches",
    "До свидания – Adiós",
    "Спасибо – Gracias",
    "Пожалуйста – Por favor / De nada",
    "Я не понимаю – No entiendo",
    "Где туалет? – ¿Dónde está el baño?",
    "Сколько это стоит? – ¿Cuánto cuesta?",
    "Я люблю тебя – Te amo",
    "Я говорю по-испански – Hablo español",
    "Я не говорю по-русски – No hablo ruso",
    "Помогите! – ¡Ayuda!",
    "Извините – Perdón / Disculpe",
    "Как пройти к ...? – ¿Cómo llegar a ...?",
    "Это очень красиво – Es muy bonito",
    "Мне это нравится – Me gusta esto",
    "До встречи! – ¡Hasta luego!",
]

class PDFUnicode(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Vocabulario esencial de ruso - Español", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
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

# Luego registra fuente sin el parámetro uni:
pdf = PDFUnicode()
pdf.add_font("DejaVu", "", font_path)
pdf.add_font("DejaVu", "B", font_path)
pdf.add_page()

pdf.section_title("100 Sustantivos comunes")
pdf.section_body(sustantivos)

pdf.section_title("40 Verbos frecuentes")
pdf.section_body(verbos)

pdf.section_title("40 Adjetivos esenciales")
pdf.section_body(adjetivos)

pdf.section_title("20 Adverbios útiles")
pdf.section_body(adverbios)

pdf.section_title("Frases de uso común")
pdf.section_body(frases)

pdf.output("Vocabulario_ruso_espanol_unicode.pdf")
print("PDF generado: Vocabulario_ruso_espanol_unicode.pdf")
