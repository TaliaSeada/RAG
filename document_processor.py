import pdfplumber
import os

# Extract text from PDF
def extractTextFromPDF(pdf_path):
    txt = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt += page.extract_text()+"\n"
    return txt

# Save text to file
def saveTextToFile(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

# Process documents
def process_documents(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            text = extractTextFromPDF(pdf_path)
            saveTextToFile(text, output_path)
            print(f"Processed {filename} -> {output_path}")

