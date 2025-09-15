import fitz  # PyMuPDF
import os

def extract_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

if __name__ == "__main__":
    # Path to your PDF in the data folder
    pdf_file = os.path.join("..", "data", "policy_pdfs", "HRPolicy.pdf")
    raw_text = extract_pdf_text(pdf_file)

    # Basic cleaning: remove empty lines and redundant spaces
    cleaned_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)

    # Save cleaned text for further processing
    output_path = os.path.join("..", "data", "hr_policy_clean.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Extraction & cleaning complete. Output at: {output_path}")
