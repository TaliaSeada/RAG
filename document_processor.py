import pdfplumber
import os
import pdfkit
import re

def download_page_as_pdf(url, output_path):
    try:
        config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")

        # Generate PDF from the provided URL
        pdfkit.from_url(url, output_path, configuration=config)  

        print(f"PDF saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def url_to_pdf():
    links = ['https://github.com/langchain-ai/langgraph', 
             'https://galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew', 
             'https://www.linkedin.com/pulse/langgraph-detailed-technical-exploration-ai-workflow-jagadeesan-n9woc/',
             'https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787',
             'https://medium.com/@hao.l/why-langgraph-stands-out-as-an-exceptional-agent-framework-44806d969cc6',
             'https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1']
    names = ['GitHub Repository', 'Galileo Blog', 'LinkedIn Article', 'Towards Data Science Article', 'Medium Article', 'Towards AI Article']
    
    for i in range(len(links)):
        url = links[i]
        output_path = 'data/input/' + names[i] + '.pdf'
        download_page_as_pdf(url, output_path)

# Extract text from PDF
def extractTextFromPDF(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Save text to file
def saveTextToFile(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text):
    # Remove common GitHub UI elements
    text = re.sub(r"Star|Fork|Issues|Pull requests|Discussions|Actions|Projects|Security|Insights", "", text)
    # Remove statistics
    text = re.sub(r"\d+(\.\d+)?[km]? (stars|forks|issues|downloads)", "", text)
    # Remove navigation elements
    text = re.sub(r"Code|Branch|Tags|Activity|Notifications", "", text)
    # Remove URLs
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
    # Remove file paths
    text = re.sub(r"[a-zA-Z]:\\[^\s]+", "", text)
    # Remove extra whitespace and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    # Remove GitHub footer
    text = re.sub(r"© \d{4} GitHub, Inc\..*", "", text)
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into smaller chunks with overlap, using a memory-efficient approach.
    Much smaller default chunk size to prevent memory issues.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If this is not the first chunk, include the overlap
        if start > 0:
            start = max(start - overlap, 0)
            
        # Get the chunk and clean it
        chunk = text[start:end].strip()
        if chunk:  # Only append non-empty chunks
            chunks.append(chunk)
            
        start = end
        
        # Memory management: yield chunks periodically
        if len(chunks) >= 1000:  # Process in batches of 1000 chunks
            for c in chunks:
                yield c
            chunks = []  # Clear the chunks list
            
    # Yield any remaining chunks
    for c in chunks:
        yield c

def process_documents(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            text = extractTextFromPDF(pdf_path)
            cleaned_text = clean_text(text)  
            saveTextToFile(cleaned_text, output_path)
            print(f"Processed {filename} -> {output_path}")

