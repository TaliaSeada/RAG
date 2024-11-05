import pdfplumber
import os
import pdfkit
import re

def download_page_as_pdf(url, output_path):
    """
    Download page from link and convert it to pdf

    url: link to the internet page
    output_path: path to the place to save the result pdf 
    """
    try:
        # if wkhtmltopdf is not in PATH use this (change the path to the releant path):
        config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")

        # Generate PDF from the provided URL
        pdfkit.from_url(url, output_path, configuration=config)  

        print(f"PDF saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def url_to_pdf():
    """
    Converts url into pdf file 
    """
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

def extractTextFromPDF(pdf_path):
    """
    Extract text from PDF using pdfplumber

    pdf_path: path to the input pdf 
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def saveTextToFile(text, output_path):
    """
    Save text to file

    text: text we wan to save into the file
    output_path: path to the file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def clean_text(text):
    """
    Clean the text in order to get clean text for better results when searching

    text: text we want to clean

    return: cleaned text
    """
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

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into smaller chunks with overlap, using a memory-efficient approach.
    Much smaller default chunk size to prevent memory issues.

    text: the text to split into chunks
    chunk_size: the desired size of each chunk
    overlap: number of overlapping characters between chunks

    Yields: Each chunk of text.
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
        if chunk:  
            chunks.append(chunk)
            
        start = end

        # Process in batches of 1000 chunks
        if len(chunks) >= 1000:  
            for c in chunks:
                yield c
            # Clear the chunks list
            chunks = []  
            
    # Yield any remaining chunks
    for c in chunks:
        yield c

def process_documents(input_folder, output_folder):
    """
    Process pdf documents from given folder 

    input_folder: path to the folder we want to take the documents from
    output_folder: path to the folder we want to save to results into
    """
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

