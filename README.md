# RAG
Document-based Conversational AI System for LangGraph Information Retrieval

## Overview
This task aims to create a conversational AI system capable of answering questions about LangGraph by retrieving and synthesizing information from provided sources. Using the provided articles and repositories, I built a Retrieval-Augmented Generation (RAG) pipeline that retrieves information from a knowledge base and answers questions with a conversational interface.

Data Sources:
> GitHub Repository: [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)</br>
> Galileo Blog: [Mastering Agents with LangGraph, AutoGen, and CREW](https://galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew)</br>
> LinkedIn Article: [Detailed Technical Exploration of LangGraph](https://www.linkedin.com/pulse/langgraph-detailed-technical-exploration-ai-workflow-jagadeesan-n9woc/)</br>
> Towards Data Science Article: [LangGraph from Basics to Advanced](https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787)</br>
> Medium Article: [Why LangGraph Stands Out as an Agent Framework](https://medium.com/@hao.l/why-langgraph-stands-out-as-an-exceptional-agent-framework-44806d969cc6)</br>
> Towards AI Article: [AI Project Management with LangGraph](https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1)</br>

## Documentation

<details> 
  <summary>Document Processing and Text Extraction</summary>
  
  ### Overview
  This phase downloads, extracts, cleans, and chunks LangGraph-related documents for efficient retrieval and processing in the RAG pipeline.

* Downloading PDFs: download_page_as_pdf uses pdfkit to convert specified URLs to PDFs, which are saved for offline access. A configured path to wkhtmltopdf is required for pdfkit to function.

* Text Extraction: extractTextFromPDF uses pdfplumber to extract text from each PDF, making the content accessible for processing.

* Cleaning Text: clean_text removes extraneous elements like UI components, URLs, file paths, and whitespace to yield a streamlined version of the document.

* Chunking: chunk_text splits the cleaned text into manageable, overlapping sections, optimizing memory and ensuring coherence within each chunk.

* Saving Text: The processed text is saved as .txt files, organized in the output folder for easy access in the pipeline.

### Tools & Challenges
* Tools: pdfkit, pdfplumber, re (for regex-based cleaning), and os (for file management).
* Challenges: Configuring wkhtmltopdf, designing regex patterns for precise text cleaning, and managing memory in chunking for large documents.
</details>


<details> 
  <summary>Embedding and Vector Storage Setup</summary>
  
  ### Overview
  This phase converts document text into embeddings and stores them in a FAISS index for fast, semantic search.

* Embedding Engine: The EmbeddingEngine class uses SentenceTransformer to generate embeddings for text chunks, storing them efficiently in memory.

* Chunk Embedding: embed_chunks processes text in manageable batches, minimizing memory load by embedding in batches of 32.

* Index Building: build_index iterates through text files, using chunk_text to split documents, then embeds each chunk. All embeddings are stored in a FAISS index (IndexFlatL2) for vector similarity search.

* Outputs: The FAISS index and the list of processed text chunks are returned for use in the RAG pipeline.

### Tools & Challenges
* Tools: SentenceTransformer, FAISS, numpy, and custom document_processor module.
* Challenges: Balancing memory usage when embedding large text files, and managing batch processing efficiently.

</details>


<details> 
  <summary>Question-Answering Pipeline Setup</summary>

  ### Overview
  This stage uses a question-answering (QA) model to generate answers based on retrieved context, enhancing the RAG pipeline with precise responses.

* QA System: The QASystem class employs a pre-trained deepset/roberta-base-squad2 model to answer questions by finding the most relevant text spans within context.

* Answer Generation: generate_answer iterates over context passages, tokenizes inputs with a 512-token limit, and scores possible answer spans. The highest-confidence answer is selected and returned.

* Outputs: Returns the answer with a confidence score to indicate response reliability.

### Tools & Challenges
* Tools: AutoTokenizer, AutoModelForQuestionAnswering from Hugging Face, and torch.
* Challenges: Optimizing tokenization within the max length limit to avoid truncation and accurately identifying answer spans.
</details>


<details> 
  <summary>User Interface (UI) </summary>

  ### Overview
  This phase retrieves relevant documents and presents answers through a user-friendly interface.

* Retrieval: The main() function uses the EmbeddingEngine class to create document embeddings and build an index for efficient retrieval. When a query is entered, the most relevant chunks are retrieved from this index, based on similarity to the question.

* UI: Streamlit provides an interactive and accessible interface, configured with:

  * Document Processing: The sidebar enables document processing via buttons that convert URLs to PDFs and prepare the files for retrieval.
  * Question-Answer Interface: A main input box allows users to ask questions, retrieve answers, and view confidence scores and relevant documents.
### Tools & Challenges
* Tools: streamlit, faiss, document_processor for data management and retrieval.
* Challenges: Efficient retrieval of relevant text chunks and maintaining real-time response generation in a user-friendly UI.
</details>

## How to run
Make sure to install:</br>
`pip install pdfplumber transformers faiss-cpu openai pinecone-client streamlit gradio pdfkit sentence-transformers`</br>
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
1. Open new terminal
2. Navigate to the project's location
3. run `streamlit run main.py`


## Python version
`Python 3.11.0` 



Links:
* https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
* https://www.pinecone.io/learn/series/faiss/faiss-tutorial/
* https://medium.com/@ypredofficial/faiss-vector-database-be3a9725172f
* https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad
* https://wkhtmltopdf.org/downloads.html


