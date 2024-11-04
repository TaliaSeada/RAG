import document_processor as dp
import embedding
import QA_pipeline as pipeline

def main():
    # Convert URLs to PDFs and extract text from PDFs to TXT files
    # dp.url_to_pdf()
    # dp.process_documents('data/input', 'data/output')

    try:
        # Initialize components
        embedding_engine = embedding.EmbeddingEngine()
        qa_system = pipeline.QASystem()
        
        # Process documents and build index
        index, texts = embedding_engine.build_index('data/output')
        
        while(True):
            # Question to ask
            question = input("Ask any question about LangGraph (or exit): ")
            if question == 'exit':
                break

            # Retrieve relevant documents
            query_embedding = embedding_engine.embed_chunks([question])[0].reshape(1, -1)
            k = 3  
            D, I = index.search(query_embedding, k)
            relevant_chunks = [texts[i] for i in I[0]]
            
            # Generate answer
            print("Generating answer...")
            result = qa_system.generate_answer(question, relevant_chunks)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence Score: {result['confidence']}")

        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
