import document_processor as dp
import embedding
import QA_pipeline as pipeline
import streamlit as st


def main():
    # Configure the Streamlit page
    st.set_page_config(
        page_title="LangGraph Q&A Assistant",
        page_icon="🔍",
        layout="wide"  # Changed to "wide" to maximize use of screen space
    )

    # Title and instructions
    st.title("LangGraph Q&A Assistant")
    st.markdown("Ask questions about LangGraph and receive detailed answers. Use the sidebar to process documents.")

    # Sidebar for document processing options and settings
    with st.sidebar:
        st.header("Document Processing")
        if st.button("Process Documents"):
            # Convert URLs to PDFs and extract text from PDFs to TXT files
            dp.url_to_pdf()
            dp.process_documents('data/input', 'data/output')
            st.success("Documents processed successfully!")
        
        # User can select the number of documents to retrieve
        k = st.slider("Number of documents to retrieve:", min_value=1, max_value=10, value=3)

    try:
        # Initialize components
        embedding_engine = embedding.EmbeddingEngine()
        qa_system = pipeline.QASystem()

        # Process documents and build index if not already done
        index, texts = embedding_engine.build_index('data/output')

        # User input for query
        user_query = st.text_input("Ask any question about LangGraph:")

        if st.button("Get Answer"):
            if user_query:
                # Display progress
                with st.spinner("Generating answer..."):
                    # Retrieve relevant documents
                    query_embedding = embedding_engine.embed_chunks([user_query])[0].reshape(1, -1)
                    D, I = index.search(query_embedding, k)
                    relevant_chunks = [texts[i] for i in I[0]]

                    # Generate answer
                    result = qa_system.generate_answer(user_query, relevant_chunks)

                # Display answer and confidence score 
                st.markdown("### Answer")
                st.write(result['answer'])
                
                # Use expander to show additional information
                with st.expander("Show Details"):
                    st.write("**Confidence Score:**", result['confidence'])
                    st.write("**Relevant Documents:**")
                    for chunk in relevant_chunks:
                        st.markdown(f"- {chunk}")

            else:
                st.warning("Please enter a question.")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
