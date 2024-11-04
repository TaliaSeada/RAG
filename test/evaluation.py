import sys
import os
import time
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import document_processor as dp
import embedding
import QA_pipeline as pipeline

embedding_engine = embedding.EmbeddingEngine()
qa_system = pipeline.QASystem()
index, texts = embedding_engine.build_index('data/output')
k = 3

# Sample questions for testing
sample_questions = [
    "What are the main features of LangGraph?",
    "What is LangGraph?",
    "What are the key components of LangGraph’s architecture?",
]

# Logging results
results = []

# Loop through each question and evaluate
for question in sample_questions:
    start_time = time.time()
    
    # Generate answer from the pipeline
    query_embedding = embedding_engine.embed_chunks([question])[0].reshape(1, -1)
    D, I = index.search(query_embedding, k)
    relevant_chunks = [texts[i] for i in I[0]]
    result = qa_system.generate_answer(question, relevant_chunks)
    
    end_time = time.time()
    response_time = end_time - start_time
    
    # Mock scoring for relevance and accuracy (to be done manually)
    relevance_score = None  # Placeholder for relevance score (1-5)
    accuracy_status = None  # Placeholder for accuracy status ("Correct", "Partially Correct", "Incorrect")
    
    # Store result
    results.append({
        "Question": question,
        "Answer": result['answer'],
        "Confidence": result['confidence'],
        "Response Time (s)": response_time,
        "Relevance Score": relevance_score,
        "Accuracy Status": accuracy_status
    })

# Convert results to DataFrame for easy analysis
df_results = pd.DataFrame(results)
print(df_results)

# Save the results to a CSV file
df_results.to_csv("evaluation_results.csv", index=False)

