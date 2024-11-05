import sys
import os
import time
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import document_processor as dp
import embedding
import QA_pipeline as pipeline

# Initialize components
embedding_engine = embedding.EmbeddingEngine()
qa_system = pipeline.QASystem()

# Build index
index, texts = embedding_engine.build_index('data/output')
k = 3  

# Set confidence threshold for automated scoring
CONFIDENCE_THRESHOLD = 10  

# Sample questions for testing
sample_questions = [
    "What are the main features of LangGraph?",
    "What is LangGraph?",
    "What are the key components of LangGraph's architecture?",
    "What does LangGraph do?",
    "How to install LangGraph?",
]

# Initialize results list
results = []

# Evaluation process
for question in sample_questions:
    start_time = time.time()
    
    # Embed question and retrieve top `k` relevant chunks
    query_embedding = embedding_engine.embed_chunks([question])[0].reshape(1, -1)
    D, I = index.search(query_embedding, k)
    relevant_chunks = [texts[i] for i in I[0]]
    
    # Generate answer
    result = qa_system.generate_answer(question, relevant_chunks)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Automatic scoring based on confidence threshold
    relevance_score = "High" if result['confidence'] >= CONFIDENCE_THRESHOLD else "Low"
    accuracy_status = "Likely Correct" if relevance_score == "High" else "Needs Review"
    
    # Append to results
    results.append({
        "Question": question,
        "Answer": result['answer'],
        "Confidence": result['confidence'],
        "Response Time (s)": response_time,
        "Relevance Score": relevance_score,
        "Accuracy Status": accuracy_status
    })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Calculate overall statistics
avg_confidence = df_results["Confidence"].mean()
avg_response_time = df_results["Response Time (s)"].mean()

# Display summary
print("Evaluation Summary:")
print(f"Average Confidence: {avg_confidence:.2f}")
print(f"Average Response Time: {avg_response_time:.2f} seconds")
print("\nQuestions needing review (Confidence < Threshold):")
print(df_results[df_results["Confidence"] < CONFIDENCE_THRESHOLD][["Question", "Confidence"]])

# Save to CSV
df_results.to_csv("evaluation_results.csv", index=False)
print("\nEvaluation results saved to 'evaluation_results.csv'.")
