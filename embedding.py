from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from typing import List, Tuple
import document_processor as dp

class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        
    def embed_chunks(self, chunks, batch_size=32):
        """
        Embed chunks of text in batches to manage memory

        chunks: chunks to embed
        batch_size: size of a batch

        return: array of result embeddings
        """
        embeddings = []
        current_batch = []
        
        for chunk in chunks:
            current_batch.append(chunk)
            
            if len(current_batch) >= batch_size:
                # Embed current batch
                batch_embeddings = self.model.encode(current_batch, convert_to_tensor=False)
                embeddings.extend(batch_embeddings)
                # Clear the batch
                current_batch = []  
                
        # Handle any remaining chunks
        if current_batch:
            batch_embeddings = self.model.encode(current_batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
    
    def build_index(self, input_folder):
        """
        Build index with memory-efficient processing

        input_folder: path to the folder we want to take the documents from

        rteurn: 
            index: index we built
            text: text we read from the documents
        """
        embeddings_list = []
        self.texts = []
        total_size = 0
        
        # Process each file one at a time
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                # print(f"Processing {filename}...")
                file_path = os.path.join(input_folder, filename)
                
                # Read and process file in chunks
                with open(file_path, "r", encoding="utf-8") as f:
                    current_chunks = []
                    
                    # Read file in chunks
                    for chunk in dp.chunk_text(f.read()):
                        current_chunks.append(chunk)
                        total_size += len(chunk)
                        
                        # Process when we have enough chunks or memory usage is high
                        if len(current_chunks) >= 100: 
                            chunk_embeddings = self.embed_chunks(current_chunks)
                            embeddings_list.append(chunk_embeddings)
                            self.texts.extend(current_chunks)
                            current_chunks = []  
                            
                    # Process any remaining chunks
                    if current_chunks:
                        chunk_embeddings = self.embed_chunks(current_chunks)
                        embeddings_list.append(chunk_embeddings)
                        self.texts.extend(current_chunks)
        
        # print(f"Total text size processed: {total_size} characters")
        # print(f"Number of chunks: {len(self.texts)}")
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([])
        
        # Build the FAISS index
        if len(all_embeddings) > 0:
            dimension = all_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(all_embeddings)
            # print(f"Index built with {len(self.texts)} chunks")
            return self.index, self.texts
        else:
            raise ValueError("No text was processed successfully")
        


