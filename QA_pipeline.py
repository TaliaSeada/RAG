from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import List, Dict

class QASystem:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
    def generate_answer(self, question: str, contexts: List[str], max_length: int = 512) -> Dict[str, str]:
        best_answer = ""
        best_score = float('-inf')
        
        for context in contexts:
            # Tokenize input
            inputs = self.tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
            
            # Find the best answer span
            start_scores = start_scores.squeeze()
            end_scores = end_scores.squeeze()
            
            max_start = torch.argmax(start_scores)
            max_end = torch.argmax(end_scores)
            
            if max_end >= max_start:
                answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        inputs["input_ids"][0][max_start:max_end + 1]
                    )
                )
                
                score = torch.max(start_scores) + torch.max(end_scores)
                if score > best_score:
                    best_score = score
                    best_answer = answer
        
        return {
            "answer": best_answer,
            "confidence": float(best_score)
        }