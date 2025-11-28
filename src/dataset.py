import json
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(jsonl_file, 'r') as f:
            self.examples = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example['prompt']
        completion = example['completion']
        
        text = f"{prompt}\n{completion}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }
