import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class ProgrammingAI(nn.Module):
    def __init__(self, model_name="microsoft/CodeGPT-small-py"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
