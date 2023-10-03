from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import PeftModel, PeftConfig


from helpers import print_number_of_trainable_model_parameters, print_dash_line

huggingface_dataset_name = "knkarthick/dialogsum"
training_dataset = load_dataset(huggingface_dataset_name)


base_model_name = 'google/flan-t5-small'

base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./full-dialogue-summary-training/checkpoint-15500", torch_dtype=torch.bfloat16)

peft_model = PeftModel.from_pretrained(base_model,
                                       "./peft-dialogue-summary-training/checkpoint-1500",
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

tokenizer = AutoTokenizer.from_pretrained(base_model_name);


dialog_index = 200

dialogue = training_dataset['test'][dialog_index]['dialogue']
summary = training_dataset['test'][dialog_index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output_base = tokenizer.decode(
    base_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

output_instruct = tokenizer.decode(
    instruct_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

output_peft = tokenizer.decode(
    peft_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

print_dash_line()
print(f'INPUT PROMPT:\n{prompt}')
print_dash_line()
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print_dash_line()
print(f'MODEL GENERATION - BASE:\n{output_base}')
print_dash_line()
print(f'MODEL GENERATION - INSTRUCT:\n{output_instruct}')
print_dash_line()
print(f'MODEL GENERATION - PEFT:\n{output_peft}')