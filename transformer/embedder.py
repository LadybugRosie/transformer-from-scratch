from transformers import AutoTokenizer
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

test_string = "Hello, my name is David Hovey!"

inputs = tokenizer(test_string, return_tensors="pt")

print(inputs)
