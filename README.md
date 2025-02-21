# Fine-Tuning Llama 3 Model

This notebook demonstrates how to fine-tune the Llama 3 model using the Unsloth library. The process includes setting up the environment, loading the model, preparing the dataset, training the model, and saving the fine-tuned model.

## Table of Contents
1. [Installation](#installation)
2. [Loading the Model](#loading-the-model)
3. [Preparing the Dataset](#preparing-the-dataset)
4. [Training the Model](#training-the-model)
5. [Memory and Time Statistics](#memory-and-time-statistics)
6. [Inference](#inference)
7. [Saving the Model](#saving-the-model)
8. [Merging and Saving in Different Formats](#merging-and-saving-in-different-formats)

## Installation

First, ensure that the necessary libraries are installed. The following commands install the required packages:

```bash
%%capture
import torch
major_version, minor_version = torch.cuda.get_device_capability()
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if major_version >= 8:
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
```

## Loading the Model

Load the Llama 3 model using the `FastLanguageModel` class from the Unsloth library. The model is loaded with 4-bit quantization to reduce memory usage.

```python
import torchvision
import torch
from unsloth import FastLanguageModel

max_seq_length = 2048  # Llama-3 supports up to 8k tokens
dtype = torch.float16  # Use torch.float16 or leave as None
load_in_4bit = True    # Use 4-bit quantization

# List of available 4-bit models
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",  # Specify desired model
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Model and tokenizer loaded successfully!")
```

## Preparing the Dataset

The dataset used for fine-tuning is the Alpaca dataset. The dataset is formatted using a custom prompt template.

```python
# System prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # End-of-sequence token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## Training the Model

The model is fine-tuned using the `SFTTrainer` from the `trl` library. The training arguments are configured to optimize the training process.

```python
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        num_train_epochs=4,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
```

## Memory and Time Statistics

After training, you can check the memory usage and training time.

```python
# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```

## Inference

After training, you can perform inference using the fine-tuned model.

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "List the prime numbers contained within the range.",  # instruction
            "1-50",  # input
            "",  # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
tokenizer.batch_decode(outputs)
```

## Saving the Model

The fine-tuned model can be saved locally or pushed to the Hugging Face Hub.

```python
model.save_pretrained("lora_model")  # Local saving
# model.push_to_hub("your_name/lora_model", token="...")  # Online saving
```

## Merging and Saving in Different Formats

The model can be saved in different formats, such as 16-bit, 4-bit, or LoRA adapters.

```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method="lora")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="lora", token="")

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token="")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="q4_k_m", token="")
```

