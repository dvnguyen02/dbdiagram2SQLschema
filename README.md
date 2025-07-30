# DB Diagram to JSON Schema with Fine-tuned Qwen 2.5-VL

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.49.0+-yellow.svg)](https://huggingface.co/transformers/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Qwen2.5--VL--Diagrams2SQL-yellow.svg)](https://huggingface.co/zodiac2525/Qwen2.5-VL-Diagrams2SQL)

## 💡 Why This Project?

I often encounter database schema diagrams in documentation, whiteboards, or design documents that need to be converted to structured formats. Manually transcribing these diagrams is time-consuming and error-prone. 

This project explores **fine-tuning a vision-language model** to automatically extract structured JSON schema from ER diagram images, potentially saving hours of manual work in database documentation and migration projects.

## 🧠 What I Built

A **fine-tuned Qwen 2.5 Vision Language Model** that can look at database schema diagrams and extract:
- Table structures with columns and data types
- Primary and foreign key relationships
- Table relationships and cardinalities
- Clean, structured JSON output ready for further processing

## 🛠️ Technical Approach

### Base Model Choice
I chose **Qwen2.5-VL-3B-Instruct** because:
- Excellent vision-language understanding capabilities
- Manageable size for fine-tuning on consumer hardware
- Strong performance on structured output tasks
- Active community and good documentation

### Fine-tuning Strategy
- **Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Target Modules**: q_proj, v_proj, k_proj, o_proj (attention layers)
- **LoRA Configuration**: rank=16, alpha=32 for optimal performance vs. efficiency

### Dataset Creation
I created a diverse dataset of 400+ schema diagrams covering:
- **E-commerce**: Products, orders, customers, payments
- **Healthcare**: Patients, appointments, medical records  
- **Education**: Students, courses, grades, enrollment
- **Finance**: Accounts, transactions, investments
- **IoT/Social Media**: Users, posts, device data

Each diagram is paired with detailed JSON annotations including table structures, column definitions, and relationship mappings.

## � Results & Performance

The fine-tuned model shows significant improvements over the base model:

| Metric | Base Qwen2.5-VL | Fine-tuned Model | Improvement |
|--------|------------------|------------------|-------------|
| **Table Detection Accuracy** | 65.2% | 89.7% | **+24.5%** |
| **Relationship Accuracy** | 58.9% | 84.3% | **+25.4%** |
| **Overall Schema Score** | 62.1% | 87.0% | **+24.9%** |
| **JSON Format Compliance** | 78.1% | 96.2% | **+18.1%** |



## 📓 Notebook Walkthrough

The core of this project is in the `finetuning.ipynb` notebook.


### Comet ML Experiment Tracking
The training process was thoroughly monitored using Comet ML, tracking:
- Loss curves (training & validation)
- Learning rate schedules  
- Memory usage patterns
- Evaluation metrics per epoch

#### Training Loss Progression
![Training Loss vs Steps](public/train_loss%20VS%20step.svg)

#### Validation Error Metrics
![Validation Error](public/val_error.svg)

## 🚀 Quick Usage

### Simple Inference

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import json

# Load the fine-tuned model
model_id = "zodiac2525/Qwen2.5-VL-Diagrams2SQL"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Load your ER diagram
image = Image.open("path/to/your/diagram.png").convert("RGB")

# Prepare the input (same format used in training)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract data in JSON format"}
        ]
    }
]

# Process and generate
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

# Generate schema
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0  # Deterministic output
    )

# Extract JSON from response
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
json_start = generated_text.find('{')
if json_start != -1:
    schema_json = json.loads(generated_text[json_start:])
    print(json.dumps(schema_json, indent=2))
```

## 🤗 Model on Hugging Face

The fine-tuned model is available on Hugging Face Hub:

**[zodiac2525/Qwen2.5-VL-Diagrams2SQL](https://huggingface.co/zodiac2525/Qwen2.5-VL-Diagrams2SQL)**

You can directly use it with the transformers library or try it in the web interface for quick testing.

## � Repository Structure

```
dbdiagram2sql/
├── 📓 finetuning (2).ipynb          # Main training notebook - START HERE
├── 📂 real_diagrams/                # Training dataset  
│   ├── training_dataset.json        # 2000+ annotated schemas
│   └── images/                      # Schema diagram images
│       ├── schema_0000.png          # E-commerce schema
│       ├── schema_0001.png          # Healthcare schema  
│       └── ...                      # More domain examples
├── 📂 src/                          # Organized source code
│   ├── models/qwen_vl_model.py      # Model wrapper classes
│   ├── data/dataset.py              # Data loading utilities
│   └── utils/                       # Helper functions
├── 📂 configs/                      # Training configurations
├── 📂 tests/                        # Unit tests
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```
