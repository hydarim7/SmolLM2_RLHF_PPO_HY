---
license: apache-2.0
datasets:
- HumanLLMs/Human-Like-DPO-Dataset
language:
- en
base_model:
- HuggingFaceTB/SmolLM2-135M-Instruct
- ligaydima/ppo-reward-model
pipeline_tag: reinforcement-learning
tags:
- Reinforcement_learning
- RLT
- ProximalPolicyOptimization
- PPO
- RL
- RLHF
---

# 📌 Model Description
This model fine-tunes **SmolLM2-135M-Instruct** using **Reinforcement Learning with Human Feedback (RLHF)**.  
It leverages the **HumanLLMs/Human-Like-DPO-Dataset**, which provides prompts with both accepted and rejected responses.  

We applied **Proximal Policy Optimization (PPO)**, guided by a reward model (**ligaydima/ppo-reward-model**).  
The training process compares new responses with:  
- the **previous model’s output**  
- the **base model’s output**  

The reward model evaluates whether the new response is better while ensuring it doesn’t drift too far from the base model, keeping training **stable and aligned with human preferences**.  

---

# 📂 Dataset
**Name:** [HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)  
**Contents:**  
- Prompt  
- Rejected Response  
- Accepted Response  

---

# 🎯 Training Objective
- Encourage alignment with **accepted answers**  
- Avoid **rejected answers**  
- Improve outputs gradually with PPO while keeping stability  

---

# 🚀 Use Cases
- Human-aligned **text generation**  
- **RLHF & PPO** research experiments  
- Educational resource for reinforcement learning on LLMs  

---
### 🔧 Key Hyperparameters
- **Learning Rate:** `2e-6`  
- **PPO Epochs:** `4`  
- **Mini-batches:** `2`  
- **Batch Size (per device):** `16`  
- **Gradient Accumulation Steps:** `2`  
- **Total Episodes:** `3000`  
- **Clip Range:** `0.2`  
- **KL Coefficient:** `0.02`  
- **Missing EOS Penalty:** `1.0`  

### ⚠️ Note on Competition Limitations
These hyperparameters were **not tuned for maximum performance** but chosen to **fit the constraints of the competition** (limited time and compute).  
They provide a balance between training stability and efficiency, showing how PPO can be applied even under resource restrictions.  

---



# 💻 Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your PPO model
print("📦 Loading your model...")
model = AutoModelForCausalLM.from_pretrained("HYDARIM7/SmolLM2_RLHF_PPO_HY")
tokenizer = AutoTokenizer.from_pretrained("HYDARIM7/SmolLM2_RLHF_PPO_HY")
print("✅ Model loaded!\n")

# Ask a few questions
questions = [
    "What is your name?",
    "How are you today?",
    "What can you help me with?",
    "Tell me about yourself.",
]

print("🤖 Testing your PPO model:\n")
print("="*60)

for question in questions:
    print(f"\n❓ Question: {question}")
    
    # Generate answer
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"💬 Answer: {answer}")
    print("-"*60)

print("\n✅ Test complete!")
```
**#For more details**, please visit our **GitHub** repository:

🔗 Full Details: **[GitHub Repository](https://github.com/hydarim7/SmolLM2_RLHF_PPO_HY)**
