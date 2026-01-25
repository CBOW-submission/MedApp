import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc


def clear_gpu_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def load_final_model():
    """Load the final model with both adapters optimized for GPU"""
    model_name = "Qwen/Qwen3-1.7B"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model with GPU optimization...")
    clear_gpu_memory()
    
    # Load base model with optimized settings for GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_state_dict=False,
    )
    
    # Force model to GPU
    print("Moving model to GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    
    print("Loading both adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        "./qwen1.7b-symptoms-precautions-phase2/final_adapters",
        device_map="auto",
        torch_dtype=torch.float16,
        offload_state_dict=False,
    )
    # Ensure model is in evaluation mode
    model.eval()
    
    # Force model to stay on GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Model loaded on device: {model.device}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, tokenizer, device

def run_inference_on_sample(model, tokenizer, clinical_note, device):
    """Run inference on a single clinical note with GPU optimization"""
    # Clear instruction with exact format
    instruction = """You are a medical diagnostic assistant. Analyze the clinical note and provide:
1. Most suspected disease
2. Other diseases with significant risk  
3. Diagnostic reasoning
4. Precautions for the most suspected disease

Format your response EXACTLY as follows:
**Most Suspected Disease:** [name of the disease]
**Other Significant Risks:** [list other possible diseases]
**Diagnostic Reasoning:** [explain your reasoning step by step]
**Precautions:** [recommend precautions for the suspected disease]

Be concise and clinical in your response."""
    
    prompt = f"<|im_start|>user\n{instruction}\n\nClinical Note:\n{clinical_note}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized settings
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
    
    # Decode response WITHOUT skipping special tokens
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(full_response)
    
    # Extract ONLY the assistant's response
    assistant_start = "<|im_start|>assistant"
    if assistant_start in full_response:
        # Get everything after the last occurrence of assistant_start
        parts = full_response.rsplit(assistant_start, 1)
        if len(parts) > 1:
            assistant_part = parts[1]
            # Remove any trailing end tokens
            if "<|im_end|>" in assistant_part:
                assistant_part = assistant_part.split("<|im_end|>")[0]
            assistant_part = assistant_part.strip()
        else:
            assistant_part = full_response.strip()
    else:
        # If no assistant tag found, try to find after the last user message
        user_end = "<|im_end|>\n<|im_start|>assistant"
        if user_end in full_response:
            assistant_part = full_response.split(user_end)[-1].strip()
            if "<|im_end|>" in assistant_part:
                assistant_part = assistant_part.split("<|im_end|>")[0].strip()
        else:
            assistant_part = full_response.strip()
    
    # Remove any remaining special tokens
    assistant_part = assistant_part.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<think>","").replace("</think>","").strip()
    
    # Clear memory
    del inputs, outputs
    clear_gpu_memory()
    
    return assistant_part
