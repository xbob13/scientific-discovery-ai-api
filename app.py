from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scientific Discovery AI", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    try:
        model_path = os.getenv("MODEL_PATH", "gpt2")
        logger.info(f"Loading model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        raise RuntimeError("Could not load model")

@app.get("/")
async def root():
    return {"message": "Scientific Discovery AI is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy" if model is not None else "unhealthy"}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = f"Scientific Analysis: {request.prompt}\n\nInsights:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        return {"prompt": request.prompt, "response": generated_text, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
