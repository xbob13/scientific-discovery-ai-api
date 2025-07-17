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
    """Load model without requiring accelerate"""
    global model, tokenizer
    
    try:
        model_path = os.getenv("MODEL_PATH", "gpt2")
        logger.info(f"Loading model: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model WITHOUT device_map (this was causing the error)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Move to CPU manually (instead of using device_map)
        model = model.to("cpu")
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        logger.error("Failed to load model!")
        raise RuntimeError("Could not load model")

@app.get("/")
async def root():
    return {"message": "Scientific Discovery AI is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy" if model is not None else "unhealthy"}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate scientific insights from prompt"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Add scientific context
        enhanced_prompt = f"Scientific Analysis: {prompt}\n\nInsights:"
        
        # Tokenize
        inputs = tokenizer(
            enhanced_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400,
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(enhanced_prompt):].strip()
        
        return {
            "prompt": request.prompt,
            "response": generated_text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/discover")
async def discover_patterns(request: GenerateRequest):
    """Specialized endpoint for scientific discovery"""
    
    discovery_prompt = f"""Cross-disciplinary scientific analysis:

Query: {request.prompt}

This question involves potential connections between multiple scientific fields. Key insights:

1. Mathematical frameworks that might apply
2. Experimental techniques that could be relevant
3. Theoretical principles connecting different domains
4. Novel interdisciplinary perspectives

Discovery insights:"""
    
    enhanced_request = GenerateRequest(
        prompt=discovery_prompt,
        max_length=request.max_length,
        temperature=request.temperature
    )
    
    return await generate_text(enhanced_request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
