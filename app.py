from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scientific Discovery AI", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

# Scientific knowledge base for intelligent responses
SCIENTIFIC_PATTERNS = {
    "quantum": [
        "quantum mechanics principles apply to microscopic systems",
        "quantum entanglement creates non-local correlations",
        "quantum tunneling enables nuclear fusion in stars",
        "quantum coherence affects biological processes",
        "quantum computing leverages superposition states"
    ],
    "biology": [
        "biological systems exhibit emergent complexity",
        "evolutionary processes optimize survival strategies",
        "cellular mechanisms involve molecular machines",
        "genetic information encodes functional proteins",
        "neural networks process environmental information"
    ],
    "chemistry": [
        "chemical reactions follow thermodynamic principles",
        "molecular structures determine functional properties",
        "catalysts lower activation energy barriers",
        "chemical bonding involves electron sharing",
        "reaction kinetics depend on molecular collisions"
    ],
    "physics": [
        "physical laws govern energy and matter interactions",
        "conservation principles constrain system behavior",
        "field theories describe fundamental forces",
        "statistical mechanics explains macroscopic properties",
        "symmetry principles guide theoretical frameworks"
    ],
    "algorithms": [
        "algorithmic complexity determines computational efficiency",
        "machine learning models extract patterns from data",
        "optimization algorithms find optimal solutions",
        "data structures organize information efficiently",
        "parallel processing accelerates computational tasks"
    ],
    "astronomy": [
        "astronomical observations reveal cosmic evolution",
        "gravitational forces shape celestial structures",
        "electromagnetic radiation carries cosmic information",
        "dark matter influences galactic dynamics",
        "stellar nucleosynthesis creates heavier elements"
    ]
}

CROSS_CONNECTIONS = {
    ("quantum", "biology"): "Quantum effects in biological systems include photosynthesis efficiency, enzyme catalysis, and potentially consciousness mechanisms.",
    ("quantum", "chemistry"): "Quantum mechanics underlies chemical bonding, reaction pathways, and molecular orbital theory.",
    ("quantum", "physics"): "Quantum field theory unifies quantum mechanics with special relativity and describes fundamental particles.",
    ("quantum", "algorithms"): "Quantum algorithms like Shor's and Grover's provide exponential speedups for specific computational problems.",
    ("quantum", "astronomy"): "Quantum mechanics explains stellar nucleosynthesis, black hole radiation, and early universe physics.",
    ("biology", "chemistry"): "Biochemical processes involve molecular recognition, enzymatic catalysis, and metabolic pathway regulation.",
    ("biology", "physics"): "Biophysics applies physical principles to understand cellular mechanics, neural signaling, and biomolecular structure.",
    ("biology", "algorithms"): "Bioinformatics algorithms analyze genetic sequences, predict protein structures, and model evolutionary processes.",
    ("biology", "astronomy"): "Astrobiology investigates life's potential in extreme environments and searches for extraterrestrial biosignatures.",
    ("chemistry", "physics"): "Physical chemistry applies quantum mechanics and thermodynamics to understand molecular behavior and reactions.",
    ("chemistry", "algorithms"): "Computational chemistry uses algorithms to predict molecular properties, design drugs, and optimize reactions.",
    ("chemistry", "astronomy"): "Astrochemistry studies molecular formation in space, planetary atmospheres, and interstellar medium composition.",
    ("physics", "algorithms"): "Computational physics simulates complex systems, solves differential equations, and models physical phenomena.",
    ("physics", "astronomy"): "Astrophysics applies physical laws to understand stellar evolution, galaxy formation, and cosmological processes.",
    ("algorithms", "astronomy"): "Astronomical algorithms process observational data, detect exoplanets, and analyze cosmic microwave background."
}

ANOMALY_PATTERNS = [
    "unexpected scaling behavior suggests new underlying mechanisms",
    "deviation from theoretical predictions indicates novel physics",
    "unusual correlations between variables point to hidden connections",
    "systematic errors in measurements may reveal instrumental limitations",
    "outlier data points often signal breakthrough discoveries",
    "inconsistent results across experiments suggest methodological issues",
    "emergent properties not predictable from individual components",
    "phase transitions exhibit critical behavior and scaling laws"
]

def extract_keywords(text):
    """Extract scientific keywords from text"""
    text = text.lower()
    keywords = []
    
    for domain in SCIENTIFIC_PATTERNS.keys():
        if domain in text or any(term in text for term in [
            "quantum" if domain == "quantum" else domain,
            "bio" if domain == "biology" else "",
            "chem" if domain == "chemistry" else "",
            "phys" if domain == "physics" else "",
            "algo" if domain == "algorithms" else "",
            "astro" if domain == "astronomy" else ""
        ]):
            keywords.append(domain)
    
    return keywords

def generate_scientific_response(prompt, keywords):
    """Generate intelligent scientific response based on keywords"""
    
    if not keywords:
        keywords = ["physics", "chemistry"]  # Default domains
    
    response_parts = []
    
    # Add domain-specific insights
    for keyword in keywords[:2]:  # Limit to 2 domains
        if keyword in SCIENTIFIC_PATTERNS:
            patterns = SCIENTIFIC_PATTERNS[keyword]
            selected_pattern = random.choice(patterns)
            response_parts.append(f"From {keyword}: {selected_pattern}.")
    
    # Add cross-disciplinary connections
    if len(keywords) >= 2:
        key_pair = tuple(sorted(keywords[:2]))
        if key_pair in CROSS_CONNECTIONS:
            response_parts.append(f"Cross-disciplinary insight: {CROSS_CONNECTIONS[key_pair]}")
    
    # Add methodology or anomaly detection
    if "anomaly" in prompt.lower() or "unusual" in prompt.lower():
        anomaly_insight = random.choice(ANOMALY_PATTERNS)
        response_parts.append(f"Anomaly detection: {anomaly_insight}.")
    
    # Add discovery potential
    discovery_endings = [
        "This suggests potential for breakthrough discoveries through interdisciplinary collaboration.",
        "Further research could reveal universal principles underlying these phenomena.",
        "The convergence of these fields may unlock new technological applications.",
        "These patterns indicate opportunities for novel experimental approaches.",
        "Understanding these connections could accelerate scientific progress."
    ]
    
    response_parts.append(random.choice(discovery_endings))
    
    return " ".join(response_parts)

@app.on_event("startup")
async def startup_event():
    logger.info("Scientific Discovery AI started successfully!")
    logger.info("Using lightweight rule-based scientific reasoning system")

@app.get("/")
async def root():
    return {
        "message": "Scientific Discovery AI is running!",
        "status": "healthy",
        "mode": "lightweight",
        "capabilities": [
            "Cross-disciplinary pattern recognition",
            "Scientific insight generation",
            "Anomaly detection guidance",
            "Research opportunity identification"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "memory_usage": "< 100MB", "model": "rule-based"}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate scientific insights from prompt"""
    
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Extract scientific keywords
        keywords = extract_keywords(prompt)
        
        # Generate intelligent response
        response = generate_scientific_response(prompt, keywords)
        
        return {
            "prompt": request.prompt,
            "response": response,
            "keywords_detected": keywords,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/discover")
async def discover_patterns(request: GenerateRequest):
    """Specialized endpoint for scientific discovery"""
    
    keywords = extract_keywords(request.prompt)
    
    if not keywords:
        keywords = ["physics", "chemistry", "biology"]
    
    # Generate comprehensive discovery response
    discovery_parts = []
    
    discovery_parts.append(f"Cross-disciplinary analysis of: {request.prompt}")
    discovery_parts.append("")
    
    # Mathematical frameworks
    discovery_parts.append("Mathematical frameworks that apply:")
    discovery_parts.append("- Differential equations for dynamic systems")
    discovery_parts.append("- Statistical mechanics for emergent properties")
    discovery_parts.append("- Graph theory for network analysis")
    discovery_parts.append("- Optimization theory for efficiency principles")
    discovery_parts.append("")
    
    # Experimental techniques
    discovery_parts.append("Relevant experimental techniques:")
    discovery_parts.append("- Spectroscopic methods for molecular identification")
    discovery_parts.append("- Microscopy for structural analysis")
    discovery_parts.append("- Computational modeling for prediction")
    discovery_parts.append("- Data mining for pattern recognition")
    discovery_parts.append("")
    
    # Theoretical principles
    discovery_parts.append("Connecting theoretical principles:")
    for keyword in keywords[:2]:
        if keyword in SCIENTIFIC_PATTERNS:
            pattern = random.choice(SCIENTIFIC_PATTERNS[keyword])
            discovery_parts.append(f"- {keyword.capitalize()}: {pattern}")
    
    discovery_parts.append("")
    discovery_parts.append("This interdisciplinary approach reveals potential breakthroughs through methodology transfer and conceptual synthesis.")
    
    response = "\n".join(discovery_parts)
    
    return {
        "prompt": request.prompt,
        "response": response,
        "analysis_type": "cross-disciplinary",
        "domains": keywords,
        "status": "success"
    }

@app.post("/anomaly")
async def detect_anomalies(request: GenerateRequest):
    """Detect potential anomalies and breakthrough opportunities"""
    
    keywords = extract_keywords(request.prompt)
    
    anomaly_response = f"Anomaly detection analysis for: {request.prompt}\n\n"
    
    # Anomaly indicators
    anomaly_response += "Potential anomaly indicators:\n"
    anomaly_response += "1. Unexpected scaling relationships in data\n"
    anomaly_response += "2. Systematic deviations from theoretical predictions\n"
    anomaly_response += "3. Correlation patterns not explained by current models\n"
    anomaly_response += "4. Reproducible experimental anomalies\n\n"
    
    # Domain-specific anomaly patterns
    if keywords:
        anomaly_response += f"Domain-specific considerations for {', '.join(keywords)}:\n"
        for keyword in keywords[:2]:
            if keyword in SCIENTIFIC_PATTERNS:
                anomaly_response += f"- {keyword.capitalize()}: {random.choice(ANOMALY_PATTERNS)}\n"
    
    anomaly_response += "\nRecommendation: Investigate these patterns through controlled experiments and cross-validation with independent research groups."
    
    return {
        "prompt": request.prompt,
        "response": anomaly_response,
        "analysis_type": "anomaly_detection",
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
