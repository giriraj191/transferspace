# Env Creation
# pip installation


# source myenv/bin/activate
# CPU : pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117



from transformers import pipeline
import requests
import gc

# === Local Hugging Face Pipeline ===
def local_pipeline_demo():
    # Explicitly specify model and enable GPU if available
    classifier = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=0  # Use GPU; set to -1 for CPU
    )
    text = "Hugging Face makes machine learning easy and fun!"
    result = classifier(text)
    print("Local Pipeline Output:", result)

# === Hugging Face Inference API ===
def hf_inference_api_demo():
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer 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"}
    text = "The new iPhone is amazing with its advanced camera features!"
    candidate_labels = ["technology", "sports", "politics"]

    payload = {"inputs": text,
            "parameters": {"candidate_labels": candidate_labels}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    print("HF Inference API Output:", response.json())

# Run both demos
if __name__ == "__main__":
    local_pipeline_demo()
    hf_inference_api_demo()
    # Clean up resources
    gc.collect()
