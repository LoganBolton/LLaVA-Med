#!/usr/bin/env python3
"""
MedGemma Script - Run with dedicated virtual environment
Usage: source medgemma_env/bin/activate && python run_medgemma.py
"""
import os
from transformers import pipeline
from PIL import Image
import torch

print("Starting MedGemma inference...")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

try:
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nPlease ensure you have:")
    print("1. Accepted license: https://huggingface.co/google/medgemma-4b-it")
    print("2. Logged in: huggingface-cli login")
    exit(1)

directory = "/home/log/Github/LLaVA-Med/images"

print(f"\nProcessing images from: {directory}")
print("=" * 60)

for filename in sorted(os.listdir(directory)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filepath = os.path.join(directory, filename)
        image = Image.open(filepath)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Does this patient have pneumonia? Provide your reasoning and then answer Yes/No"},
                    {"type": "image", "image": image},
                ],
            }
        ]

        try:
            result = pipe(text=messages, generate_kwargs={"do_sample": False, "max_new_tokens": 512})
            print(f"\n📁 {filename}")
            print(f"📊 {result[0]['generated_text'][-1]['content'].strip()}")
            print("-" * 60)
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            continue

print("\n🎉 Processing complete!")