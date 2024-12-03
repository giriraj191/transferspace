# pip install openai matplotlib pillow requests


import requests
from PIL import Image
import matplotlib.pyplot as plt
import os
from openai import OpenAI

api_key = "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"

client = OpenAI(
#  organization='org-eDlIO9iRlWeXkMSzgpKYN3Yk',
#  project='$PROJECT_ID',
  api_key=api_key
)

os.makedirs("dalle_output", exist_ok=True)


prompts = [
    "A futuristic Iron Man suit flying over New York City",
 #   "An Avengers team poster in a cyberpunk universe",
 #   "Thor summoning lightning in a stormy sky",
 #   "A serene view of Wakanda with advanced technology in the background",
 #   "Captain America holding his shield in a desert landscape during sunset",
]

def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    print(image_url)
    image_data = requests.get(image_url).content
    image_path = f"dalle_output/image_1.png"
    with open(image_path,"wb") as f:
        f.write(image_data)

 

generate_image(prompts[0])

print(f"Images generated and saved to 'dalle_output' folder.")