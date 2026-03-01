'''
Inference code 
'''

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import time

# =========================
# CONFIG
# =========================

MODEL_PATH = "clip_ai_detector.pt"
IMAGE_PATH = "test.png"   # change to your image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# LOAD PROCESSOR
# =========================

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =========================
# LOAD CLIP BACKBONE
# =========================

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =========================
# CLASSIFIER MODEL
# =========================

class CLIPClassifier(nn.Module):

    def __init__(self, clip_model):

        super().__init__()

        self.clip = clip_model

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 2)

        )

    def forward(self, pixel_values):

        with torch.no_grad():

            outputs = self.clip.vision_model(
                pixel_values=pixel_values
            )

            features = outputs.pooler_output

            features = self.clip.visual_projection(features)

        features = features / torch.norm(
            features,
            dim=-1,
            keepdim=True
        )

        logits = self.classifier(features)

        return logits

# =========================
# LOAD TRAINED MODEL
# =========================

model = CLIPClassifier(clip_model)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully")

# =========================
# LOAD IMAGE
# =========================

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

pixel_values = inputs["pixel_values"].to(DEVICE)

# =========================
# RUN INFERENCE WITH TIMING
# =========================

start_time = time.time()

with torch.no_grad():

    logits = model(pixel_values)

    probs = torch.softmax(logits, dim=1)

end_time = time.time()

# =========================
# OUTPUT RESULTS
# =========================

prob_human = probs[0][0].item()
prob_ai = probs[0][1].item()

prediction = torch.argmax(probs, dim=1).item()

label_map = {
    0: "HUMAN",
    1: "AI"
}

print("\nPrediction:", label_map[prediction])

print(f"Confidence Human: {prob_human:.4f}")
print(f"Confidence AI: {prob_ai:.4f}")

print(f"Inference time: {end_time - start_time:.4f} seconds")