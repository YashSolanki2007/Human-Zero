import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import clip

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max
app.config['UPLOAD_FOLDER'] = '/tmp/humanzero_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'best_model.pt')
FRAMES_PER_VIDEO = 8
FRAME_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

# ── Model ────────────────────────────────────────────────────────────────────

class CLIPVideoDetector(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.encoder = clip_model.visual
        embed_dim = self._get_embed_dim(clip_model)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def _get_embed_dim(self, clip_model):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
            feat = clip_model.encode_image(dummy)
        return feat.shape[-1]

    def forward(self, x):
        feats = self.encoder(x.type(self.encoder.conv1.weight.dtype))
        feats = feats.float()
        return self.head(feats).squeeze(1)


def load_model():
    print(f'Loading CLIP ViT-L/14 on {DEVICE}...')
    clip_model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    clip_model = clip_model.float()
    model = CLIPVideoDetector(clip_model).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f'Loaded checkpoint: {CHECKPOINT_PATH}')
    else:
        print(f'WARNING: No checkpoint found at {CHECKPOINT_PATH}. Using random weights.')
    model.eval()
    return model, preprocess


MODEL, PREPROCESS = load_model()

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_frames(video_path, n_frames=8, size=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(Image.fromarray(frame))
    cap.release()
    while len(frames) < n_frames and frames:
        frames.append(frames[-1])
    return frames or None


@torch.no_grad()
def predict_video(video_path):
    frames = extract_frames(video_path, n_frames=FRAMES_PER_VIDEO)
    if frames is None:
        return None
    tensors = torch.stack([PREPROCESS(f) for f in frames]).to(DEVICE)
    with torch.cuda.amp.autocast():
        logits = MODEL(tensors)
    probs = torch.sigmoid(logits).cpu().numpy().tolist()
    score = float(np.mean(probs))
    return {'score': score, 'frame_probs': probs, 'prediction': int(score > 0.5)}


@torch.no_grad()
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((FRAME_SIZE, FRAME_SIZE))
    tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.cuda.amp.autocast():
        logit = MODEL(tensor)
    score = float(torch.sigmoid(logit).cpu().item())
    return {'score': score, 'frame_probs': [score], 'prediction': int(score > 0.5)}


def confidence_label(score):
    dist = abs(score - 0.5)
    if dist > 0.35:
        return 'High'
    elif dist > 0.18:
        return 'Medium'
    return 'Low'

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    filename = f'{uuid.uuid4().hex}{ext}'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    try:
        if ext in ALLOWED_VIDEO_EXTS:
            result = predict_video(path)
            media_type = 'video'
        else:
            result = predict_image(path)
            media_type = 'image'

        if result is None:
            return jsonify({'error': 'Could not process file'}), 422

        result['confidence'] = confidence_label(result['score'])
        result['label'] = 'AI-Generated' if result['prediction'] == 1 else 'Real'
        result['media_type'] = media_type
        return jsonify(result)

    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
