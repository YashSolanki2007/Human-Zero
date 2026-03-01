# import os
# import uuid
# import numpy as np
# import torch
# import torch.nn as nn
# import cv2
# from PIL import Image
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename

# # Image model uses HuggingFace transformers
# from transformers import CLIPModel, CLIPProcessor

# # Video model uses OpenAI CLIP
# import clip  # pip install git+https://github.com/openai/CLIP.git

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
# app.config['UPLOAD_FOLDER'] = '/tmp/humanzero_uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# IMAGE_CHECKPOINT = os.environ.get('IMAGE_CHECKPOINT', 'clip_ai_detector.pt')
# VIDEO_CHECKPOINT = os.environ.get('VIDEO_CHECKPOINT', 'best_model.pt')

# FRAMES_PER_VIDEO = 8
# FRAME_SIZE       = 224
# DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'

# ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

# print(f'Using device: {DEVICE}')

# # ── IMAGE MODEL  (from finetune.py) ──────────────────────────────────────────
# # Architecture: CLIPClassifier wrapping CLIPModel (ViT-base-patch32)
# # Output: softmax over [human, AI] — index 1 is AI probability

# class CLIPClassifier(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.clip = clip_model
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 2)
#         )

#     def forward(self, pixel_values):
#         with torch.no_grad():
#             outputs = self.clip.vision_model(pixel_values=pixel_values)
#             features = outputs.pooler_output
#             features = self.clip.visual_projection(features)
#         features = features / torch.norm(features, dim=-1, keepdim=True)
#         return self.classifier(features)


# def load_image_model():
#     print('Loading image model (CLIPClassifier, ViT-base-patch32)...')
#     MODEL_PATH = "clip_ai_detector.pt"
#     processor  = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
#     clip_base  = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
#     model = CLIPClassifier(clip_base)
#     # if os.path.exists(IMAGE_CHECKPOINT):
#     #     model.load_state_dict(torch.load(IMAGE_CHECKPOINT, map_location=DEVICE))
#     #     print(f'Loaded image checkpoint: {IMAGE_CHECKPOINT}')
#     # else:
#     #     print(f'WARNING: Image checkpoint not found at {IMAGE_CHECKPOINT}. Using random weights.')
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model = model.to(DEVICE)
#     model.eval()
#     return model, processor


# # ── VIDEO MODEL  (from video-detect.py) ───────────────────────────────────────
# # Architecture: CLIPVideoDetector wrapping CLIP ViT-L/14
# # Output: sigmoid over single logit — that value IS the AI probability

# class CLIPVideoDetector(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.encoder = clip_model.visual
#         embed_dim = self._get_embed_dim(clip_model)
#         self.head = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim, 256),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1)
#         )

#     def _get_embed_dim(self, clip_model):
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
#             feat = clip_model.encode_image(dummy)
#         return feat.shape[-1]

#     def forward(self, x):
#         feats = self.encoder(x.type(self.encoder.conv1.weight.dtype))
#         feats = feats.float()
#         return self.head(feats).squeeze(1)


# def load_video_model():
#     print('Loading video model (CLIPVideoDetector, ViT-L/14)...')
#     clip_model, preprocess = clip.load('ViT-L/14', device=DEVICE)
#     clip_model = clip_model.float()
#     model = CLIPVideoDetector(clip_model).to(DEVICE)
#     if os.path.exists(VIDEO_CHECKPOINT):
#         model.load_state_dict(torch.load(VIDEO_CHECKPOINT, map_location=DEVICE))
#         print(f'Loaded video checkpoint: {VIDEO_CHECKPOINT}')
#     else:
#         print(f'WARNING: Video checkpoint not found at {VIDEO_CHECKPOINT}. Using random weights.')
#     model.eval()
#     return model, preprocess


# # Load both models at startup
# IMAGE_MODEL, IMAGE_PROCESSOR = load_image_model()
# VIDEO_MODEL, VIDEO_PREPROCESS = load_video_model()


# # ── INFERENCE ─────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def predict_image(image_path):
#     """Uses CLIPClassifier (finetune.py). Returns AI probability from softmax index 1."""
#     image = Image.open(image_path).convert('RGB')
#     inputs = IMAGE_PROCESSOR(images=image, return_tensors='pt')
#     pixel_values = inputs['pixel_values'].to(DEVICE)

#     logits = IMAGE_MODEL(pixel_values)
#     probs  = torch.softmax(logits, dim=1)
#     print(torch.argmax(probs, dim=1).item())
#     prob_ai    = probs[0][1].item()
#     prediction = int(torch.argmax(probs, dim=1).item())  # 0=human, 1=AI

#     return {
#         'score':      prob_ai,
#         'prediction': prediction,
#         'media_type': 'image',
#     }


# def extract_frames(video_path, n_frames=8, size=224):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total < 1:
#         cap.release()
#         return None
#     indices = np.linspace(0, total - 1, n_frames, dtype=int)
#     frames = []
#     for idx in indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (size, size))
#         frames.append(Image.fromarray(frame))
#     cap.release()
#     while len(frames) < n_frames and frames:
#         frames.append(frames[-1])
#     return frames or None


# @torch.no_grad()
# def predict_video(video_path):
#     """Uses CLIPVideoDetector (video-detect.py). Returns averaged sigmoid score."""
#     frames = extract_frames(video_path, n_frames=FRAMES_PER_VIDEO)
#     if frames is None:
#         return None
#     tensors = torch.stack([VIDEO_PREPROCESS(f) for f in frames]).to(DEVICE)

#     with torch.cuda.amp.autocast():
#         logits = VIDEO_MODEL(tensors)

#     probs      = torch.sigmoid(logits).cpu().numpy().tolist()
#     score      = float(np.mean(probs))
#     prediction = int(score > 0.5)

#     return {
#         'score':      score,
#         'prediction': prediction,
#         'media_type': 'video',
#     }


# def confidence_label(score):
#     dist = abs(score - 0.5)
#     if dist > 0.35: return 'High'
#     if dist > 0.18: return 'Medium'
#     return 'Low'


# # ── ROUTES ────────────────────────────────────────────────────────────────────

# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     f = request.files['file']
#     if not f.filename:
#         return jsonify({'error': 'Empty filename'}), 400

#     ext = os.path.splitext(f.filename)[1].lower()
#     if ext not in ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS:
#         return jsonify({'error': f'Unsupported file type: {ext}'}), 400

#     filename = f'{uuid.uuid4().hex}{ext}'
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     f.save(path)

#     try:
#         if ext in ALLOWED_VIDEO_EXTS:
#             result = predict_video(path)
#         else:
#             result = predict_image(path)

#         if result is None:
#             return jsonify({'error': 'Could not process file'}), 422

#         result['label']      = 'AI-Generated' if result['prediction'] == 1 else 'Real'
#         result['confidence'] = confidence_label(result['score'])
#         return jsonify(result)

#     finally:
#         if os.path.exists(path):
#             os.remove(path)


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Image model uses HuggingFace transformers
from transformers import CLIPModel, CLIPProcessor

# Video model uses OpenAI CLIP
import clip  # pip install git+https://github.com/openai/CLIP.git

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.config['UPLOAD_FOLDER'] = '/tmp/humanzero_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

IMAGE_CHECKPOINT = os.environ.get('IMAGE_CHECKPOINT', 'clip_ai_detector.pt')
VIDEO_CHECKPOINT = os.environ.get('VIDEO_CHECKPOINT', 'best_model.pt')

FRAMES_PER_VIDEO = 8
FRAME_SIZE       = 224
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

print(f'Using device: {DEVICE}')

# ── IMAGE MODEL  (from finetune.py) ──────────────────────────────────────────
# Architecture: CLIPClassifier wrapping CLIPModel (ViT-base-patch32)
# Output: softmax over [human, AI] — index 1 is AI probability

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
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            features = outputs.pooler_output
            features = self.clip.visual_projection(features)
        features = features / torch.norm(features, dim=-1, keepdim=True)
        return self.classifier(features)


def load_image_model():
    print('Loading image model (CLIPClassifier, ViT-base-patch32)...')
    processor  = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_base  = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPClassifier(clip_base)
    if os.path.exists(IMAGE_CHECKPOINT):
        model.load_state_dict(torch.load(IMAGE_CHECKPOINT, map_location=DEVICE))
        print(f'Loaded image checkpoint: {IMAGE_CHECKPOINT}')
    else:
        print(f'WARNING: Image checkpoint not found at {IMAGE_CHECKPOINT}. Using random weights.')
    model = model.to(DEVICE)
    model.eval()
    return model, processor


# ── VIDEO MODEL  (from video-detect.py) ───────────────────────────────────────
# Architecture: CLIPVideoDetector wrapping CLIP ViT-L/14
# Output: sigmoid over single logit — that value IS the AI probability

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


def load_video_model():
    print('Loading video model (CLIPVideoDetector, ViT-L/14)...')
    clip_model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    clip_model = clip_model.float()
    model = CLIPVideoDetector(clip_model).to(DEVICE)
    if os.path.exists(VIDEO_CHECKPOINT):
        model.load_state_dict(torch.load(VIDEO_CHECKPOINT, map_location=DEVICE))
        print(f'Loaded video checkpoint: {VIDEO_CHECKPOINT}')
    else:
        print(f'WARNING: Video checkpoint not found at {VIDEO_CHECKPOINT}. Using random weights.')
    model.eval()
    return model, preprocess


# Load both models at startup
IMAGE_MODEL, IMAGE_PROCESSOR = load_image_model()
VIDEO_MODEL, VIDEO_PREPROCESS = load_video_model()


# ── INFERENCE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(image_path):
    """Uses CLIPClassifier (finetune.py). Returns AI probability from softmax index 1."""
    image = Image.open(image_path).convert('RGB')
    inputs = IMAGE_PROCESSOR(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(DEVICE)

    logits = IMAGE_MODEL(pixel_values)
    probs  = torch.softmax(logits, dim=1)

    prob_ai    = probs[0][1].item()
    prediction = int(torch.argmax(probs, dim=1).item())  # 0=human, 1=AI

    return {
        'score':      prob_ai,
        'prediction': prediction,
        'media_type': 'image',
    }


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
    """Uses CLIPVideoDetector (video-detect.py). Returns averaged sigmoid score."""
    frames = extract_frames(video_path, n_frames=FRAMES_PER_VIDEO)
    if frames is None:
        return None
    tensors = torch.stack([VIDEO_PREPROCESS(f) for f in frames]).to(DEVICE)

    with torch.cuda.amp.autocast():
        logits = VIDEO_MODEL(tensors)

    probs      = torch.sigmoid(logits).cpu().numpy().tolist()
    score      = float(np.mean(probs))
    prediction = int(score > 0.5)

    return {
        'score':      score,
        'prediction': prediction,
        'media_type': 'video',
    }


def confidence_label(score):
    dist = abs(score - 0.5)
    print(score)
    print(dist)
    if dist > 0.15: return 'High'
    # if dist > 0.18: return 'Medium'
    return 'Low'


# ── ROUTES ────────────────────────────────────────────────────────────────────

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
        else:
            result = predict_image(path)

        if result is None:
            return jsonify({'error': 'Could not process file'}), 422

        result['label']      = 'AI-Generated' if result['prediction'] == 1 else 'Real'
        result['confidence'] = confidence_label(result['score'])
        return jsonify(result)

    finally:
        if os.path.exists(path):
            os.remove(path)


# ── FGSM PURIFICATION ROUTE ───────────────────────────────────────────────────

class CLIPClassifierWithAttack(nn.Module):
    """Wraps loaded IMAGE_MODEL and exposes forward_for_attack for FGSM."""
    def __init__(self, base_model):
        super().__init__()
        self.clip       = base_model.clip
        self.classifier = base_model.classifier

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs  = self.clip.vision_model(pixel_values=pixel_values)
            features = self.clip.visual_projection(outputs.pooler_output)
        features = features / torch.norm(features, dim=-1, keepdim=True)
        return self.classifier(features)

    def forward_for_attack(self, pixel_values):
        """Gradients flow through vision model — required for FGSM."""
        outputs  = self.clip.vision_model(pixel_values=pixel_values)
        features = self.clip.visual_projection(outputs.pooler_output)
        features = features / torch.norm(features, dim=-1, keepdim=True)
        return self.classifier(features)


# Build once at module level so it reuses the already-loaded IMAGE_MODEL weights
_attack_model = CLIPClassifierWithAttack(IMAGE_MODEL).to(DEVICE)
_attack_model.eval()


@app.route('/purify', methods=['POST'])
def purify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return jsonify({'error': 'Images only for purification'}), 400

    try:
        epsilon = float(request.form.get('epsilon', 0.05))
        epsilon = max(0.0, min(0.5, epsilon))
    except ValueError:
        return jsonify({'error': 'Invalid epsilon value'}), 400

    fname = f'{uuid.uuid4().hex}{ext}'
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(fpath)

    try:
        # Load & preprocess image
        image        = Image.open(fpath).convert('RGB')
        orig_size    = image.size                              # preserve original resolution
        inputs       = IMAGE_PROCESSOR(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(DEVICE)
        pixel_values.requires_grad = True

        # --- Original prediction ---
        with torch.no_grad():
            logits    = _attack_model(pixel_values)
            probs     = torch.softmax(logits, dim=1)
            orig_pred = int(torch.argmax(probs, dim=1).item())

        # --- FGSM (matches combined.py exactly) ---
        # target = flipped label so gradient pushes away from current class
        criterion    = nn.CrossEntropyLoss()
        target_label = torch.tensor([1 - orig_pred], device=DEVICE)

        logits_attack = _attack_model.forward_for_attack(pixel_values)
        loss          = criterion(logits_attack, target_label)
        _attack_model.zero_grad()
        loss.backward()

        perturbation = epsilon * pixel_values.grad.sign()
        adv_image    = torch.clamp(pixel_values + perturbation, 0, 1).detach()

        # --- Adversarial prediction ---
        with torch.no_grad():
            adv_logits = _attack_model(adv_image)
            adv_probs  = torch.softmax(adv_logits, dim=1)
            adv_pred   = int(torch.argmax(adv_probs, dim=1).item())

        # --- Tensor → PNG base64 ---
        import io, base64
        adv_np  = adv_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_pil = Image.fromarray((adv_np * 255).astype('uint8'))
        adv_pil = adv_pil.resize(orig_size, Image.LANCZOS)

        buf = io.BytesIO()
        adv_pil.save(buf, format='PNG')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')

        label_map = {0: 'Real', 1: 'AI-Generated'}
        return jsonify({
            'image_b64':    img_b64,
            'orig_label':   label_map[orig_pred],
            'adv_label':    label_map[adv_pred],
            'orig_ai_prob': round(probs[0][1].item(), 4),
            'adv_ai_prob':  round(adv_probs[0][1].item(), 4),
            'flipped':      orig_pred != adv_pred,
        })

    finally:
        if os.path.exists(fpath):
            os.remove(fpath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)