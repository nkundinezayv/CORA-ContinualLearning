#!/usr/bin/env python3
"""
Continuum-Prime-CML (CML-Core) – Inference Engine

This script:
- Loads the trained CML-Core model snapshot from ./models
- Supports image and text prediction for any task with:
    encoder_{TASK}.pth
    head_{TASK}.pth
- Keeps all training / continual-learning logic private
"""

import os
import glob
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

LATENT_DIM = 64
TASK_EMB_DIM = 16
D_MODEL = LATENT_DIM + TASK_EMB_DIM
HIDDEN_EXP = 64
MAX_EXPERTS = 64
INIT_EXPERTS = 2
TOP_K = 2
REASON_STEPS = 2
TEXT_MAX_LEN = 32
TEXT_EMB_DIM = 64
SELF_AWARE_HIDDEN = 128

DEFAULT_MODEL_DIR = "./models"


# ============================================================
# CLASS LABEL MAPPINGS (OPTIONAL, FOR HUMAN-READABLE OUTPUT)
# ============================================================

CLASS_NAMES = {
    "MNIST": {i: str(i) for i in range(10)},
    "FASHIONMNIST": {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
        4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Boot"
    },
    "EMNIST-LETTERS": {i: chr(ord("A") + i) for i in range(26)},
    "CIFAR01": {0: "Airplane", 1: "Automobile"},
    "CIFAR10": {
        0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
        5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
    },
    "TEXT-SENT": {0: "Negative", 1: "Positive"},
    "TEXT-IMDB": {0: "Negative", 1: "Positive"},
    "TEXT-QA": {0: "Long answer", 1: "Short answer"},
    "TEXT-MULTINLI": {0: "Entailment", 1: "Neutral", 2: "Contradiction"},
    "TEXT-AGNEWS": {0: "World", 1: "Business", 2: "Sports", 3: "Technology"},
    "TEXT-GOEMOTIONS": {
        0: "Anger", 1: "Disgust", 2: "Fear", 3: "Joy", 4: "Sadness", 5: "Surprise"
    },
}


# ============================================================
# VOCAB
# ============================================================

class SimpleVocab:
    def __init__(self, token_to_id=None, id_to_token=None):
        self.token_to_id = token_to_id or {}
        self.id_to_token = id_to_token or []

    def encode(self, text: str, max_len: int = TEXT_MAX_LEN) -> torch.Tensor:
        words = text.lower().strip().split()
        ids = [self.token_to_id.get(w, 1) for w in words]  # 1 = <unk>
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))              # 0 = <pad>
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long)


# ============================================================
# ENCODERS
# ============================================================

class StandardMNISTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc(h))


class CIFAR10Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc(h))


class EMNISTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc(h))


class FashionMNISTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc(h))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = TEXT_EMB_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.proj = nn.Linear(emb_dim, latent_dim)

    def forward(self, x):
        emb = self.emb(x)                                  # (B, L, E)
        mask = (x != 0).float().unsqueeze(-1)              # (B, L, 1)
        summed = (emb * mask).sum(1)                       # (B, E)
        lengths = mask.sum(1).clamp(min=1.0)               # (B, 1)
        mean = summed / lengths
        return F.relu(self.proj(mean))


# ============================================================
# BRAIN MODULES
# ============================================================

class IntrinsicMotivationHead(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.novelty_net = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Linear(d // 2, 1), nn.Sigmoid(),
        )
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def forward(self, h, h_pred, forward_error=None):
        if forward_error is None:
            forward_error = torch.norm(h_pred - h, dim=-1, keepdim=True)
        safe_var = torch.clamp(self.running_var, min=1e-6)
        h_norm = (h - self.running_mean) / torch.sqrt(safe_var)
        novelty = self.novelty_net(h_norm)
        error_normalized = torch.sigmoid(forward_error - 0.5)
        curiosity = 0.4 * error_normalized + 0.6 * novelty
        curiosity = torch.clamp(curiosity, min=0.1, max=0.9)
        return curiosity, novelty, error_normalized


class SelfAwarenessNetwork(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.why_network = nn.Sequential(
            nn.Linear(d * 2, SELF_AWARE_HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(SELF_AWARE_HIDDEN, SELF_AWARE_HIDDEN), nn.ReLU(),
            nn.Linear(SELF_AWARE_HIDDEN, SELF_AWARE_HIDDEN // 2),
        )
        self.if_network = nn.Sequential(
            nn.Linear(d + SELF_AWARE_HIDDEN // 2, SELF_AWARE_HIDDEN),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(SELF_AWARE_HIDDEN, SELF_AWARE_HIDDEN // 2), nn.ReLU(),
            nn.Linear(SELF_AWARE_HIDDEN // 2, 1), nn.Sigmoid(),
        )
        self.think_network = nn.Sequential(
            nn.Linear(d + SELF_AWARE_HIDDEN // 2, SELF_AWARE_HIDDEN),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(SELF_AWARE_HIDDEN, SELF_AWARE_HIDDEN), nn.ReLU(),
            nn.Linear(SELF_AWARE_HIDDEN, d),
        )
        self.output_network = nn.Sequential(
            nn.Linear(d * 2, SELF_AWARE_HIDDEN), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(SELF_AWARE_HIDDEN, SELF_AWARE_HIDDEN), nn.ReLU(),
            nn.Linear(SELF_AWARE_HIDDEN, d),
        )
        self.register_buffer("reflection_mean", torch.zeros(1))

    def forward(self, h, expert_mixing):
        expert_context = (
            expert_mixing.mean(dim=1, keepdim=True).expand(-1, h.size(1))
            if expert_mixing.dim() > 1
            else expert_mixing
        )
        h_with_context = torch.cat([h, expert_context * h], dim=-1)
        why_reasoning = self.why_network(h_with_context)

        h_with_why = torch.cat([h, why_reasoning], dim=-1)
        if_quality = self.if_network(h_with_why)
        think_alternative = self.think_network(h_with_why)

        blended = h + if_quality * think_alternative
        h_refined = self.output_network(torch.cat([h, blended], dim=-1))

        return h_refined, why_reasoning, if_quality, think_alternative


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, HIDDEN_EXP), nn.ReLU(),
            nn.Linear(HIDDEN_EXP, D_MODEL), nn.ReLU(),
        )

    def forward(self, h):
        return self.net(h)


class ForwardModel(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, h):
        return self.net(h)


class SharedBrain(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_emb = nn.Embedding(num_tasks, TASK_EMB_DIM)
        self.experts = nn.ModuleList()
        self.num_experts = 0

        self.router_1 = nn.Linear(D_MODEL, MAX_EXPERTS)
        self.router_2 = nn.Linear(D_MODEL, MAX_EXPERTS)

        self.forward_model = ForwardModel(D_MODEL)
        self.intrinsic_motivation = IntrinsicMotivationHead(D_MODEL)
        self.self_awareness = SelfAwarenessNetwork(D_MODEL)

        self.register_buffer("usage_1", torch.zeros(MAX_EXPERTS))
        self.register_buffer("usage_2", torch.zeros(MAX_EXPERTS))

        for _ in range(INIT_EXPERTS):
            self.add_expert()

    def add_expert(self):
        if self.num_experts >= MAX_EXPERTS:
            return
        self.experts.append(Expert())
        self.num_experts += 1

    def forward(self, z, task_ids, steps: int = REASON_STEPS, top_k: int = TOP_K):
        B = z.size(0)
        task_vec = self.task_emb(task_ids)            # (B, T)
        h = torch.cat([z, task_vec], dim=-1)          # (B, D_MODEL)

        if self.num_experts == 0:
            h_self_aware, _, _, _ = self.self_awareness(
                h, torch.ones(B, 1, device=h.device)
            )
            return h_self_aware, torch.zeros(B, 1, device=h.device)

        curiosities = []
        mix_history = []

        for _ in range(steps):
            scores_1 = self.router_1(h)[:, :self.num_experts]
            scores_2 = self.router_2(h)[:, :self.num_experts]

            if self.num_experts == 1:
                mix = torch.ones(B, 1, device=h.device)
            else:
                k = min(top_k, self.num_experts)

                topv_1, topi_1 = torch.topk(scores_1, k, dim=-1)
                probs_1 = F.softmax(topv_1, dim=-1)
                mix_1 = torch.zeros(B, self.num_experts, device=h.device)
                mix_1.scatter_(1, topi_1, probs_1)

                topv_2, topi_2 = torch.topk(scores_2, k, dim=-1)
                probs_2 = F.softmax(topv_2, dim=-1)
                mix_2 = torch.zeros(B, self.num_experts, device=h.device)
                mix_2.scatter_(1, topi_2, probs_2)

                mix = (mix_1 + mix_2) / 2.0

            outs = []
            for i in range(self.num_experts):
                outs.append(self.experts[i](h).unsqueeze(-1))
            all_out = torch.cat(outs, dim=-1)
            h_next = (all_out * mix.unsqueeze(1)).sum(-1)

            h_pred = self.forward_model(h)
            curiosity, novelty, pred_error = self.intrinsic_motivation(
                h, h_next, forward_error=None
            )
            curiosities.append(curiosity)
            mix_history.append(mix)

            h = h_next

        avg_curiosity = torch.stack(curiosities, dim=0).mean(dim=0)
        avg_mix = torch.stack(mix_history, dim=0).mean(dim=0)

        h_self_aware, why_reasoning, if_quality, think_alternative = \
            self.self_awareness(h, avg_mix)
        self.last_if_quality = if_quality

        h_final = 0.7 * h + 0.3 * h_self_aware
        return h_final, avg_curiosity


class Head(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(D_MODEL, n_classes)

    def forward(self, h):
        return self.fc(h)


# ============================================================
# MODEL LOADING
# ============================================================

def build_encoder(task_name: str, vocab_size: int) -> nn.Module:
    if "TEXT" in task_name:
        return TextEncoder(vocab_size)
    if task_name in ["MNIST"]:
        return StandardMNISTEncoder()
    if task_name in ["EMNIST-LETTERS"]:
        return EMNISTEncoder()
    if task_name in ["FASHIONMNIST"]:
        return FashionMNISTEncoder()
    if task_name.startswith("CIFAR"):
        return CIFAR10Encoder()
    # Fallback
    return TextEncoder(vocab_size)


def load_cml_models(model_dir: str = DEFAULT_MODEL_DIR):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Task mapping
    mapping_path = os.path.join(model_dir, "task_name_to_id.pth")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("task_name_to_id.pth not found in model directory.")

    task_name_to_id = torch.load(mapping_path, map_location=DEVICE)
    task_id_to_name = {v: k for k, v in task_name_to_id.items()}

    # Vocab
    vocab_path = os.path.join(model_dir, "vocab.pth")
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path, map_location=DEVICE)
        vocab = SimpleVocab(
            token_to_id=vocab_data["token_to_id"],
            id_to_token=vocab_data["id_to_token"],
        )
    else:
        vocab = SimpleVocab()

    vocab_size = max(len(vocab.id_to_token), 2)

    # Brain
    brain_path = os.path.join(model_dir, "brain.pth")
    if not os.path.exists(brain_path):
        raise FileNotFoundError("brain.pth not found in model directory.")

    num_tasks = len(task_name_to_id)
    brain = SharedBrain(num_tasks).to(DEVICE)
    brain_state = torch.load(brain_path, map_location=DEVICE)

    # Ensure the correct number of experts before loading
    max_expert_idx = -1
    for k in brain_state.keys():
        if k.startswith("experts.") and ".net." in k:
            parts = k.split(".")
            try:
                idx = int(parts[1])
                max_expert_idx = max(max_expert_idx, idx)
            except ValueError:
                continue

    needed_experts = max_expert_idx + 1
    while brain.num_experts < needed_experts:
        brain.add_expert()

    brain.load_state_dict(brain_state, strict=False)
    brain.eval()

    # Encoders and heads
    encoders: Dict[str, nn.Module] = {}
    heads: Dict[str, nn.Module] = {}

    encoder_files = glob.glob(os.path.join(model_dir, "encoder_*.pth"))
    for enc_path in encoder_files:
        fname = os.path.basename(enc_path)
        task = fname.replace("encoder_", "").replace(".pth", "")
        if task not in task_name_to_id:
            continue

        encoder = build_encoder(task, vocab_size).to(DEVICE)
        state = torch.load(enc_path, map_location=DEVICE)
        encoder.load_state_dict(state, strict=False)
        encoder.eval()
        encoders[task] = encoder

    head_files = glob.glob(os.path.join(model_dir, "head_*.pth"))
    for head_path in head_files:
        fname = os.path.basename(head_path)
        task = fname.replace("head_", "").replace(".pth", "")
        if task not in task_name_to_id:
            continue

        state = torch.load(head_path, map_location=DEVICE)
        if "fc.weight" not in state:
            continue
        n_classes = state["fc.weight"].shape[0]
        head = Head(n_classes).to(DEVICE)
        head.load_state_dict(state, strict=False)
        head.eval()
        heads[task] = head

    if not encoders or not heads:
        raise RuntimeError("No encoders/heads loaded; check model directory contents.")

    return brain, encoders, heads, vocab, task_name_to_id, task_id_to_name


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path: str, task: str) -> torch.Tensor:
    img = Image.open(image_path)

    if task in ["MNIST", "EMNIST-LETTERS"]:
        img = img.convert("L")
        img = img.resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[np.newaxis, :, :]                # (1, 28, 28)
    elif task in ["FASHIONMNIST"]:
        img = img.convert("L")
        img = img.resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr[np.newaxis, :, :]
    else:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))        # (3, 32, 32)

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
    return x.to(DEVICE)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_image(
    brain: SharedBrain,
    encoders: Dict[str, nn.Module],
    heads: Dict[str, nn.Module],
    task_name_to_id: Dict[str, int],
    image_path: str,
    task: str,
) -> Dict:
    if task not in encoders or task not in heads:
        return {"error": f"Task '{task}' not available for image inference."}

    x = preprocess_image(image_path, task)
    enc = encoders[task]
    head = heads[task]
    tid = task_name_to_id[task]

    with torch.no_grad():
        z = enc(x)
        task_ids = torch.full((x.size(0),), tid, dtype=torch.long, device=DEVICE)
        h, curiosity = brain(z, task_ids)
        logits = head(h)
        probs = F.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred].item())

    return {
        "task": task,
        "prediction": pred,
        "prediction_name": CLASS_NAMES.get(task, {}).get(pred, str(pred)),
        "confidence": conf,
        "curiosity": float(curiosity.mean().item()),
        "probabilities": probs[0].cpu().numpy().tolist(),
    }


def predict_text(
    brain: SharedBrain,
    encoders: Dict[str, nn.Module],
    heads: Dict[str, nn.Module],
    vocab: SimpleVocab,
    task_name_to_id: Dict[str, int],
    text: str,
    task: str = "TEXT-SENT",
) -> Dict:
    if task not in encoders or task not in heads:
        return {"error": f"Task '{task}' not available for text inference."}

    if not vocab.token_to_id:
        return {"error": "No vocabulary loaded; text tasks are not available."}

    ids = vocab.encode(text, TEXT_MAX_LEN).unsqueeze(0).to(DEVICE)
    enc = encoders[task]
    head = heads[task]
    tid = task_name_to_id[task]

    with torch.no_grad():
        z = enc(ids)
        task_ids = torch.full((ids.size(0),), tid, dtype=torch.long, device=DEVICE)
        h, curiosity = brain(z, task_ids)
        logits = head(h)
        probs = F.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred].item())

    return {
        "task": task,
        "input": text,
        "prediction": pred,
        "prediction_name": CLASS_NAMES.get(task, {}).get(pred, str(pred)),
        "confidence": conf,
        "curiosity": float(curiosity.mean().item()),
        "probabilities": probs[0].cpu().numpy().tolist(),
    }


# ============================================================
# PRETTY PRINT HELPERS
# ============================================================

def print_text_result(result: Dict):
    if "error" in result:
        print(f"\nERROR: {result['error']}\n")
        return

    print("\n==============================")
    print("        TEXT PREDICTION       ")
    print("==============================\n")
    print(f"Task: {result['task']}")
    print(f"Input: {result.get('input', '')}\n")

    name = result.get("prediction_name", str(result["prediction"]))
    idx = result["prediction"]
    conf_pct = result["confidence"] * 100.0

    print(f"Prediction: {idx}   ({name})")
    print(f"Confidence: {conf_pct:.2f}%")
    print(f"Curiosity: {result.get('curiosity', 0.0):.4f}")
    print("\n------------------------------\n")


def print_image_result(result: Dict, image_path: str):
    if "error" in result:
        print(f"\nERROR: {result['error']}\n")
        return

    print("\n==============================")
    print("       IMAGE PREDICTION       ")
    print("==============================\n")
    print(f"Task: {result['task']}")
    print(f"Image: {image_path}\n")

    name = result.get("prediction_name", str(result["prediction"]))
    idx = result["prediction"]
    conf_pct = result["confidence"] * 100.0

    print(f"Prediction: {idx}   ({name})")
    print(f"Confidence: {conf_pct:.2f}%")
    print(f"Curiosity: {result.get('curiosity', 0.0):.4f}")
    print("\n------------------------------\n")


# ============================================================
# MAIN / CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CML-Core Inference Engine (image + text)"
    )
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to directory containing brain/encoder/head weights.")
    parser.add_argument("--image", type=str, help="Path to image file.")
    parser.add_argument("--text", type=str, help="Text input.")
    parser.add_argument("--task", type=str, help="Task name (e.g., MNIST, TEXT-IMDB).")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit.")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode if no args provided.")

    args = parser.parse_args()

    brain, encoders, heads, vocab, task_name_to_id, task_id_to_name = load_cml_models(
        args.model_dir
    )

    if args.list_tasks:
        print("\nAvailable tasks:\n")
        for t in sorted(encoders.keys()):
            head = heads.get(t, None)
            n_classes = head.fc.out_features if isinstance(head, Head) else "?"
            print(f"  {t:<20}  ({n_classes} classes)")
        print()
        return

    # Single image
    if args.image:
        if not args.task:
            print("Please specify --task for image inference (no auto-detection implemented).")
            return
        res = predict_image(brain, encoders, heads, task_name_to_id, args.image, args.task)
        print_image_result(res, args.image)
        return

    # Single text
    if args.text:
        task = args.task or "TEXT-SENT"
        res = predict_text(brain, encoders, heads, vocab, task_name_to_id, args.text, task)
        print_text_result(res)
        return

    # Interactive mode (if chosen or no arguments)
    print("\nCML-Core Inference – Interactive Mode")
    print("Type a file path for image inference, or plain text for text inference.")
    print("Use 'tasks' to list available tasks, 'quit' to exit.\n")

    while True:
        try:
            line = input(">> ").strip()
            if not line:
                continue
            if line.lower() == "quit":
                break
            if line.lower() == "tasks":
                print("\nAvailable tasks:\n")
                for t in sorted(encoders.keys()):
                    head = heads.get(t, None)
                    n_classes = head.fc.out_features if isinstance(head, Head) else "?"
                    print(f"  {t:<20}  ({n_classes} classes)")
                print()
                continue

            if os.path.exists(line):
                # Image path
                tname = input("Task name for this image >> ").strip()
                if not tname:
                    print("Task name is required for image inference.\n")
                    continue
                res = predict_image(brain, encoders, heads, task_name_to_id, line, tname)
                print_image_result(res, line)
            else:
                # Treat as text
                tname = args.task or "TEXT-SENT"
                res = predict_text(brain, encoders, heads, vocab, task_name_to_id, line, tname)
                print_text_result(res)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"\nERROR: {e}\n")


if __name__ == "__main__":
    main()
