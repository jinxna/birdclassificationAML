"""
COS30082 Assignment 1 - Bird Species Classifier (Streamlit web app)

Upload a bird photo (or pick an example) and the app shows:
  1. the top-5 predicted species with confidence scores, and
  2. a Grad-CAM heatmap showing which part of the photo the model looked at.

Files needed next to this script:
  - best_model.pth     (weights saved by the training notebook; fp32 or fp16 both work)
  - class_names.json   (the 200 species names, in label order)
  - examples/          (a few sample photos)

Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).parent
BACKBONE = "resnet50"           # must be the same backbone used when training (Run C / D)

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦", layout="wide")


# ---------------------------------------------------------------------------
# 1. LOAD THE MODEL ONCE (st.cache_resource keeps it in memory between button clicks)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    class_names = json.load(open(HERE / "class_names.json"))
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(class_names), drop_rate=0.3)
    state = torch.load(HERE / "best_model.pth", map_location="cpu")
    model.load_state_dict(state)            # fp16 weights are converted to fp32 automatically here
    model.eval()                            # inference mode (dropout off)
    return model, [n.replace("_", " ") for n in class_names]


model, pretty_names = load_model()

# ---------------------------------------------------------------------------
# 2. PREPROCESSING - exactly the same as the evaluation transform in the notebook
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),                 # shorter side -> 256
    transforms.CenterCrop(224),             # centre 224 x 224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
crop_only = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])   # for display


# ---------------------------------------------------------------------------
# 3. GRAD-CAM (same idea as in the notebook)
# ---------------------------------------------------------------------------
def grad_cam(x):
    """Return (heatmap 0..1 of size 224x224, logits) for one image tensor of shape (1, 3, 224, 224)."""
    feats = model.forward_features(x)               # feature maps of the last conv stage (1, 2048, 7, 7)
    feats.retain_grad()                             # keep their gradient after backward()
    logits = model.forward_head(feats)              # pooling + final layer
    model.zero_grad()
    logits[0, logits.argmax(1).item()].backward()   # gradient of the predicted class score

    weights = feats.grad.mean(dim=(2, 3), keepdim=True)          # importance of each feature map
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy(), logits.detach()


def predict(image):
    """Return (top-5 list of (species, probability), heatmap overlay as uint8 array)."""
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0)
    cam, logits = grad_cam(x)

    probs = torch.softmax(logits, dim=1)[0]         # scores -> probabilities that sum to 1
    top_p, top_i = probs.topk(5)
    top5 = [(pretty_names[int(i)], float(p)) for p, i in zip(top_p, top_i)]

    # Blend the heatmap (jet colours) over the cropped photo
    base = np.array(crop_only(image)).astype(np.float32) / 255
    heat = cm.jet(cam)[..., :3].astype(np.float32)
    overlay = (0.55 * base + 0.45 * heat).clip(0, 1)
    return top5, (overlay * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. WEB PAGE LAYOUT
# ---------------------------------------------------------------------------
st.title("🐦 Bird Species Classifier")
st.write(
    "Upload a photo of a bird and the model will guess which of the **200 species** "
    "(Caltech-UCSD Birds 200) it is. The heatmap (Grad-CAM) shows where the model looked: "
    "**red = most important area**."
)

with st.sidebar:
    st.header("About this model")
    st.markdown(
        "- **Model:** ResNet50 pretrained on ImageNet, fine-tuned in two phases (new head first, then the whole network).\n"
        "- **Training data:** 4829 photos from Caltech-UCSD Birds 200 (about 24 per species).\n"
        "- **Against overfitting:** strong data augmentation, dropout, weight decay, label smoothing and early stopping.\n"
        "- **Grad-CAM:** shows which image regions most increased the score of the predicted species.\n"
        "- Only the 200 dataset species can be predicted, so any other bird is forced into the closest match."
    )
    st.caption("COS30082 Assignment 1 - Norman Ross Moneng (102782452)")

source = st.radio("Choose an image", ["Upload a photo", "Try an example photo"], horizontal=True)

image = None
if source == "Upload a photo":
    uploaded = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
else:
    example_files = sorted((HERE / "examples").glob("*.jpg"))
    if example_files:
        choice = st.selectbox("Example photo (from the test set)", example_files, format_func=lambda p: p.name)
        image = Image.open(choice)

if image is not None:
    top5, overlay = predict(image)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Your photo")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Top-5 predictions")
        # Honest feedback: if the model is not confident, say so instead of pretending
        if top5[0][1] < 0.30:
            st.warning("Low confidence. The model is unsure - this may not be one of the 200 species, or the photo is hard.")
        else:
            st.success(f"Most likely: **{top5[0][0]}** ({top5[0][1]:.0%} confidence)")
        for name, prob in top5:
            st.write(f"{name} - {prob:.1%}")
            st.progress(min(prob, 1.0))

    with col3:
        st.subheader("Where the model looked")
        st.image(overlay, width="stretch", caption="Grad-CAM heatmap (red = most important)")
