"""
COS30082 Assignment 1 - Bird Species Classifier (Streamlit web app)
Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).parent
BACKBONE = "resnet50"   # must be the same model that was trained in the notebook (Run C / D)

st.set_page_config(page_title="Bird Species Classifier")


# 1. LOAD THE MODEL (only once, then Streamlit keeps it in memory)
@st.cache_resource
def load_model():
    # Species names, in the same order as the labels used in training
    with open(HERE / "class_names.json") as f:
        class_names = json.load(f)

    # Same architecture as training, but without downloading pretrained weights
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(class_names))

    # Load the weights learned in the notebook (fp16 weights are converted to fp32 automatically)
    weights = torch.load(HERE / "best_model.pth", map_location="cpu")
    model.load_state_dict(weights)

    model.eval()   # evaluation mode (dropout off)

    # "Black_footed_Albatross" -> "Black footed Albatross" (nicer to read)
    nice_names = [name.replace("_", " ") for name in class_names]
    return model, nice_names


model, nice_names = load_model()

# 2. PREPROCESSING - must be the same as the evaluation transform in the notebook
preprocess = transforms.Compose([
    transforms.Resize(256),        # shorter side -> 256
    transforms.CenterCrop(224),    # take the centre 224 x 224
    transforms.ToTensor(),         # picture -> tensor with values 0..1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # ImageNet mean / std
])


# 3. PREDICTION
def predict(image):
    """Return the 5 most likely species as a list of (name, probability)."""
    image = image.convert("RGB")                 # make sure it has 3 colour channels
    x = preprocess(image).unsqueeze(0)           # add a batch dimension: (1, 3, 224, 224)

    with torch.no_grad():                        # no gradients needed for prediction
        logits = model(x)                        # raw scores for the 200 species
        probs = torch.softmax(logits, dim=1)[0]  # scores -> probabilities (sum to 1)

    top_probs, top_ids = probs.topk(5)           # the 5 highest probabilities

    results = []
    for p, i in zip(top_probs, top_ids):
        results.append((nice_names[int(i)], float(p)))
    return results


# 4. WEB PAGE

st.title("Bird Species Classifier")
st.write(
    "Upload a photo of a bird and the model will guess which of the **200 species** "
    "(Caltech-UCSD Birds 200) it is."
)

with st.sidebar:
    st.header("About this model")
    st.markdown(
        "- **Model:** ResNet50 pretrained on ImageNet, then fine-tuned on bird photos.\n"
        "- **Training data:** 4829 photos (about 24 per species).\n"
        "- **Against overfitting:** data augmentation, dropout, weight decay and early stopping.\n"
        "- It can only answer with the 200 dataset species, so any other bird is matched to the closest one."
    )
    st.caption("COS30082 Assignment 1 - Norman Ross Moneng (102782452)")

# Let the user choose between uploading a photo and using an example
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

# Show the picture and the predictions
if image is not None:
    results = predict(image)
    best_name, best_prob = results[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your photo")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Top-5 predictions")
        # Be honest when the model is not sure
        if best_prob < 0.30:
            st.warning("Low confidence. The model is unsure about this photo.")
        else:
            st.success(f"Most likely: **{best_name}** ({best_prob:.0%} confidence)")

        for name, prob in results:
            st.write(f"{name} - {prob:.1%}")
            st.progress(min(prob, 1.0))
