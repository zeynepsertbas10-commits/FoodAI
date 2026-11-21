
import os
import re
import json

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd
import numpy as np

from torchvision import models, transforms
from torch.ao.quantization import quantize_dynamic
from PIL import Image

from recipe_search import normalize_query


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="FoodAI",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ FoodAI - Intelligent Recipe Finder")

st.write(
    "Upload a food image, understand the model's prediction "
    "and discover recipes using AI."
)


# ==========================================
# 2. FILE PATHS
# ==========================================

VGG_MODEL_PATH = "models/vgg16_quantized.pth"
TRANSFORMER_MODEL_PATH = "models/minilm_quantized.pt"

RECIPES_PATH = "data/recipes.json"
EMBEDDINGS_PATH = "data/recipe_embeddings_quantized.npy"

TRAIN_PATH = "data/images/train"


# ==========================================
# 3. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. RECIPE DATA CLEANING
# ==========================================

def clean_recipe_items(value):

    if isinstance(value, str):
        raw_items = [value]

    elif isinstance(value, list):
        raw_items = value

    else:
        return []

    cleaned_items = []

    for item in raw_items:

        parts = re.split(
            r"<extra_id_\d+>|\n",
            str(item)
        )

        for part in parts:

            part = re.sub(
                r"<extra_id_\d+>",
                " ",
                part
            )

            part = re.sub(
                r"\s+",
                " ",
                part
            ).strip(" ,;")

            if part:
                cleaned_items.append(part)

    return cleaned_items


def clean_recipe(recipe):

    return {
        "id": recipe.get("id", 0),
        "title": str(recipe.get("title", "")).strip(),
        "ingredients": clean_recipe_items(
            recipe.get("ingredients", [])
        ),
        "directions": clean_recipe_items(
            recipe.get("directions", [])
        )
    }


# ==========================================
# 5. LOAD QUANTIZED VGG16
# ==========================================

@st.cache_resource
def load_image_model():

    if not os.path.exists(VGG_MODEL_PATH):
        raise FileNotFoundError(
            f"VGG16 model not found: {VGG_MODEL_PATH}"
        )

    from torchvision.datasets import ImageFolder

    classes = ImageFolder(TRAIN_PATH).classes

    model = models.vgg16(weights=None)

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        len(classes)
    )

    model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    state_dict = torch.load(
        VGG_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, classes


# ==========================================
# 6. LOAD QUANTIZED TRANSFORMER
# ==========================================

@st.cache_resource
def load_recipe_model():

    required_files = [
        TRANSFORMER_MODEL_PATH,
        RECIPES_PATH,
        EMBEDDINGS_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    with open(
        RECIPES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        raw_recipes = json.load(file)

    recipes = [
        clean_recipe(recipe)
        for recipe in raw_recipes
    ]

    model = torch.load(
        TRANSFORMER_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    embeddings = np.load(EMBEDDINGS_PATH)

    if len(recipes) != len(embeddings):
        raise ValueError(
            "Recipe and embedding counts do not match."
        )

    return recipes, model, embeddings


# ==========================================
# 7. QUANTIZED CLASSIFIER FOR GRAD-CAM
# ==========================================

def classifier_with_gradients(model, feature_maps):

    x = model.avgpool(feature_maps)
    x = torch.flatten(x, 1)

    for layer in model.classifier:

        if isinstance(layer, nnqd.Linear):

            weight = layer.weight().dequantize()
            bias = layer.bias()

            x = F.linear(
                x,
                weight,
                bias
            )

        else:
            x = layer(x)

    return x


# ==========================================
# 8. GENERATE GRAD-CAM
# ==========================================

def generate_gradcam(model, tensor, class_index):

    model.eval()
    model.zero_grad()

    feature_maps = model.features(tensor)
    feature_maps.retain_grad()

    logits = classifier_with_gradients(
        model,
        feature_maps
    )

    class_score = logits[0, class_index]
    class_score.backward()

    gradients = feature_maps.grad

    if gradients is None:
        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * feature_maps).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = F.interpolate(
        cam,
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    )

    return cam.squeeze().detach().cpu().numpy()


# ==========================================
# 9. CREATE GRAD-CAM OVERLAY
# ==========================================

def create_gradcam_overlay(original_image, cam):

    width, height = original_image.size

    heatmap = Image.fromarray(
        np.uint8(cam * 255)
    ).resize((width, height))

    heatmap_array = (
        np.array(heatmap).astype(np.float32) / 255.0
    )

    colored_heatmap = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    colored_heatmap[:, :, 0] = np.uint8(
        np.clip(heatmap_array * 255, 0, 255)
    )

    colored_heatmap[:, :, 1] = np.uint8(
        np.clip(heatmap_array * 100, 0, 255)
    )

    colored_heatmap[:, :, 2] = 0

    colored_heatmap = Image.fromarray(
        colored_heatmap
    )

    overlay = Image.blend(
        original_image.convert("RGB"),
        colored_heatmap,
        alpha=0.45
    )

    return heatmap, overlay


# ==========================================
# 10. TRANSFORMER RECIPE SEARCH
# ==========================================

def search_recipes(query, recipes, model, embeddings):

    normalized_query, constraints = normalize_query(query)

    if not normalized_query.strip():
        normalized_query = query.strip()

    query_embedding = model.encode(
        [normalized_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(
        embeddings,
        query_embedding[0]
    )

    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for index in top_indices:

        results.append({
            "recipe": recipes[index],
            "score": float(scores[index])
        })

    return results, normalized_query, constraints


# ==========================================
# 11. DISPLAY TOP-5 RECIPES
# ==========================================

def display_recipes(results):

    st.subheader("🍽️ Top 5 Recipe Recommendations")

    for rank, result in enumerate(results, start=1):

        recipe = result["recipe"]
        score = result["score"]

        with st.expander(
            f"{rank}. {recipe['title']} "
            f"— Similarity: {score:.4f}",
            expanded=(rank == 1)
        ):

            st.write(
                "**Similarity Score:**",
                f"{score:.4f}"
            )

            st.write("**Ingredients:**")

            for ingredient in recipe["ingredients"]:
                st.write(f"- {ingredient}")

            st.write("**Preparation:**")

            for step_number, step in enumerate(
                recipe["directions"],
                start=1
            ):

                st.write(f"{step_number}. {step}")


# ==========================================
# 12. USER INTERFACE
# ==========================================

st.header("📷 Food Image Recognition")

uploaded = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food Image",
        width=350
    )


st.divider()

st.header("🔎 Recipe Search")

query = st.text_input(
    "Describe the ingredients or recipe you want:",
    placeholder="Example: chicken, rice, tomato and onion"
)

find_button = st.button(
    "Find Recipes",
    type="primary"
)


# ==========================================
# 13. RUN FOODAI
# ==========================================

if find_button:

    if uploaded is None and not query.strip():

        st.warning(
            "Please upload a food image or enter a recipe request."
        )

    else:

        try:

            # IMAGE CLASSIFICATION AND GRAD-CAM

            if uploaded is not None:

                with st.spinner("Analyzing food image..."):

                    image_model, classes = load_image_model()

                    tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():

                        output = image_model(tensor)

                        probabilities = torch.softmax(
                            output,
                            dim=1
                        )

                    confidence, prediction = probabilities.max(1)

                    food = classes[prediction.item()]

                    cam = generate_gradcam(
                        image_model,
                        tensor,
                        prediction.item()
                    )

                    heatmap, overlay = create_gradcam_overlay(
                        image,
                        cam
                    )

                st.success(
                    f"Detected Food: {food.replace('_', ' ').title()}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence.item() * 100:.2f}%"
                )

                st.subheader("🔥 Grad-CAM Visualization")

                st.write(
                    "The heatmap highlights image regions "
                    "that contributed to the model prediction."
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        heatmap,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay",
                        use_container_width=True
                    )

            # RECIPE RETRIEVAL

            if query.strip():

                with st.spinner(
                    "Searching recipes with Quantized Transformer..."
                ):

                    recipes, recipe_model, embeddings = (
                        load_recipe_model()
                    )

                    results, normalized_query, constraints = (
                        search_recipes(
                            query,
                            recipes,
                            recipe_model,
                            embeddings
                        )
                    )

                st.divider()

                st.subheader("🧠 Query Processing")

                st.write(
                    "**Normalized Query:**",
                    normalized_query
                )

                st.write(
                    "**Maximum Cooking Time:**",
                    constraints["max_time"]
                    if constraints["max_time"] is not None
                    else "Not specified"
                )

                st.write(
                    "**Dietary Preferences:**",
                    ", ".join(constraints["diet"])
                    if constraints["diet"]
                    else "Not specified"
                )

                display_recipes(results)

            elif uploaded is not None:

                st.info(
                    "Enter ingredients or a recipe request "
                    "to receive the Top-5 recommendations."
                )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )
