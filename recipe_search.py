
import ast
import json
import os
import re
import copy

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.ao.quantization import quantize_dynamic
from nltk.stem import PorterStemmer


# ==========================================
# 1. SETTINGS
# ==========================================

DATA_DIR = "data"
MODEL_DIR = "models"

RECIPES_FILE = os.path.join(DATA_DIR, "recipes.json")
TEST_FILE = "retrieval_tests.json"

QUANTIZED_MODEL_FILE = os.path.join(
    MODEL_DIR, "minilm_quantized.pt"
)
QUANTIZED_EMBEDDINGS_FILE = os.path.join(
    DATA_DIR, "recipe_embeddings_quantized.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RECIPE_LIMIT = 500

stemmer = PorterStemmer()

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. TEXT CLEANING AND NORMALIZATION
# ==========================================

def clean_text(text):

    if isinstance(text, list):
        text = " ".join(str(item) for item in text)

    if not isinstance(text, str):
        return ""

    text = re.sub(r"<extra_id_\d+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def stem_text(text):

    words = clean_text(text).split()

    return " ".join(
        stemmer.stem(word) for word in words
    )


# ==========================================
# 3. QUERY CONSTRAINT EXTRACTION
# ==========================================

def extract_constraints(query):

    text = query.lower()

    constraints = {
        "max_time": None,
        "diet": []
    }

    time_patterns = [
        r"(?:under|less than|within|in)\s+(\d+)\s*(?:minutes|minute|mins|min)\b",
        r"(\d+)\s*(?:minutes|minute|mins|min)\b"
    ]

    for pattern in time_patterns:

        match = re.search(pattern, text)

        if match:
            constraints["max_time"] = int(match.group(1))
            break

    diet_patterns = {
        "vegan": r"\bvegan\b",
        "vegetarian": r"\bvegetarian\b",
        "gluten-free": r"\bgluten[\s-]*free\b",
        "dairy-free": r"\bdairy[\s-]*free\b",
        "keto": r"\bketo\b"
    }

    for diet, pattern in diet_patterns.items():

        if re.search(pattern, text):
            constraints["diet"].append(diet)

    return constraints


def normalize_query(query):

    constraints = extract_constraints(query)

    normalized_query = query.lower()

    normalized_query = re.sub(
        r"(?:under|less than|within|in)\s+\d+\s*(?:minutes|minute|mins|min)\b",
        " ",
        normalized_query
    )

    normalized_query = re.sub(
        r"\d+\s*(?:minutes|minute|mins|min)\b",
        " ",
        normalized_query
    )

    for diet in [
        "vegan",
        "vegetarian",
        "gluten free",
        "gluten-free",
        "dairy free",
        "dairy-free",
        "keto"
    ]:

        normalized_query = re.sub(
            re.escape(diet),
            " ",
            normalized_query
        )

    normalized_query = clean_text(normalized_query)

    return normalized_query, constraints


# ==========================================
# 4. INGREDIENT PARSING
# ==========================================

def parse_list(value):

    if isinstance(value, list):

        raw_items = []

        for item in value:
            raw_items.extend(
                re.split(r"<extra_id_\d+>|\n", str(item))
            )

    elif isinstance(value, str):

        value = value.strip()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):

                raw_items = []

                for item in parsed:
                    raw_items.extend(
                        re.split(r"<extra_id_\d+>|\n", str(item))
                    )

            else:
                raw_items = re.split(
                    r"<extra_id_\d+>|\n",
                    value
                )

        except (ValueError, SyntaxError):

            raw_items = re.split(
                r"<extra_id_\d+>|\n",
                value
            )

    else:
        return []

    cleaned_items = []

    for item in raw_items:

        item = re.sub(r"<extra_id_\d+>", " ", str(item))
        item = re.sub(r"\s+", " ", item)
        item = item.strip(" ,;")

        if item:
            cleaned_items.append(item)

    return cleaned_items


def normalize_recipe(recipe):

    return {
        "id": recipe.get("id", 0),
        "title": str(recipe.get("title", "")).strip(),
        "ingredients": parse_list(
            recipe.get("ingredients", [])
        ),
        "directions": parse_list(
            recipe.get("directions", [])
        )
    }


# ==========================================
# 5. LOAD RECIPES
# ==========================================

def load_recipes():

    if os.path.exists(RECIPES_FILE):

        print("Loading existing recipes...")

        with open(RECIPES_FILE, "r", encoding="utf-8") as file:
            recipes = json.load(file)

        recipes = [
            normalize_recipe(recipe)
            for recipe in recipes
        ]

        print(f"{len(recipes)} recipes loaded and cleaned.")

        return recipes

    print("Downloading recipe dataset...")

    dataset = load_dataset(
        "parquet",
        data_files=(
            "https://huggingface.co/datasets/"
            "EmTpro01/recipe-nlg-50k/resolve/main/"
            "data/train-00000-of-00001.parquet"
        ),
        split="train"
    )

    recipes = []

    for item in dataset:

        recipe = normalize_recipe(item)

        if not recipe["title"] or not recipe["ingredients"]:
            continue

        recipe["id"] = len(recipes)
        recipes.append(recipe)

        if len(recipes) >= RECIPE_LIMIT:
            break

    with open(RECIPES_FILE, "w", encoding="utf-8") as file:

        json.dump(
            recipes,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"{len(recipes)} recipes saved.")

    return recipes


def get_recipe_texts(recipes):

    return [
        clean_text(
            recipe["title"] + " " +
            " ".join(recipe["ingredients"])
        )
        for recipe in recipes
    ]


# ==========================================
# 6. TF-IDF MODEL
# ==========================================

def build_tfidf(recipes):

    recipe_texts = []

    for recipe in recipes:

        text = (
            recipe["title"] + " " +
            " ".join(recipe["ingredients"])
        )

        recipe_texts.append(stem_text(text))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(recipe_texts)

    return vectorizer, tfidf_matrix


# ==========================================
# 7. NORMAL TRANSFORMER MODEL
# ==========================================

def build_transformer(recipes):

    print("Loading normal Transformer model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    recipe_texts = get_recipe_texts(recipes)

    print("Calculating normal Transformer embeddings...")

    embeddings = model.encode(
        recipe_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return model, embeddings


# ==========================================
# 8. QUANTIZED TRANSFORMER MODEL
# ==========================================

def build_quantized_transformer(recipes):

    print("\nLoading model for quantization...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    model.eval()

    print("Applying INT8 dynamic quantization...")

    # Quantize the Transformer neural network.
    model[0].auto_model = quantize_dynamic(
        model[0].auto_model,
        {nn.Linear},
        dtype=torch.qint8
    )

    model.eval()

    print("Quantized Transformer model created.")

    # Save the quantized model.
    torch.save(
        model,
        QUANTIZED_MODEL_FILE
    )

    print("Quantized model saved:", QUANTIZED_MODEL_FILE)

    recipe_texts = get_recipe_texts(recipes)

    print("Calculating quantized recipe embeddings...")

    embeddings = model.encode(
        recipe_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(
        QUANTIZED_EMBEDDINGS_FILE,
        embeddings
    )

    print(
        "Quantized embeddings saved:",
        QUANTIZED_EMBEDDINGS_FILE
    )

    return model, embeddings


# ==========================================
# 9. TF-IDF SEARCH
# ==========================================

def search_tfidf(
    query,
    recipes,
    vectorizer,
    tfidf_matrix,
    top_k=5
):

    normalized_query, constraints = normalize_query(query)

    query_vector = vectorizer.transform([
        stem_text(normalized_query)
    ])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:

        results.append({
            "recipe": recipes[index],
            "score": float(scores[index])
        })

    return results


# ==========================================
# 10. TRANSFORMER SEARCH
# ==========================================

def search_transformer(
    query,
    recipes,
    model,
    embeddings,
    top_k=5
):

    normalized_query, constraints = normalize_query(query)

    query_embedding = model.encode(
        [normalized_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(
        embeddings,
        query_embedding[0]
    )

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:

        results.append({
            "recipe": recipes[index],
            "score": float(scores[index])
        })

    return results


# ==========================================
# 11. RETRIEVAL EVALUATION
# ==========================================

def calculate_metrics(
    queries,
    recipes,
    search_function
):

    recalls = []
    mrrs = []

    available_ids = {
        recipe["id"] for recipe in recipes
    }

    for test in queries:

        query = test["query"]
        relevant_ids = set(test["relevant_recipe_ids"])

        if not relevant_ids or not relevant_ids.issubset(available_ids):
            continue

        results = search_function(query)

        result_ids = [
            result["recipe"]["id"]
            for result in results
        ]

        hits = [
            recipe_id
            for recipe_id in result_ids
            if recipe_id in relevant_ids
        ]

        recalls.append(
            len(hits) / len(relevant_ids)
        )

        rank = next(
            (
                position
                for position, recipe_id in enumerate(
                    result_ids, start=1
                )
                if recipe_id in relevant_ids
            ),
            None
        )

        mrrs.append(
            1 / rank if rank else 0
        )

    if not recalls:
        return 0.0, 0.0

    return float(np.mean(recalls)), float(np.mean(mrrs))


def evaluate_retrieval(
    recipes,
    vectorizer,
    tfidf_matrix,
    transformer,
    embeddings,
    quantized_transformer,
    quantized_embeddings
):

    if not os.path.exists(TEST_FILE):

        print("\nTest file not found. Evaluation skipped.")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    queries = test_data["queries"]

    if not queries:

        print("\nNo test queries found.")
        return

    tfidf_search = lambda q: search_tfidf(
        q, recipes, vectorizer, tfidf_matrix
    )

    transformer_search = lambda q: search_transformer(
        q, recipes, transformer, embeddings
    )

    quantized_search = lambda q: search_transformer(
        q, recipes, quantized_transformer,
        quantized_embeddings
    )

    tfidf_recall, tfidf_mrr = calculate_metrics(
        queries, recipes, tfidf_search
    )

    transformer_recall, transformer_mrr = calculate_metrics(
        queries, recipes, transformer_search
    )

    quantized_recall, quantized_mrr = calculate_metrics(
        queries, recipes, quantized_search
    )

    print("\n========== RETRIEVAL EVALUATION ==========")
    print("Number of test queries:", len(queries))

    print("\nTF-IDF:")
    print(f"Recall@5: {tfidf_recall:.4f}")
    print(f"MRR:      {tfidf_mrr:.4f}")

    print("\nTransformer:")
    print(f"Recall@5: {transformer_recall:.4f}")
    print(f"MRR:      {transformer_mrr:.4f}")

    print("\nQuantized Transformer:")
    print(f"Recall@5: {quantized_recall:.4f}")
    print(f"MRR:      {quantized_mrr:.4f}")


# ==========================================
# 12. MODEL SIZE
# ==========================================

def show_model_size():

    if os.path.exists(QUANTIZED_MODEL_FILE):

        size_mb = (
            os.path.getsize(QUANTIZED_MODEL_FILE)
            / (1024 * 1024)
        )

        print("\n========== QUANTIZED MODEL SIZE ==========")
        print(f"Quantized Transformer: {size_mb:.2f} MB")


# ==========================================
# 13. DISPLAY RESULTS
# ==========================================

def show_results(results):

    for rank, result in enumerate(results, start=1):

        recipe = result["recipe"]

        print(f"\n{rank}. {recipe['title']}")
        print(f"Similarity Score: {result['score']:.4f}")

        print("Ingredients:")

        for ingredient in recipe["ingredients"][:10]:
            print(f" - {ingredient}")


# ==========================================
# 14. MAIN PROGRAM
# ==========================================

def main():

    recipes = load_recipes()

    if len(recipes) < 200:

        print("ERROR: At least 200 recipes are required.")
        return

    print("\nPreparing TF-IDF model...")

    vectorizer, tfidf_matrix = build_tfidf(recipes)

    print("\nPreparing normal Transformer...")

    transformer, embeddings = build_transformer(recipes)

    print("\nPreparing Quantized Transformer...")

    quantized_transformer, quantized_embeddings = (
        build_quantized_transformer(recipes)
    )

    show_model_size()

    evaluate_retrieval(
        recipes,
        vectorizer,
        tfidf_matrix,
        transformer,
        embeddings,
        quantized_transformer,
        quantized_embeddings
    )

    print("\nFoodAI Recipe Search is ready!")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your ingredients: ")

        if query.lower().strip() == "exit":

            print("FoodAI Recipe Search closed.")
            break

        if not query.strip():

            print("Please enter at least one ingredient.")
            continue

        normalized_query, constraints = normalize_query(query)

        print("\n========== QUERY PROCESSING ==========")
        print("Normalized Query:", normalized_query)
        print("Maximum Cooking Time:", constraints["max_time"])
        print("Dietary Preferences:", constraints["diet"])

        print("\n========== TF-IDF RESULTS ==========")

        tfidf_results = search_tfidf(
            query,
            recipes,
            vectorizer,
            tfidf_matrix
        )

        show_results(tfidf_results)

        print("\n======= TRANSFORMER RESULTS =======")

        transformer_results = search_transformer(
            query,
            recipes,
            transformer,
            embeddings
        )

        show_results(transformer_results)

        print("\n======= QUANTIZED TRANSFORMER RESULTS =======")

        quantized_results = search_transformer(
            query,
            recipes,
            quantized_transformer,
            quantized_embeddings
        )

        show_results(quantized_results)


if __name__ == "__main__":
    main()
