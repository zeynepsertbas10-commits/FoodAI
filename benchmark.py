
import os
import io
import time
import platform

import numpy as np
import torch

from torch.ao.nn.quantized.dynamic import Linear

from recipe_search import (
    load_recipes,
    build_tfidf,
    build_transformer,
    normalize_query
)


# ==========================================
# SETTINGS
# ==========================================

QUERY = "chicken, rice, tomato, onion"

RUNS = 100
WARMUP = 10

QUANTIZED_MODEL_FILE = "models/minilm_quantized.pt"
QUANTIZED_EMBEDDINGS_FILE = "data/recipe_embeddings_quantized.npy"


# ==========================================
# HARDWARE INFORMATION
# ==========================================

torch.set_num_threads(1)

print("\n========== HARDWARE INFORMATION ==========")

print("Operating System:", platform.platform())
print("Processor:", platform.processor() or "Not reported")
print("Logical CPU Cores:", os.cpu_count())
print("PyTorch Version:", torch.__version__)
print("Device: CPU")
print("PyTorch CPU Threads:", torch.get_num_threads())


# ==========================================
# MODEL SIZE
# ==========================================

def get_encoder_size_mb(model):

    buffer = io.BytesIO()

    torch.save(
        model[0].auto_model,
        buffer
    )

    return buffer.getbuffer().nbytes / (1024 * 1024)


def get_file_size_mb(path):

    return os.path.getsize(path) / (1024 * 1024)


# ==========================================
# LATENCY
# ==========================================

def measure_latency(name, function):

    for _ in range(WARMUP):
        function()

    times = []

    for _ in range(RUNS):

        start = time.perf_counter()

        function()

        end = time.perf_counter()

        times.append((end - start) * 1000)

    average = float(np.mean(times))
    median = float(np.median(times))

    print(f"\n{name}")
    print(f"Average Latency: {average:.4f} ms")
    print(f"Median Latency:  {median:.4f} ms")
    print(f"Runs: {RUNS}")

    return average, median


# ==========================================
# LOAD DATA
# ==========================================

print("\n========== LOADING DATA ==========")

recipes = load_recipes()

print("Number of recipes:", len(recipes))


# ==========================================
# TF-IDF
# ==========================================

print("\n========== TF-IDF ==========")

vectorizer, tfidf_matrix = build_tfidf(recipes)


def tfidf_search():

    normalized_query, _ = normalize_query(QUERY)

    query_vector = vectorizer.transform([
        normalized_query
    ])

    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().flatten()

    return np.argsort(scores)[::-1][:5]


# ==========================================
# NORMAL TRANSFORMER
# ==========================================

print("\n========== NORMAL TRANSFORMER ==========")

transformer, transformer_embeddings = build_transformer(
    recipes
)


def transformer_search():

    normalized_query, _ = normalize_query(QUERY)

    query_embedding = transformer.encode(
        [normalized_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = np.dot(
        transformer_embeddings,
        query_embedding[0]
    )

    return np.argsort(scores)[::-1][:5]


# ==========================================
# QUANTIZED TRANSFORMER
# ==========================================

print("\n========== QUANTIZED TRANSFORMER ==========")

if not os.path.exists(QUANTIZED_MODEL_FILE):
    raise FileNotFoundError(QUANTIZED_MODEL_FILE)

if not os.path.exists(QUANTIZED_EMBEDDINGS_FILE):
    raise FileNotFoundError(QUANTIZED_EMBEDDINGS_FILE)

quantized_transformer = torch.load(
    QUANTIZED_MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

quantized_transformer.eval()

quantized_embeddings = np.load(
    QUANTIZED_EMBEDDINGS_FILE
)


def quantized_transformer_search():

    normalized_query, _ = normalize_query(QUERY)

    query_embedding = quantized_transformer.encode(
        [normalized_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = np.dot(
        quantized_embeddings,
        query_embedding[0]
    )

    return np.argsort(scores)[::-1][:5]


# ==========================================
# QUANTIZATION VERIFICATION
# ==========================================

quantized_layers = [
    module
    for module in quantized_transformer.modules()
    if isinstance(module, Linear)
]

print("\n========== QUANTIZATION VERIFICATION ==========")

print("Quantized Linear layers:", len(quantized_layers))

if quantized_layers:
    print(
        "Quantized weight dtype:",
        quantized_layers[0].weight().dtype
    )


# ==========================================
# MODEL SIZE COMPARISON
# ==========================================

print("\n========== MODEL SIZE COMPARISON ==========")

normal_size = get_encoder_size_mb(transformer)

quantized_size = get_encoder_size_mb(
    quantized_transformer
)

embedding_size = get_file_size_mb(
    QUANTIZED_EMBEDDINGS_FILE
)

print(f"Normal Transformer Encoder:    {normal_size:.2f} MB")
print(f"Quantized Transformer Encoder: {quantized_size:.2f} MB")
print(f"Quantized Recipe Embeddings:   {embedding_size:.2f} MB")

if normal_size > 0:

    reduction = (
        (normal_size - quantized_size) / normal_size
    ) * 100

    print(f"Encoder Size Reduction: {reduction:.2f}%")


# ==========================================
# LATENCY BENCHMARK
# ==========================================

print("\n========== LATENCY BENCHMARK ==========")

print("Test Query:", QUERY)
print("Warm-up Runs:", WARMUP)
print("Measurement Runs:", RUNS)

tfidf_average, tfidf_median = measure_latency(
    "TF-IDF",
    tfidf_search
)

transformer_average, transformer_median = measure_latency(
    "Normal Transformer",
    transformer_search
)

quantized_average, quantized_median = measure_latency(
    "Quantized Transformer",
    quantized_transformer_search
)


# ==========================================
# TOP-5 RESULT COMPARISON
# ==========================================

print("\n========== TOP-5 RESULT COMPARISON ==========")

normal_results = transformer_search()
quantized_results = quantized_transformer_search()

print("Normal Transformer:", normal_results.tolist())
print("Quantized Transformer:", quantized_results.tolist())

print(
    "Same Top-5 Results:",
    np.array_equal(normal_results, quantized_results)
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== FINAL BENCHMARK SUMMARY ==========")

print(
    f"{'Model':<25}"
    f"{'Size (MB)':<15}"
    f"{'Avg (ms)':<15}"
    f"{'Median (ms)':<15}"
)

print("-" * 70)

print(
    f"{'TF-IDF':<25}"
    f"{'N/A':<15}"
    f"{tfidf_average:<15.4f}"
    f"{tfidf_median:<15.4f}"
)

print(
    f"{'Transformer':<25}"
    f"{normal_size:<15.2f}"
    f"{transformer_average:<15.4f}"
    f"{transformer_median:<15.4f}"
)

print(
    f"{'Quantized Transformer':<25}"
    f"{quantized_size:<15.2f}"
    f"{quantized_average:<15.4f}"
    f"{quantized_median:<15.4f}"
)

print("\nBenchmark completed successfully!")
