# Aura Soul Engine Documentation

The **Aura Soul Engine** is a FastAPI wrapper designed for Large Language Models (LLMs) like **Qwen3-8B**. It specializes in **Steering Vectors**, allowing you to dynamically influence model behavior (concepts or "hormones") without fine-tuning. This specific version is ultra-optimized for the **NVIDIA RTX 3090 (24GB VRAM)**.

---

## 🚀 Key Features

* **Dynamic Steering:** Inject "concepts" into specific model layers during inference to shift the model's persona or style.
* **DiffMean Calibration:** Create new steering vectors on-the-fly using small sets of positive and negative text samples.
* **3090 Optimization:** Uses FP16 precision with intelligent CPU offloading and memory management to maximize the 24GB VRAM buffer.
* **Latent Extraction:** Extract hidden states (geometry) from any layer for downstream analysis.

---

## 🛠 Setup and Requirements

### Hardware Requirements

* **GPU:** NVIDIA RTX 3090 (24GB VRAM) or equivalent.
* **System RAM:** 24GB+ (recommended for CPU offloading).

### Software Dependencies

Install the required Python packages:

```bash
pip install numpy uvicorn fastapi pydantic transformers accelerate

```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `Qwen/Qwen3-8B` | The HuggingFace model ID to load. |
| `PORT` | `54213` | The internal port for the FastAPI service. |

---

## 🧠 Core Concepts: Steering Vectors

Steering vectors (referred to in the code as "Soul" or "Hormones") are derived using the **DiffMean** method.

1. **Extraction:** The engine takes "positive" samples (e.g., helpful text) and "negative" samples (e.g., unhelpful text).
2. **Calculation:** It calculates the mean difference between their hidden states at a specific layer.
3. **Injection:** During generation, this vector is added back into the model’s activations, scaled by an **intensity** factor.

---

## 📡 API Reference

### 1. Generate with Soul

**`POST /generate_with_soul`** Generates text with optional steering vector injection.

**Request Body:**
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | String | (Required) | The user input or raw chat template. |
| `vector_name`| String | `None` | The name of the saved vector to inject. |
| `intensity` | Float | `0.0` | How strongly to apply the vector (can be negative). |
| `layer_idx` | Integer| `16` | The specific model layer to apply steering. |
| `max_tokens` | Integer| `1024` | Maximum new tokens to generate. |

---

### 2. Calibrate Concept

**`POST /calibrate`** Creates and saves a new steering vector to the `/vectors` directory.

**Request Body:**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `concept_name`| String | Unique name for the new vector. |
| `positive_samples`| List | Examples demonstrating the desired trait. |
| `negative_samples`| List | Examples demonstrating the opposite trait. |
| `layer_idx` | Integer | The layer where the difference is most prominent. |

---

### 3. Latent Geometry

**`POST /embed`** Extracts the mathematical representation (embedding) of a text at a specific layer.

**Request Body:**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `text` | String | The text to analyze. |
| `layer_idx` | Integer | The layer to extract states from (-1 for last). |

---

## ⚙️ Technical Optimizations

* **Concurrency Control:** Uses an `asyncio.Semaphore` set to `1`. This ensures the GPU only processes one heavy request at a time to prevent Out-Of-Memory (OOM) errors.
* **Memory Mapping:** Loads the model with `device_map="auto"`, allocating up to 22GB to the GPU and 24GB to the CPU.
* **Hook Management:** Custom forward hooks are registered and cleared dynamically for every generation to ensure no "leaking" of concepts between different user requests.

---

> **Note:** The engine automatically handles chat formatting using the model's native `chat_template` if it detects a standard prompt.


