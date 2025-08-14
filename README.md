# MACE-MAIS

**Official PyTorch implementation of**  
**"An Interpretable Multimodal AI System for Predicting Major Adverse Cardiovascular Events from Comprehensive Patient Profiles
"**

---

## 🚀 Overview

**MACE-MAIS** is an end-to-end **interpretable multimodal AI system** for predicting **Major Adverse Cardiovascular Events (MACE)**. It integrates:

- **Cardiovascular Magnetic Resonance (CMR)** imaging
- **Electronic Health Records (EHR)**

Key features:

- Handles **missing modalities** robustly
- Provides **clinically meaningful explanations**

---

## 🔧 Setup

### 📋 Prerequisites

- Python ≥ 3.9
- NVIDIA GPU + CUDA (optional, but recommended)

### 🛠 Installation

```bash
# 1. Clone the repository
git clone https://github.com/shaohao011/MACE-MAIS.git
cd MACE-MAIS

# 2. Create and activate conda environment
conda create -n mace-mais python=3.9
conda activate mace-mais

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📂 Usage Guide

### 1️⃣ Data Preparation

Place your input data under the `data/` directory. Follow the format specified in the preprocessing scripts.

```bash
# Split survival intervals
python survival_dst_make.py
```

---

### 2️⃣ Reasoning Model (LRM)

```bash
# Generate SFT data for LRM
python utils/gen_mace_cot.py

# Train LLM using LLaMA-Factory
cd LLaMA-Factory
bash train_llama.sh
```

---

### 3️⃣ CMR Image Pretraining

```bash
# Step 1: Run pretraining
cd Pre-train
bash do_pretrain.sh

# Step 2: Extract embeddings
python Pre-train/utils/get_embedding.py
```

---

### 4️⃣ Survival Analysis

```bash
# Train and evaluate
bash do_train_survival.sh

# Tip: Set max_epochs=-1 for testing only
```

---


