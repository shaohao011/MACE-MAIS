# MACE-MAIS

This is the official PyTorch implementation of our paper:  
**"MACE-MAIS: An Interpretable Multimodal AI System for Robust Prediction of Major Adverse Cardiovascular Events"**

---

## Overview

MACE-MAIS is an end-to-end, interpretable multimodal AI system designed for predicting Major Adverse Cardiovascular Events (MACE) by integrating Cardiovascular Magnetic Resonance (CMR) and Electronic Health Record (EHR) data. To achieve robust performance, the system addresses missing data modalities and ensures predictions are accompanied by clinically relevant explanations.

---


## Setup

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shaohao011/MACE-MAIS.git
   cd R1_CARDIO
2. Set up your environment and install dependencies:
    ```bash
    pip install -r requirements.txt

### Data Preparation
Prepare your data and organize it under the data/ directory. Ensure that the data structure matches the expected format defined in the preprocessing scripts.

### Pretraining
To run the pretraining step:
```bash
cd Pre-train
bash do_pretrain.sh