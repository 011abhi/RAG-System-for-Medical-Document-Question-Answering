# RAG System for Biomedical Question Answering

---

## Overview

A fully local Retrieval-Augmented Generation (RAG) pipeline evaluated across 18 experimental configurations on PubMedQA. No cloud, no API  runs entirely  locally

---

## Key Findings

- 73.3% of no-retrieval outputs falsely referenced non-existent source material
- McNemar p=0.0192 — RAG produces statistically significant improvement
- 55.3% of errors occurred despite correct evidence being retrieved — generation is the bottleneck
- Fixed-size chunking outperforms alternatives by 6-7 percentage points consistently
- BioMistral achieves 18-20pp higher accuracy than Mistral across all 18 conditions

---

## What Was Evaluated

- Chunking: Fixed-size (256 tokens, 128 stride), Sentence-level (NLTK), Semantic (BioBERT)
- Retrieval: BM25 sparse, BioBERT dense (ChromaDB), Hybrid Reciprocal Rank Fusion
- Models: Mistral-7B-Instruct (MLX), BioMistral-7B (llama-cpp GGUF)
- Dataset: PubMedQA PQA-L — 300 test instances

---

## Run the Demo

pip install streamlit
streamlit run app.py

Opens at http://localhost:8501

---

## Project Structure

chunkers/            — fixed, sentence, semantic chunking
retrievers/          — BM25, dense, hybrid RRF
generation/          — prompt building and model inference
evaluation/          — metrics, harness, grounding score
data/                — PubMedQA loader
results/             — all experimental CSVs
app.py               — Streamlit demo dashboard
run_experiment.py    — main experiment runner

---

## Hardware
Mistral via MLX, 4-bit quantisation
BioMistral via llama-cpp-python, GGUF Q4_K_M
30-40 hours total compute for all 18 conditions
