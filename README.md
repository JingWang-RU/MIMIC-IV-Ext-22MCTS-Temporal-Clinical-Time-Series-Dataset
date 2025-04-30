# MIMIC-IV-EXT-22MCTS: A 22 Million-Event Temporal Clinical Time-Series Dataset with Relative Timestamps

## 📘 Overview

**MIMIC-IV-22MCTS** is a large-scale temporal clinical dataset constructed from the MIMIC-IV database, featuring **22 million clinical events** annotated with **relative timestamps**. The dataset enables a wide range of temporal clinical modeling, including patient trajectory modeling, clinical event prediction, and time-sensitive language modeling.

This repository contains:
- Scripts for dataset preprocessing and context generation
- Downstream task evaluation pipelines
- Integration with LLMs (e.g., LLaMA) for clinical reasoning
- HPC-ready scripts for scalable execution

---

## 📁 Project Structure

. ├── data/ # physionet.com ├── postannote_.py # Python scripts for downstream tasks (e.g., prediction, evaluation) ├── run_.sbatch # SLURM job scripts for HPC environment ├── llm_generation/ # LLM interaction code (e.g., prompt building, model inference) ├── utils/ # Utility functions and helpers ├── 

## 📊 Dataset Description

- **Source**: Derived from MIMIC-IV-EXT-22MCTS clinical database from physionet.com
- **Size**: 22 million timestamped clinical events
- **Annotations**:
  - Clinical events (labs, meds, diagnoses)
  - Relative timestamps (e.g., "Day 1", "Hour 3.5 since admission")
- **Uses**:
  - Temporal reasoning
  - Forecasting tasks (e.g., next medication)
  - Event generation using large language models

---

## 🧠 Downstream Tasks

Scripts starting with `postannote_*.py` are for evaluating specific downstream tasks, including:

- `postannote_*.py`: Fine-tuning BERT and GPT

