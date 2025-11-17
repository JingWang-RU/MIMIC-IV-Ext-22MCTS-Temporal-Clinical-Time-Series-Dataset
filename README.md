# MIMIC-IV-EXT-22MCTS: A 22 Million-Event Temporal Clinical Time-Series Dataset with Relative Timestamps

## 📘 Overview

**MIMIC-IV-22MCTS** is a large-scale temporal clinical dataset constructed from the MIMIC-IV database, featuring **22 million clinical events** annotated with **relative timestamps**. The dataset enables a wide range of temporal clinical modeling, including patient trajectory modeling, clinical event prediction, and time-sensitive language modeling.

Please kindly consider cite our work if you feel the dataset is useful, thank you so much for your interest, best luck to your project:

- Wang, J., Niu, X., Zhang, T., Shen, J., Kim, J., & Weiss, J. (2025). 
 MIMIC-IV-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp 
 (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dkj6-r828
- @article{wang2025mimic,
   - title={MIMIC-$\backslash$RNum $\{$4$\}$-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp for Risk Prediction},
   - author={Wang, Jing and Niu, Xing and Kim, Juyong and Shen, Jie and Zhang, Tong and Weiss, Jeremy C},
   - journal={arXiv preprint arXiv:2505.00827},
  - year={2025}
 - }


This repository contains:
- Scripts for dataset preprocessing and context generation
- Downstream task evaluation pipelines
- Integration with LLMs (e.g., LLaMA) for clinical reasoning
- HPC-ready scripts for scalable execution

---

## 📁 Project Structure

. ├── data/ # [physionet.com](https://physionet.org/content/mimic-iv-ext-22mcts/1.0.0/) ├── postannote_.py # Python scripts for downstream tasks (e.g., prediction, evaluation) ├── run_.sbatch # SLURM job scripts for HPC environment ├── annotation # LLM interaction code (e.g., prompt building, model inference) 

## 📊 Dataset Description

- **Source**: Derived from MIMIC-IV-EXT-22MCTS clinical database from physionet.com https://physionet.org/content/mimic-iv-ext-22mcts/1.0.0/ 
- **Size**: 22 million timestamped clinical events clinical_event_timestamp.csv
- **Siize**: 1 million version after normalization with normalization_LLM_ft_generation clinical_event_timestamp_1M.csv
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

