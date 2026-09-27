# 🩺 MedTrain: Medical LLM Fine-Tuning & Clinical Database Pipeline

This directory contains everything required to train/fine-tune a medical model, manage the clinical SQLite database, and export to **GGUF** format (`model.gguf`) for on-device Flutter inference.

---

## 📁 Directory Structure

```
medtrain/
├── medical_db.py        # SQLite Database Manager implementing the patient schema
├── test_db_and_ai.py    # Test suite for database operations & AI DSL parsing
├── dataset.jsonl        # Your training dataset (empty file ready for your JSONL data)
├── model_loader.py      # Loads base model, tokenizer, and configures LoRA / merging
├── dataset_loader.py    # Reads JSONL data, formats ChatML prompts, handles fallbacks
├── convert_to_gguf.py   # Converts merged SafeTensors to model.gguf
├── train.py             # Main end-to-end training orchestrator CLI
├── requirements.txt     # Python package dependencies
├── med.ipynb            # Google Colab interactive notebook
└── README.md            # This guide
```

---

## 🗄️ Database Schema Features

* **`patients`**: Only `full_name` is mandatory (`NOT NULL`).
* **Auto-increment ID**: `patient_id INTEGER PRIMARY KEY AUTOINCREMENT`.
* **Clinical Tables**:
  - `visits`: Consultations and notes.
  - `medical_history`: Conditions and diagnoses.
  - `treatments`: Procedures and interventions.
  - `medications`: Active and past drugs.
  - `allergies`: Reactions and severity.
* **AI DSL Parsing**: As the AI parses conversations, it emits structured actions (e.g., `CREATE_PATIENT`, `LOG_VISIT`, `RECORD_DIAGNOSIS`, `PRESCRIBE_MEDICATION`, `RECORD_ALLERGY`) that insert rows into these tables automatically.

---

## 🚀 Running in Google Colab

### 1. Test the Database & AI Parsing
```bash
python test_db_and_ai.py
```

### 2. Run Model Training & GGUF Export
```bash
pip install -r requirements.txt
python train.py --epochs 3 --batch_size 4
```

---

## 📲 Deploying to Flutter
Copy the generated `model.gguf` into your Flutter app:
```bash
cp model.gguf ../medical/assets/models/model.gguf
```
