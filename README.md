# BabySafe Ingredient Analyzer

A Flask web application for analyzing baby product ingredient safety.

## 🚀 Setup & Installation

Follow these steps to set up the project locally:

**1. Clone or download the repository:**
Ensure you are in the `baby_safety_app_final` directory.

**2. Set up a Virtual Environment (Recommended):**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```
> **Note**: This application uses `easyocr` for image scanning and `scikit-learn` for ML analysis. Ensure your system has the necessary build tools if PyTorch/EasyOCR requires them.

**4. Database configuration:**
Ensure your Excel database is placed at: `data/ingredient_safety_database_enhanced.xlsx`

---

## 💻 Running the Application

You can launch the Flask server using the provided shell script or directly via Python.

**Option A - Using Python:**
```bash
python app.py
```

**Option B - Using the Run Script:**
```bash
# On macOS/Linux:
chmod +x run.sh
./run.sh
```

**First Run Notice:** 
On the very first run, the Deep Learning model will train itself on the dataset and create a local cache (`model_cache.pkl`). This process may take **5-10 minutes**. Subsequent startups will load instantly.

Once started, open your browser and navigate to: **http://localhost:5000**

---

## 🌟 Features

- **4-Tier Ingredient Matching Engine**: 
  1. Exact Match Lookup (O(1))
  2. Fuzzy Search (`difflib` ≥80%)
  3. Semantic TF-IDF Similarity (≥65%)
  4. Deep Learning Neural Net prediction for completely unknown ingredients (>95% Accuracy)
- **188 Ingredients** indexed with 526 lookup entries (including alternative names & OCR variants)
- **Real-time Streaming** results via Server-Sent Events
- **Risk Assessment**: Safe → Low → Moderate → High → Harmful
- **Baby-specific scoring** with EU regulatory flags
- **Ingredient Search** with autocomplete
- **Safety Rules** regex pattern fallback (100 rules)

## 📊 Matching Accuracy

| Method | Threshold | Example |
|--------|-----------|---------|
| Exact | 100% | "Methylparaben" → Methylparaben |
| Fuzzy | ≥80% | "Methylparbane" → Methylparaben |
| Semantic | ≥65% | "PEG-40 Castor Oil" → PEG Compounds |
| ML Neural Net (Fallback)| Predicts Tier | "UnkwnChemicalX" → Tier 4 |
| Safety Rules | Regex Pattern | Any "paraben" string → Parabens rule |
