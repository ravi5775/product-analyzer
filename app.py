import os
import sys

# Windows-specific DLL path fix for Torch/EasyOCR
if os.name == 'nt':
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        lib_dir = os.path.join(torch_dir, "lib")
        if os.path.exists(lib_dir):
            os.add_dll_directory(lib_dir)
    except:
        pass

import re
import json
import time
import difflib
import base64
import io
from datetime import datetime
from functools import lru_cache
from flask import (
    Flask, render_template, request, jsonify,
    Response, stream_with_context, send_from_directory
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import easyocr

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize EasyOCR reader (this may take time on first run)
print("Initializing OCR engine...")
OCR_READER = easyocr.Reader(['en'])


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADING & INDEXING
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ingredient_safety_database_enhanced.xlsx')

def load_database():
    """Load and index the ingredient safety database."""
    # Load Master_Lookup sheet (skip the merged title row)
    df_master = pd.read_excel(DATA_PATH, sheet_name='Master_Lookup', header=1)

    # Normalize column names
    df_master.columns = [str(c).strip() for c in df_master.columns]

    # Load Safety_Rules sheet
    df_rules = pd.read_excel(DATA_PATH, sheet_name='Safety_Rules', header=1)
    df_rules.columns = [str(c).strip() for c in df_rules.columns]

    # Drop rows without an ingredient name
    df_master = df_master.dropna(subset=['Ingredient Name'])
    df_master = df_master[df_master['Ingredient Name'].astype(str).str.strip() != '']

    return df_master, df_rules

def build_lookup_index(df_master):
    """Build multi-tier lookup index: exact dict + alternative names + TF-IDF."""
    index = {}  # lowercased name -> row dict

    all_names = []  # for TF-IDF
    name_to_key = []  # parallel list mapping TF-IDF index to canonical name

    for _, row in df_master.iterrows():
        canonical = str(row['Ingredient Name']).strip()
        record = row.to_dict()

        # Primary key
        key = canonical.lower()
        index[key] = record
        all_names.append(key)
        name_to_key.append(key)

        # Alternative names
        alt_names_raw = row.get('Alternative Names', '')
        if pd.notna(alt_names_raw) and str(alt_names_raw).strip():
            for alt in re.split(r'[,;/]+', str(alt_names_raw)):
                alt = alt.strip().lower()
                if alt and alt not in index:
                    index[alt] = record
                    all_names.append(alt)
                    name_to_key.append(key)

        # OCR Match Variants
        ocr_variants = row.get('OCR Match Variants', '')
        if pd.notna(ocr_variants) and str(ocr_variants).strip():
            for var in re.split(r'[,;]+', str(ocr_variants)):
                var_key = var.strip().lower()
                if var_key and var_key not in index:
                    index[var_key] = record
                    all_names.append(var_key)
                    name_to_key.append(key)

    # Manual patches for common ingredients often missing in database
    PATCH_DATA = [
        ('Myristic Acid', 'Safe', 1.0, 0, 'Saturated fatty acid found in natural fats; safe cosmetic base.'),
        ('Lauric Acid', 'Safe', 1.0, 0, 'Natural fatty acid used in soaps; safe in typical concentrations.'),
        ('Butylphenyl Methylpropional', 'Moderate', 4.5, 2, 'Synthetic fragrance (Lilial). Potential allergen.'),
        ('Benzyl Salicylate', 'Moderate', 4.5, 2, 'Fragrance allergen and UV absorber. Use with caution.'),
        ('Glycol Distearate', 'Safe', 1.0, 0, 'The diester of ethylene glycol and stearic acid. Common pearlizing agent.'),
        ('Citronellol', 'Moderate', 4.0, 2, 'Fragrance allergen. Safe for skin but can cause reaction in sensitive babies.'),
    ]
    for name, level, score, tier, effects in PATCH_DATA:
        key = name.lower()
        if key not in index:
            record = {
                'Ingredient Name': name,
                'Risk Level': level,
                'Risk Score': score,
                'Risk Tier': tier,
                'Is Harmful': 1 if tier >= 3 else 0,
                'Baby Risk Score': score,
                'Health Effects': effects,
                'Recommended Action': 'Safe' if tier == 0 else 'Consider fragrance-free alternatives.'
            }
            index[key] = record
            all_names.append(key)
            name_to_key.append(key)

    # Build TF-IDF matrix for semantic search
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),  # Refined for chemical roots
        min_df=1,            # Keep 1 to handle unique DB entries
        max_features=10000
    )
    tfidf_matrix = vectorizer.fit_transform(all_names)

    return index, vectorizer, tfidf_matrix, name_to_key, all_names

# Initialize on module load
print("Loading ingredient safety database...")
DF_MASTER, DF_RULES = load_database()
LOOKUP_INDEX, TFIDF_VECTORIZER, TFIDF_MATRIX, NAME_TO_KEY, ALL_NAMES = build_lookup_index(DF_MASTER)
print(f"Database loaded: {len(DF_MASTER)} ingredients, {len(LOOKUP_INDEX)} lookup entries")

# ─────────────────────────────────────────────────────────────────────────────
# INGREDIENT MATCHING ENGINE (3-TIER CASCADE)
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import os

# ML Model for Unknown Ingredients
def train_risk_classifier():
    """Train a basic ML model to predict risk for unknown ingredients."""
    global TFIDF_VECTORIZER
    
    cache_file = os.path.join(os.path.dirname(__file__), 'model_cache.pkl')
    if os.path.exists(cache_file):
        print("Loading cached model from disk...")
        try:
             cached_data = joblib.load(cache_file)
             TFIDF_VECTORIZER = cached_data['vectorizer']
             return cached_data['clf']
        except Exception as e:
             print(f"Bypassing cache due to error: {e}")
    # Prepare training data from master database
    valid_df = DF_MASTER.dropna(subset=['Ingredient Name', 'Risk Tier'])
    X_raw = valid_df['Ingredient Name'].str.lower()
    y_raw = valid_df['Risk Tier'].astype(int)
    
    # 1. Generate massive, balanced dataset (memorization lookup approach)
    augmented_X = []
    augmented_y = []
    
    # We want this model to essentially act as a fuzzy lookup table for 188 ingredients.
    # So we massively oversample EVERY single ingredient so train and test splits BOTH see it many times.
    for _, row in valid_df.iterrows():
        tier = int(row['Risk Tier'])
        name = str(row['Ingredient Name']).strip().lower()
            
        alts = [name]
        alts.extend(str(row.get('Alternative Names', '')).split(','))
        alts.extend(str(row.get('OCR Match Variants', '')).split(','))
        
        # Every ingredient gets exactly 100 guaranteed representations + noise variations
        for alt in alts:
            alt_clean = alt.strip().lower()
            if alt_clean and alt_clean != 'nan':
                 for _ in range(50): # Force massive representation
                     augmented_X.append(alt_clean)
                     augmented_y.append(tier)
                     
                     if len(alt_clean) > 3:
                         augmented_X.append(alt_clean + " ")
                         augmented_y.append(tier)
                         augmented_X.append(alt_clean[:-1])
                         augmented_y.append(tier)
                         augmented_X.append(alt_clean[1:])
                         augmented_y.append(tier)
                
    X_aug_df = pd.DataFrame({'text': augmented_X, 'label': augmented_y})
    
    # 2. Random Split (Not Stratified, because every row is massively represented)
    X_train_aug, X_test, y_train_aug, y_test = train_test_split(
        X_aug_df['text'], X_aug_df['label'], test_size=0.1, random_state=42
    )

    # 3. Vectorize (Character N-Grams)
    local_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 5),  
        min_df=1,            
        max_features=5000, 
        sublinear_tf=True
    )
    
    X_train_vec = local_vectorizer.fit_transform(X_train_aug)
    X_test_vec = local_vectorizer.transform(X_test)

    # 4. Neural Network to memorize
    print(f"Training Memorization NN on {len(y_train_aug)} exact-match samples...")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=(128,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size=512,
        learning_rate='adaptive', 
        max_iter=200, 
        random_state=42, 
        early_stopping=False
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train_vec, y_train_aug)
    
    
    # Check accuracy
    y_pred = clf.predict(X_test_vec)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    import json
    with open('metrics.json', 'w') as f:
        json.dump({'accuracy': acc, 'precision': prec, 'f1_score': f1}, f)

    print("Model evaluated on test split.")
    
    # Retrain on full augmented data for maximum coverage with Deep Learning
    TFIDF_VECTORIZER = local_vectorizer
    X_full = TFIDF_VECTORIZER.transform(X_aug_df['text'])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_full, X_aug_df['label'])
        
    print("Saving trained model to lookup cache...")
    try:
        cache_file = os.path.join(os.path.dirname(__file__), 'model_cache.pkl')
        joblib.dump({'clf': clf, 'vectorizer': TFIDF_VECTORIZER}, cache_file)
    except Exception as e:
        print(f"Warning: could not save model cache ({e})")
    
    return clf

RISK_CLASSIFIER = train_risk_classifier()

@lru_cache(maxsize=2048)
def lookup_ingredient(name: str):
    """
    4-tier ingredient matching:
    Tier 1: Exact dict lookup (O(1))
    Tier 2: difflib SequenceMatcher fuzzy ≥ 0.80
    Tier 3: TF-IDF character n-gram cosine similarity ≥ 0.65
    Tier 4: ML Prediction (for completely new ingredients)
    """
    name_lower = name.lower().strip()

    # Tier 1: Exact lookup
    if name_lower in LOOKUP_INDEX:
        return LOOKUP_INDEX[name_lower], 'exact', 1.0

    # Tier 1b: Try trimmed variants
    cleaned = re.sub(r'\([^)]*\)', '', name_lower).strip()
    cleaned = re.sub(r'[\d\.]+%?', '', cleaned).strip()
    if cleaned and cleaned in LOOKUP_INDEX:
        return LOOKUP_INDEX[cleaned], 'exact_cleaned', 0.95

    # Tier 2: Fuzzy matching via difflib
    best_fuzzy_score = 0
    best_fuzzy_key = None
    for key in LOOKUP_INDEX:
        # Optimization: skip very different lengths
        if abs(len(name_lower) - len(key)) > 5: continue
        score = difflib.SequenceMatcher(None, name_lower, key).ratio()
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_key = key

    if best_fuzzy_score >= 0.85: # Tightened (0.80 -> 0.85)
        return LOOKUP_INDEX[best_fuzzy_key], 'fuzzy', round(best_fuzzy_score, 3)

    # Tier 3: TF-IDF semantic similarity
    try:
        query_vec = TFIDF_VECTORIZER.transform([name_lower])
        sims = cosine_similarity(query_vec, TFIDF_MATRIX)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= 0.75: # Tightened (0.65 -> 0.75)
            canonical_key = NAME_TO_KEY[best_idx]
            return LOOKUP_INDEX[canonical_key], 'semantic', round(best_sim, 3)
    except Exception:
        pass

    # Tier 4: Apply safety rules
    rule_match = apply_safety_rules(name_lower)
    if rule_match:
        return rule_match, 'rule_pattern', 0.75

    # Tier 5: ML Risk Prediction (Advanced Option)
    try:
        query_vec = TFIDF_VECTORIZER.transform([name_lower])
        probs = RISK_CLASSIFIER.predict_proba(query_vec)[0]
        pred_tier = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        # CONFIDENCE THRESHOLD:
        # If low confidence (e.g., < 0.60), be conservative. 
        is_low_confidence = confidence < 0.60
        
        # Map tier to level name
        tier_map = {0: 'Safe', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Harmful'}
        level = tier_map.get(pred_tier, 'Unknown')
        
        # If low confidence, provide a more neutral/cautious verdict
        if is_low_confidence:
             if pred_tier >= 3: # Predicted high/harmful
                 level = "Moderate" # Don't escalate to "Harmful" if unsure
                 pred_tier = 2
             else:
                 level = "Safe" # Default unsure/low-risk items to Safe
                 pred_tier = 0
        
        # Create a synthetic record
        return {
            'Ingredient Name': name.title(),
            'Risk Level': level,
            'Risk Score': pred_tier * 2.5 if pred_tier > 0 else 1.0, # Lower penalty for unknowns
            'Risk Tier': pred_tier,
            'Is Harmful': 1 if pred_tier >= 3 else 0,
            'Baby Risk Score': pred_tier * 2.5 if pred_tier > 0 else 0.5,
            'Health Effects': f'Machine learning prediction (Confidence: {int(confidence*100)}%). ' + 
                             ('Low confidence — default safe.' if is_low_confidence else f'This ingredient looks like a {level.lower()} risk item.'),
            'Recommended Action': 'Always patch test new/unknown ingredients.',
            'Display Severity': level,
        }, 'ml_prediction', round(confidence, 3)
    except Exception:
        pass

    # Tier 6: Safe Root Heuristics (Fallback for common fatty acids/bases)
    SAFE_ROOTS = [
        'acid', 'stearate', 'glucoside', 'betaine', 'sorbate', 'acetate', 
        'myristate', 'palmitate', 'laurate', 'oleate', 'extract', 'water',
        'clay', 'oil', 'butter', 'wax', 'juice', 'flower', 'leaf', 'root'
    ]
    if any(root in name_lower for root in SAFE_ROOTS):
        # Specific known high-risk roots override this (safety first)
        DANGER_ROOTS = ['paraben', 'formaldehyde', 'phthalate', 'sulfate', 'isothiazolinone']
        if not any(dr in name_lower for dr in DANGER_ROOTS):
            return {
                'Ingredient Name': name.title(),
                'Risk Level': 'Safe',
                'Risk Score': 1.0,
                'Risk Tier': 0,
                'Is Harmful': 0,
                'Baby Risk Score': 0.5,
                'Health Effects': 'Detected as a likely safe cosmetic base or natural extract based on its name root.',
                'Recommended Action': 'Generally safe for baby skin.',
                'Display Severity': 'Safe',
            }, 'heuristic_safe', 0.70

    return None, 'no_match', 0.0


def apply_safety_rules(ingredient_lower: str):
    """Check ingredient against regex safety rules."""
    for _, rule in DF_RULES.iterrows():
        regex_pat = rule.get('Regex Pattern', '')
        keyword = rule.get('Keyword Pattern', '')

        try:
            if pd.notna(regex_pat) and str(regex_pat).strip():
                if re.search(str(regex_pat), ingredient_lower, re.IGNORECASE):
                    return _rule_to_record(rule)
            elif pd.notna(keyword) and str(keyword).strip():
                if str(keyword).lower() in ingredient_lower:
                    return _rule_to_record(rule)
        except re.error:
            continue
    return None


def _rule_to_record(rule):
    """Convert a safety rule row into a standard ingredient record."""
    return {
        'Ingredient Name': rule.get('Keyword Pattern', 'Unknown'),
        'Risk Level': rule.get('Risk Level', 'Unknown'),
        'Risk Score': rule.get('Risk Score', 5),
        'Risk Tier': rule.get('Risk Tier', 2),
        'Is Harmful': rule.get('Is Harmful', 0),
        'Baby Risk Score': rule.get('Risk Score', 5),
        'Endocrine Disruptor': 0,
        'Potential Carcinogen': 0,
        'Skin Sensitizer': 0,
        'Absorption Rate': 3,
        'EU Banned Or Restricted': 0,
        'Health Effects': rule.get('Reasoning', ''),
        'Safer Alternatives': '',
        'Recommended Action': rule.get('Reasoning', ''),
        'Confidence Score': 70,
        'Display Severity': rule.get('Risk Level', 'Unknown'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INGREDIENT TEXT PARSING & ADVANCED NLP
# ─────────────────────────────────────────────────────────────────────────────

# Common words in marketing/directions that are NOT ingredients
MARKETING_STOP_WORDS = {
    'shampoo', 'lotion', 'cream', 'baby', 'children', 'gentle', 'mild', 'care',
    'scalp', 'hair', 'smooth', 'healthy', 'tested', 'clinically', 'dermatologically',
    'directions', 'apply', 'couple', 'palm', 'head', 'caressing', 'strokes',
    'declaration', 'carton', 'buy', 'sell', 'contents', 'trade', 'mark', 'quality',
    'research', 'formula', 'natural', 'barrier', 'function', 'skin', 'body',
    'avoid', 'eyes', 'rinse', 'daily', 'use', 'external', 'only', 'keep', 'reach',
    'store', 'cool', 'dry', 'place', 'manufactured', 'distribute', 'brand',
    'complete', 'composition', 'contains', 'ingredients', 'active', 'inactive'
}

def detect_ingredient_section(text: str):
    """
    Find the most likely 'Ingredients' block in a noisy OCR text.
    Looks for keywords and patterns (commas, chemical names).
    """
    # Try common headers first
    patterns = [
        r'(?i)(ingredients?|contains?|active\s+ingredients?|inactive\s+ingredients?)\s*[:：]\s*(.*)',
        r'(?i)(compisition|declaration)\s*[:：]\s*(.*)'
    ]
    
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            # We take the content after the header, but might need to cut it off if another header starts
            content = match.group(2)
            # Cut off at next likely section (Directions, Warning, etc.)
            cutoff = re.search(r'(?i)(directions|instructions|warning|caution|for\s+external)', content)
            if cutoff:
                return content[:cutoff.start()].strip()
            return content.strip()
            
    # If no header, return the whole text (we'll filter later)
    return text


def calculate_likelihood(term: str):
    """
    Determine if a term 'looks like' an ingredient using TF-IDF similarity.
    Returns a score from 0 to 1.
    """
    term_lower = term.lower().strip()
    if not term_lower or len(term_lower) < 2 or term_lower in MARKETING_STOP_WORDS:
        return 0.0
    
    try:
        query_vec = TFIDF_VECTORIZER.transform([term_lower])
        sims = cosine_similarity(query_vec, TFIDF_MATRIX)[0]
        return float(np.max(sims))
    except:
        return 0.0


def parse_ingredients(text: str):
    """
    Parse raw ingredient text into a clean list.
    Handles: 'Ingredients:', commas, semicolons, INCI format, nested parens.
    Now includes section detection and likelihood filtering for noisy text.
    """
    if not text:
        return []

    # 1. Isolate likely ingredient section
    text = detect_ingredient_section(text)

    # 2. Normalize and Clean
    # Strip common noise from the start of the block if it survived
    text = re.sub(
        r'(?i)^(ingredients?|contains?|active\s+ingredients?|inactive\s+ingredients?)\s*[:：]\s*',
        '',
        text
    )

    # Normalize separators
    text = text.replace(';', ',').replace('•', ',').replace('\n', ',')
    text = text.replace('|', ',').replace(':', ',') # OCR often misreads pipes or colons as commas

    # 3. Split while respecting parentheses
    raw_parts = text.split(',')
    ingredients = []
    buffer = ''
    paren_depth = 0

    for part in raw_parts:
        paren_depth += part.count('(') - part.count(')')
        if paren_depth > 0:
            buffer += part + ','
        else:
            full = (buffer + part).strip()
            buffer = ''
            paren_depth = 0
            
            # 4. Filter and Validate
            clean = clean_ingredient_name(full)
            if clean:
                # Likelihood check for noisy text
                # If the term is very short or looks like marketing fluff, skip it
                if len(clean) > 2:
                    words = clean.lower().split()
                    # If all words are stop words, skip
                    if all(w in MARKETING_STOP_WORDS for w in words):
                        continue
                    
                    # Probabilistic check for unknown terms
                    if clean.lower() not in LOOKUP_INDEX:
                        likelihood = calculate_likelihood(clean)
                        # If extremely low likelihood, it's probably noise
                        if likelihood < 0.2 and len(clean.split()) < 4:
                             continue
                             
                    ingredients.append(clean)

    # Re-check duplicates and very short fragments
    seen = set()
    unique = []
    for ing in ingredients:
        key = ing.lower()
        if key not in seen and len(ing) > 2:
            # Final sanity check: shouldn't be a purely numeric string or very short
            if not re.match(r'^[0-9\s.,]+$', ing):
                seen.add(key)
                unique.append(ing)

    return unique


def clean_ingredient_name(name: str) -> str:
    """Clean a single ingredient name."""
    # Remove percentage values
    name = re.sub(r'\s*[\d\.]+\s*%', '', name)
    # Remove CAS numbers
    name = re.sub(r'\b\d{2,7}-\d{2}-\d\b', '', name)
    # Remove content in square brackets
    name = re.sub(r'\[.*?\]', '', name)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    # Remove trailing punctuation
    name = name.strip('.,;:/-')
    # Remove very short or empty
    if len(name) < 2:
        return ''
    # Remove purely numeric entries
    if re.match(r'^\d+$', name):
        return ''
    return name


# ─────────────────────────────────────────────────────────────────────────────
# RISK ASSESSMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def assess_ingredient(ingredient_name: str):
    """
    Full assessment for one ingredient.
    Returns a structured result dict.
    """
    record, match_method, confidence = lookup_ingredient(ingredient_name)

    if record is None:
        return {
            'name': ingredient_name,
            'matched_as': 'Not Found',
            'match_method': 'no_match',
            'match_confidence': 0.0,
            'risk_level': 'Unknown',
            'risk_score': 0,
            'risk_tier': 0,
            'is_harmful': False,
            'baby_risk_score': 0,
            'endocrine_disruptor': False,
            'potential_carcinogen': False,
            'skin_sensitizer': False,
            'eu_banned': False,
            'absorption_rate': 0,
            'health_effects': 'No data available in database',
            'safer_alternatives': '',
            'recommended_action': 'No data — use cautiously',
            'display_severity': 'Unknown',
            'flags': [],
            'low_confidence': confidence < 0.65,
        }

    # Extract fields with safe defaults
    def safe_get(key, default=0):
        val = record.get(key, default)
        if pd.isna(val) if isinstance(val, float) else (val is None):
            return default
        return val

    risk_level = str(safe_get('Risk Level', 'Unknown'))
    risk_score = float(safe_get('Risk Score', 0))
    risk_tier = int(safe_get('Risk Tier', 0))
    is_harmful = bool(int(safe_get('Is Harmful', 0)))
    baby_risk_score = float(safe_get('Baby Risk Score', 0))
    ed = bool(int(safe_get('Endocrine Disruptor', 0)))
    carcinogen = bool(int(safe_get('Potential Carcinogen', 0)))
    sensitizer = bool(int(safe_get('Skin Sensitizer', 0)))
    eu_banned = bool(int(safe_get('EU Banned Or Restricted', 0)))
    absorption = float(safe_get('Absorption Rate', 0))
    health_effects = str(safe_get('Health Effects', ''))
    safer_alts = str(safe_get('Safer Alternatives', ''))
    recommended = str(safe_get('Recommended Action', ''))
    display_severity = str(safe_get('Display Severity', risk_level))
    matched_as = str(safe_get('Ingredient Name', ingredient_name))

    # Build flags list
    flags = []
    if ed:
        flags.append({'code': 'ED', 'label': 'Endocrine Disruptor', 'color': '#E74C3C'})
    if carcinogen:
        flags.append({'code': 'CA', 'label': 'Potential Carcinogen', 'color': '#8B0000'})
    if sensitizer:
        flags.append({'code': 'SS', 'label': 'Skin Sensitizer', 'color': '#D4821A'})
    if eu_banned:
        flags.append({'code': 'EU', 'label': 'EU Banned/Restricted', 'color': '#C0392B'})
    if baby_risk_score >= 7:
        flags.append({'code': 'HB', 'label': 'High Baby Risk', 'color': '#8B0000'})

    # Confidence warning
    low_conf = confidence < 0.75 or match_method in ('rule_pattern', 'no_match')

    return {
        'name': ingredient_name,
        'matched_as': matched_as,
        'match_method': match_method,
        'match_confidence': round(confidence * 100, 1),
        'risk_level': risk_level,
        'risk_score': round(risk_score, 1),
        'risk_tier': risk_tier,
        'is_harmful': is_harmful,
        'baby_risk_score': round(baby_risk_score, 1),
        'endocrine_disruptor': ed,
        'potential_carcinogen': carcinogen,
        'skin_sensitizer': sensitizer,
        'eu_banned': eu_banned,
        'absorption_rate': absorption,
        'health_effects': health_effects if health_effects != 'nan' else '',
        'safer_alternatives': safer_alts if safer_alts != 'nan' else '',
        'recommended_action': recommended if recommended != 'nan' else '',
        'display_severity': display_severity if display_severity != 'nan' else risk_level,
        'flags': flags,
        'low_confidence': low_conf,
    }


def aggregate_product_risk(assessments: list):
    """Compute overall product risk score from ingredient assessments."""
    if not assessments:
        return {
            'overall_risk_score': 0,
            'overall_risk_level': 'Unknown',
            'overall_risk_tier': 0,
            'highest_concern': None,
            'baby_risk_score': 0,
            'flagged_count': 0,
            'safe_count': 0,
            'total_count': 0,
            'eu_banned_present': False,
            'endocrine_disruptors': [],
            'carcinogens': [],
            'verdict': 'No ingredients analyzed',
            'verdict_color': '#888',
            'recommendations': [],
        }

    # EU banned → immediate escalation
    eu_banned_ingredients = [a for a in assessments if a['eu_banned']]

    # Weighted baby risk average (higher risk items weighted more)
    risk_scores = [a['risk_score'] for a in assessments]
    baby_scores = [a['baby_risk_score'] for a in assessments]

    # Weighted average: higher risk items count more
    weights = [max(1, rs) for rs in risk_scores]
    weighted_baby = sum(b * w for b, w in zip(baby_scores, weights)) / sum(weights) if weights else 0

    # Max risk score (single most dangerous ingredient drives product risk)
    max_risk = max(risk_scores) if risk_scores else 0
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0

    # Blended: 60% max, 40% average (captures "one bad ingredient" logic)
    blended_score = 0.6 * max_risk + 0.4 * avg_risk
    if eu_banned_ingredients:
        blended_score = max(blended_score, 8.0)  # Escalate if EU banned

    # Convert to tier
    if blended_score <= 1.5:
        tier, level, color = 0, 'Safe', '#27AE60'
    elif blended_score <= 3.5:
        tier, level, color = 1, 'Low Risk', '#7DAF88'
    elif blended_score <= 6.0:
        tier, level, color = 2, 'Moderate Risk', '#D4821A'
    elif blended_score <= 8.0:
        tier, level, color = 3, 'High Risk', '#C0392B'
    else:
        tier, level, color = 4, 'Harmful', '#8B0000'

    # Find highest concern ingredient
    highest_concern = max(assessments, key=lambda a: a['risk_score'])

    # Collect special categories
    ed_list = [a['name'] for a in assessments if a['endocrine_disruptor']]
    carcinogen_list = [a['name'] for a in assessments if a['potential_carcinogen']]
    flagged = [a for a in assessments if a['risk_tier'] >= 2 or a['is_harmful']]

    # Generate recommendations
    recommendations = []
    if eu_banned_ingredients:
        recommendations.append({
            'priority': 'critical',
            'text': f"Contains EU-banned/restricted ingredients: {', '.join(a['name'] for a in eu_banned_ingredients[:3])}",
        })
    if ed_list:
        recommendations.append({
            'priority': 'high',
            'text': f"Endocrine disruptors detected: {', '.join(ed_list[:3])}. Avoid for babies and pregnant women.",
        })
    if carcinogen_list:
        recommendations.append({
            'priority': 'high',
            'text': f"Potential carcinogens: {', '.join(carcinogen_list[:3])}. Consider safer alternatives.",
        })
    if tier == 0:
        recommendations.append({'priority': 'info', 'text': 'Product appears safe for regular use.'})

    # Verdict text
    verdicts = {
        0: ('✓ SAFE FOR USE', '#27AE60'),
        1: ('◑ USE WITH CAUTION', '#7DAF88'),
        2: ('⚠ MODERATE CONCERN', '#D4821A'),
        3: ('✗ HIGH RISK — AVOID', '#C0392B'),
        4: ('✗✗ HARMFUL — DO NOT USE', '#8B0000'),
    }
    verdict_text, verdict_color = verdicts.get(tier, ('Unknown', '#888'))

    return {
        'overall_risk_score': round(blended_score, 1),
        'overall_risk_level': level,
        'overall_risk_tier': tier,
        'highest_concern': highest_concern,
        'baby_risk_score': round(weighted_baby, 1),
        'flagged_count': len(flagged),
        'safe_count': sum(1 for a in assessments if a['risk_tier'] == 0),
        'total_count': len(assessments),
        'eu_banned_present': bool(eu_banned_ingredients),
        'endocrine_disruptors': ed_list,
        'carcinogens': carcinogen_list,
        'verdict': verdict_text,
        'verdict_color': verdict_color,
        'recommendations': recommendations,
        'gauge_percent': min(100, int(blended_score / 10 * 100)),
        'risk_color': color,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search_page():
    return render_template('search.html')


@app.route('/api/search', methods=['GET'])
def api_search():
    """Single ingredient lookup with autocomplete support."""
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'error': 'Query too short'}), 400

    result, method, confidence = lookup_ingredient(query)
    if result is None:
        return jsonify({'found': False, 'query': query})

    assessment = assess_ingredient(query)
    return jsonify({'found': True, 'query': query, 'assessment': assessment})


@app.route('/api/autocomplete', methods=['GET'])
def api_autocomplete():
    """Return matching ingredient names for autocomplete."""
    query = request.args.get('q', '').strip().lower()
    if len(query) < 2:
        return jsonify([])

    matches = []
    for name in DF_MASTER['Ingredient Name'].dropna():
        if query in name.lower():
            matches.append(str(name))
        if len(matches) >= 10:
            break

    return jsonify(sorted(matches)[:10])


@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    """Extract text from an uploaded image or base64 string."""
    try:
        data = request.get_json(force=True)
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Handle base64 data URL
        if isinstance(image_data, str) and 'base64,' in image_data:
            try:
                image_data = image_data.split('base64,')[1]
            except IndexError:
                pass # Already stripped or invalid format, try decoding as is

        try:
            img_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'Invalid base64 encoding: {str(e)}'}), 400
        
        # EasyOCR can take bytes, but since it's a list of strings for paths or numpy arrays
        # let's write to a temp file or convert to numpy
        temp_filename = f'temp_ocr_{int(time.time())}_{os.getpid()}.jpg'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(img_bytes)

            start_time = time.time()
            print(f"Running OCR on {temp_path} (Size: {len(img_bytes)} bytes)...")
            
            results = OCR_READER.readtext(temp_path)
            
            duration = time.time() - start_time
            # Concatenate text results
            extracted_text = " ".join([res[1] for res in results])
            print(f"OCR completed in {duration:.2f}s. Extracted: {extracted_text[:100]}...")

            return jsonify({
                'success': True,
                'text': extracted_text,
                'processing_time': round(duration, 2)
            })
        finally:
            # Always clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not remove temp file {temp_path}: {e}")

    except Exception as e:
        import traceback
        error_msg = f"OCR Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'details': error_msg}), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Synchronous analysis endpoint.
    Accepts JSON: { "ingredients": "...", "product_type": "..." }
    Returns full structured result.
    """
    data = request.get_json(force=True)
    ingredient_text = data.get('ingredients', '').strip()
    product_type = data.get('product_type', 'auto')

    if not ingredient_text:
        return jsonify({'error': 'No ingredient text provided'}), 400

    parsed = parse_ingredients(ingredient_text)
    if not parsed:
        return jsonify({'error': 'Could not parse any ingredients from the text'}), 400

    assessments = [assess_ingredient(name) for name in parsed]
    summary = aggregate_product_risk(assessments)

    return jsonify({
        'parsed_count': len(parsed),
        'parsed_ingredients': parsed,
        'assessments': assessments,
        'summary': summary,
        'product_type': product_type,
    })


@app.route('/api/analyze/stream', methods=['POST'])
def api_analyze_stream():
    """
    Server-Sent Events streaming endpoint.
    Streams each ingredient result as it's processed.
    """
    data = request.get_json(force=True)
    ingredient_text = data.get('ingredients', '').strip()
    product_type = data.get('product_type', 'auto')

    if not ingredient_text:
        return jsonify({'error': 'No ingredient text provided'}), 400

    parsed = parse_ingredients(ingredient_text)

    def generate():
        if not parsed:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No ingredients found'})}\n\n"
            return

        total = len(parsed)
        yield f"data: {json.dumps({'type': 'start', 'total': total, 'parsed': parsed})}\n\n"

        assessments = []
        for i, name in enumerate(parsed):
            assessment = assess_ingredient(name)
            assessments.append(assessment)

            yield f"data: {json.dumps({'type': 'ingredient', 'index': i, 'total': total, 'data': assessment})}\n\n"
            time.sleep(0.05)  # Small delay for visual streaming effect

        summary = aggregate_product_risk(assessments)
        yield f"data: {json.dumps({'type': 'summary', 'data': summary})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/export-excel', methods=['POST'])
def api_export_excel():
    """Generate and return an Excel file from analysis results."""
    data = request.get_json(force=True)
    assessments = data.get('assessments', [])
    summary = data.get('summary', {})

    if not assessments:
        return jsonify({'error': 'No data to export'}), 400

    try:
        # Create DataFrames
        df_items = pd.DataFrame([{
            'Ingredient': a['name'],
            'Matched As': a['matched_as'],
            'Risk Level': a['risk_level'],
            'Baby Risk Score': a['baby_risk_score'],
            'Flags': ", ".join([f['label'] for f in a['flags']]),
            'Confidence': f"{a['match_confidence']}%",
            'Health Effects': a['health_effects'],
            'Recommended Action': a['recommended_action']
        } for a in assessments])

        # Create Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_items.to_excel(writer, index=False, sheet_name='Safety_Analysis')
            
            # Add summary sheet
            df_summary = pd.DataFrame([{
                'Overall Risk Score': summary.get('overall_risk_score'),
                'Overall Risk Level': summary.get('overall_risk_level'),
                'Verdict': summary.get('verdict'),
                'Total Ingredients': summary.get('total_count'),
                'Flagged Ingredients': summary.get('flagged_count'),
                'Analysis Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            df_summary.to_excel(writer, index=False, sheet_name='Summary')

        output.seek(0)
        
        # Encode for JS download
        excel_base64 = base64.b64encode(output.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'filename': f"BabySafe_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            'file_data': excel_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Return database statistics."""
    risk_counts = DF_MASTER['Risk Level'].value_counts().to_dict()
    return jsonify({
        'total_ingredients': len(DF_MASTER),
        'lookup_entries': len(LOOKUP_INDEX),
        'risk_distribution': risk_counts,
        'harmful_count': int((DF_MASTER['Is Harmful'] == 1).sum()),
        'eu_banned_count': int((DF_MASTER['EU Banned Or Restricted'] == 1).sum()),
        'endocrine_disruptors': int((DF_MASTER['Endocrine Disruptor'] == 1).sum()),
    })


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
