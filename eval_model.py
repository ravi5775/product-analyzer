import os
import sys
import json
import time

# Ensure we're running from the right directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the main app logic
t0 = time.time()
import app
t_load = time.time() - t0
print(f"app module load time: {t_load:.2f} seconds")

# Test list from the user's prompt examples and edge cases
test_ingredients = [
    "Formaldehyde",          # Known toxic Tier 4
    "BPA",                   # Known toxic Tier 4
    "parahydroxybenzoate",   # Variant of paraben (Tier 3)
    "Watrer",                # Typo (Water)
    "Aqua",                  # Alt name (Water)
    "Oryza Sativa Bran Oil", # Specific low-risk
    "random text that is not a chemical", # Completely unknown string
]

print("\n=== Running Live Prediction Tests ===")
for ing in test_ingredients:
    t_start = time.time()
    res = app.assess_ingredient(ing)
    t_end = time.time()
    risk = res['risk_tier']
    conf = res.get('match_confidence', 0.0)
    print(f"Ingredient: {ing:40} => Predicted Tier: {risk} (Confidence: {conf*100:.1f}%) [Time: {(t_end-t_start)*1000:.1f}ms]")
