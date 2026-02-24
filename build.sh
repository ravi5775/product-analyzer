#!/usr/bin/env bash
# exit on error
set -o errexit

# Update pip
pip install --upgrade pip

# Install requirements
# Using CPU-only torch to save space and avoid build issues
pip install -r requirements.txt
