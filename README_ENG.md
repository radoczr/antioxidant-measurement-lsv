# Antioxidant Analysis – Voltammetry

This script evaluates voltammetric data using the standard addition method.

## Requirements
- Python 3
- numpy
- pandas
- matplotlib
- scipy

Install with:
pip install numpy pandas matplotlib scipy

## Usage

1. Place your `.txt` measurement files in a folder
2. Run the script:
python antioxidant.py

3. Enter:
- folder path
- ascorbic acid concentration

## Output

The script generates:
- Voltammograms (PDF)
- Calibration curve (PDF)
- Peak current table (PDF)

## Method

The added concentration is calculated as:

c_added = (c_AA * V_added) / (V_sample + V_added)

The unknown concentration is determined from the x-intercept of the calibration curve.
