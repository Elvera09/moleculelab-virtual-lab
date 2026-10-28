# MoleculeLab — Molecular Intelligence Studio

An original glassmorphism-inspired Streamlit learning app for introductory cheminformatics. It demonstrates molecular structure rendering, SMILES parsing, common descriptors, comparisons, charts, and a short learning quiz.

## Features
- Search and filter example molecules
- Render 2D structures from SMILES
- Calculate formula, molecular weight, exact mass, cLogP, TPSA, hydrogen-bond donors/acceptors, and rotatable bonds
- Compare molecules with interactive charts
- Export a CSV properties report
- Learn introductory SMILES and descriptor concepts with a quiz

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `packages.txt` to the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set branch to `main` and main file path to `app.py`.
4. Deploy and review logs if any dependency or structure errors occur.

## Disclaimer
For education only. Calculated descriptors do not establish drug efficacy, safety, dosage, or clinical suitability. Verify structures and data against reliable sources before academic submission.
