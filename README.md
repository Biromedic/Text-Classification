
# Text Classification Project

## Introduction
This repository contains a project focused on classifying Turkish newspaper articles by author using machine learning and deep learning techniques. Two approaches were implemented:
1. **TF-IDF Approach**: Achieved an average accuracy of ~86% using traditional machine learning classifiers.
2. **BERT-Based Approach**: Leveraged the pre-trained `dbmdz/bert-base-turkish-cased` model for fine-tuning, achieving significantly better results.

---

## Project Features
- **Data Collection**: Articles were collected and labeled from directories corresponding to each author.
- **Text Cleaning**: Preprocessing involved removing extra whitespace and normalizing text.
- **Modeling**: Includes both traditional ML (TF-IDF + Logistic Regression) and advanced DL (BERT for Sequence Classification).
- **Cross-Validation**: Used 5-fold cross-validation for robust evaluation.
- **Evaluation Metrics**: Precision, Recall, and F1-scores were calculated for each class, along with macro and weighted averages.

## Results

### TF-IDF Results
- The TF-IDF approach achieved an average accuracy of ~86%, showing its capability for basic text classification tasks but lacked semantic and contextual understanding.

### BERT Results
The BERT-based approach significantly outperformed TF-IDF. Below are the average metrics:
- **Macro-Average F1-Score**: 0.942
- **Weighted-Average F1-Score**: 0.940

For detailed metrics for each class, refer to the report.
