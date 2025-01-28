
# Text Classification Project Report

## 1. Introduction
This project focuses on classifying Turkish newspaper articles by author using machine learning and deep learning techniques. Two main approaches were explored:

1. **TF-IDF Approach**: Used a traditional machine learning classifier on text transformed using TF-IDF. This achieved an **average accuracy of ~86%**.
2. **BERT-Based Approach**: Leveraged the pre-trained `dbmdz/bert-base-turkish-cased` model for fine-tuning on the dataset. This approach achieved significantly better results.

---

## 2. Data Processing

### Data Collection
- Articles were stored in folders, with each folder corresponding to a specific author.
- A script iterated through the directories, reading and labeling the files appropriately.

### Data Cleaning
- A `clean_text` function was used to remove unnecessary whitespace and prepare the text for tokenization:
  ```python
  def clean_text(text):
      text = re.sub(r'\s+', ' ', text).strip()
      return text
  ```

### Label Encoding
- Author names were converted into numeric labels using `pandas`:
  ```python
  df['author_label'] = df['author'].astype('category').cat.codes
  ```

### Final Dataset
The processed data was stored in a Pandas DataFrame with the following columns:
- **text**: The original article text.
- **cleaned_text**: The cleaned article text.
- **author_label**: Encoded numeric labels for authors.

---

## 3. Models and Strategies

### TF-IDF Approach
#### Description
- Texts were transformed using TF-IDF, which represents each document as a vector of term importance.

#### Results
- A classical machine learning model (e.g., Logistic Regression or SVM) was trained on these vectors, achieving an **average score of ~86%**.

### BERT-Based Approach
#### Tokenizer
- The `BertTokenizer` from the `dbmdz/bert-base-turkish-cased` model was used to tokenize text and convert it into inputs compatible with BERT.

#### Model
- The `BertForSequenceClassification` model was fine-tuned for multi-class classification.
- Dropout layers were added to the classification head to reduce overfitting:
  ```python
  model.classifier = torch.nn.Sequential(
      Dropout(0.3),
      Linear(model.config.hidden_size, num_labels)
  )
  ```

#### Training Strategy
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Batch Size**: 8
- **Scheduler**: Linear scheduler with warm-up steps.
- **Early Stopping**: Training stopped if no improvement in validation loss for 3 epochs.

---

## 4. Cross-Validation

### K-Fold Cross-Validation
- A 5-fold cross-validation strategy was used to evaluate the model’s performance across different data splits.
- This ensured robust and unbiased performance metrics.

### Evaluation Metrics
- Precision, recall, and F1-scores were calculated for each class, along with macro and weighted averages.

---

## 5. Device Configuration

To optimize training, the code dynamically checks for GPU availability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## 6. Results

### TF-IDF Results
- The TF-IDF approach achieved an **average accuracy of ~86%**, showing its capability for basic text classification tasks.
- However, it failed to capture the semantic and contextual meaning of the text effectively.

### BERT Results
The BERT-based approach significantly outperformed TF-IDF. Below are the detailed precision, recall, and F1-scores for each class and overall averages:

| Class ID           | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| 0                  | 0.968     | 0.973  | 0.970    |
| 1                  | 0.852     | 0.888  | 0.864    |
| 2                  | 0.973     | 0.982  | 0.977    |
| 3                  | 0.880     | 1.000  | 0.932    |
| 4                  | 1.000     | 1.000  | 1.000    |
| 5                  | 1.000     | 1.000  | 1.000    |
| 6                  | 0.772     | 0.902  | 0.830    |
| 7                  | 0.946     | 0.910  | 0.921    |
| 8                  | 0.986     | 0.958  | 0.970    |
| 9                  | 0.935     | 0.894  | 0.910    |
| 10                 | 1.000     | 0.985  | 0.992    |
| 11                 | 0.967     | 0.927  | 0.944    |
| 12                 | 0.944     | 0.798  | 0.844    |
| 13                 | 0.934     | 0.863  | 0.894    |
| 14                 | 1.000     | 0.971  | 0.985    |
| 15                 | 0.887     | 0.846  | 0.858    |
| 16                 | 0.971     | 1.000  | 0.985    |
| 17                 | 0.950     | 0.971  | 0.960    |
| 18                 | 1.000     | 1.000  | 1.000    |
| 19                 | 0.950     | 1.000  | 0.973    |
| 20                 | 1.000     | 0.975  | 0.975    |
| 21                 | 0.911     | 0.937  | 0.937    |
| 22                 | 0.883     | 0.915  | 0.914    |
| 23                 | 0.970     | 0.984  | 0.984    |
| 24                 | 0.978     | 0.883  | 0.925    |
| 25                 | 0.964     | 0.980  | 0.969    |
| 26                 | 1.000     | 1.000  | 1.000    |
| 27                 | 0.979     | 0.982  | 0.979    |
| 28                 | 0.907     | 0.817  | 0.843    |
| 29                 | 0.897     | 0.930  | 0.912    |
| **Macro Avg**      | 0.947     | 0.945  | 0.942    |
| **Weighted Avg**   | 0.948     | 0.941  | 0.940    |

---

## 7. Conclusion

### Performance
- The BERT-based approach significantly outperformed TF-IDF, achieving a **macro-average F1-score of 0.942** and a **weighted-average F1-score of 0.940**.

### Strengths
- The BERT model effectively captures context and nuances, making it ideal for complex language tasks.

### Challenges
- Certain classes (e.g., 6, 28) had slightly lower scores, indicating areas for improvement.

---
