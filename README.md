# Fine-tuning DistilBERT on COPA for Commonsense Reasoning

This repository contains a project for fine-tuning a pre-trained DistilBERT model on the COPA (Choice of Plausible Alternatives) dataset to perform commonsense reasoning via a multiple-choice question answering task. This project was developed as part of Homework #6 for COSC-243: Natural Language Processing (Spring 2025) at Amherst College under the guidance of Professor Shira Wein.

---

## Overview

The goal of this project is to adapt a pre-trained DistilBERT model from the Hugging Face Transformers library for the COPA task. The dataset comprises a training set (with labels) and a test/validation set. The model is fine-tuned using a custom training loop that leverages the Accelerate library for streamlined training. In addition, an evaluation function calculates accuracy and provides detailed insights into performance.

---

## Project Structure

- **copa_starter.py**:  
  The main Python script that includes:  
  - Functions for data loading, tokenization, and custom batching.
  - The training loop using Hugging Face’s Accelerate.
  - An evaluation routine for computing overall accuracy.
  - In-code analysis comments that address key performance questions.

- **train.jsonl & val.jsonl**:  
  The JSON Lines files containing training and validation data, respectively.

- **README.md**:  
  This document, providing a comprehensive overview, setup instructions, and analysis.

---

## Installation

Ensure that you have Python 3.7 or later installed. Then, install the required packages using pip:

```bash
pip install datasets transformers accelerate torch tqdm
```

---

## Usage

1. **Data Preparation:**  
   Ensure that the training (`train.jsonl`) and validation (`val.jsonl`) files are available in the repository directory.

2. **Running the Script:**  
   Execute the training script with:
   ```bash
   python copa_starter.py
   ```
   The script will fine-tune the DistilBERT model, display training metrics, and evaluate the model on the validation set after each epoch.

3. **Evaluation:**  
   An `evaluate` function processes the validation set and outputs the overall accuracy. The console output will display the evaluation results after each epoch.

---

## Model and Data Details

- **Model:**  
  The model used is `DistilBertForMultipleChoice` from the Hugging Face Transformers library, pre-trained on DistilBERT.

- **Dataset:**  
  The COPA dataset, designed for commonsense reasoning, consists of a premise and two possible answer choices. During tokenization, the premise is paired with each answer choice. Note that the training set contains an "answer" key (providing the correct label), while the test set does not include labels.

- **Tokenization:**  
  The tokenizer is configured to create input sequences with a fixed maximum length (default of 128 tokens), applying necessary padding and truncation.

- **Custom Collation:**  
  A custom collate function is used in the DataLoader to correctly format the nested lists produced by tokenization into tensors with dimensions `[batch_size, num_choices, sequence_length]`.

---

## Training & Evaluation

- **Training Loop:**  
  The training loop iterates for a fixed number of epochs (e.g., 15 epochs) using the Accelerate library to manage GPU acceleration and distributed training.  
  During each batch iteration, the model computes the loss, performs backpropagation, and updates model weights.

- **Evaluation Routine:**  
  The `evaluate` function processes the validation data without computing gradients, aggregates predictions, and calculates overall accuracy by comparing predictions against the true labels.

- **Performance Analysis:**  
  In-code comments provide answers to the following analysis questions:
  
  ```python
  # Analysis Questions:
  #
  # 1. How well does the LLM perform on the task?
  #    - The fine-tuned model achieves moderate performance on the COPA task,
  #      indicating that while it can effectively distinguish between plausible and implausible alternatives,
  #      there remains significant room for improvement.
  #
  # 2. Did you observe signs of overfitting to the training data? If so, about how many epochs did this take?
  #    - Overfitting signs are observed after approximately 10-12 epochs, where training accuracy continues to improve
  #      while validation accuracy plateaus or declines.
  #
  # 3. Do you notice any trends in the model performance?
  #    - The model shows rapid initial improvements, followed by a plateau in performance.
  #      The growing gap between training and validation accuracy indicates that early stopping and regularization
  #      techniques may be beneficial.
  ```

---

## Contributing

This project was developed for an individual homework assignment and collaboration is not permitted according to the course policies. However, suggestions for improvements or enhancements to the code structure and documentation are welcome via pull requests or issues—provided they adhere to the academic integrity guidelines of the course.

---

## License

This project is provided for educational purposes only. Use and distribution are governed by the guidelines provided by Amherst College and the course instructor. Please refer to the institution's academic collaboration policy for further details.

---

## Acknowledgements

- **COPA Dataset:**  
  Originally developed for commonsense reasoning tasks ([Roemmele et al., 2011](http://commonsensereasoning.org/2011/papers/Roemmele.pdf)).

- **Hugging Face & Accelerate:**  
  Thanks to the Hugging Face team for providing robust libraries that simplify model fine-tuning and distributed training.

- **Course Staff:**  
  Special thanks to Professor Shira Wein and the teaching assistants for their support and guidance throughout the course.

---

By following these instructions and utilizing the provided scripts, you can fine-tune and evaluate a DistilBERT model on the COPA commonsense reasoning task. For any further questions or clarifications, please consult the course materials or contact the course staff.

Happy Fine-Tuning!
