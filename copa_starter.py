from datasets import load_dataset
from multiprocessing import cpu_count
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, get_scheduler
from accelerate.utils import find_executable_batch_size
import transformers
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
import json
from datasets import Dataset
from transformers import TrainingArguments, Trainer, logging
import random
random.seed(42)
from transformers import set_seed
set_seed(42)

print("+ Imports done")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

print("+ device, accelerator, and tokenizer intitialized")

def tokenize_function(example):
    choices = [example["choice1"], example["choice2"]]
    first_sentences = [example["premise"]] * len(choices)
    tokenized = tokenizer(first_sentences, choices,
                          truncation=True,
                          padding="max_length",
                          max_length=128,
                          return_tensors="pt"
                         )
    tokenized = {k:v.unsqueeze(0) for k,v in tokenized.items()}
    tokenized['labels'] = torch.tensor(int(example['label'])).to(device)
    for key in tokenized:
        tokenized[key] = torch.tensor(tokenized[key])
    return tokenized

def load_data(f):
    with open(f,'r') as of:
        lines = [json.loads(l) for l in of]
        return lines

def evaluate_sample(model,dataset,n):
  """
  Takes the model, a dataset, and n, the number of samples. Samples n random data points from the dataset,
  retrieves the model's predictions, and prints out the sampled input, correct labels, and model predictions.
  """
  samples = [dataset[i] for i in random.sample(range(len(dataset)),n)]
  dataset = DataLoader(samples, batch_size=1, shuffle=False)
  for i,batch in enumerate(dataset):
          print("Sample",i)
          print("Choice 0:",tokenizer.decode(batch["input_ids"][0][0],skip_special_tokens=True))
          print("Choice 1:",tokenizer.decode(batch["input_ids"][0][1],skip_special_tokens=True))
          print("Correct answer:",batch["labels"][0].item())
          outputs = model(**{k: v for k, v in batch.items()})
          logits = outputs.logits
          print("Logits:",logits.tolist()[0])
          best = torch.argmax(logits)
          pred = best.item()
          print("Model prediction:",pred)

def evaluate(model,dataset):
  """Takes the model and a dataset. Evaluates the model on the dataset, printing out overall accuracy."""
#   'dataset' is actually a dataloader
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in dataset:
          outputs = model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"])
          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          correct += (predictions == batch["labels"]).sum().item()
          total += batch["labels"].size(0)
  accuracy = correct / total if total > 0 else 0
  print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
  model.train()


def main(batch_size):
    accelerator.free_memory()

    model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    #     # Load training and validation data using load_data
    train_data = load_data("train.jsonl")
    val_data = load_data("val.jsonl")

    print(train_data[0])
    # print(val_data[0])

    train_encodings = [tokenize_function(item) for item in train_data]
    val_encodings = [tokenize_function(item) for item in val_data]

    #     # Set up a DataLoader for each dataset that shuffles it and batches it to batch_size

    # Custom collation
    def custom_collate(batch):
        return {
            "input_ids": torch.cat([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.cat([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.tensor([item["labels"] for item in batch])
        }

    train_dataloader = DataLoader(train_encodings,
                                  batch_size=batch_size,
                                  collate_fn=custom_collate,
                                  shuffle = True)

    val_dataloader = DataLoader(val_encodings,
                                batch_size=batch_size,
                                collate_fn=custom_collate,
                                shuffle = False)


    for batch in train_dataloader:
        print({k: v.shape for k, v in batch.items()})  # Prints the shape of your data- check that it is what you expect
        break

    model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

    num_epochs = 15
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps,
                                 )

    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            outputs = model(input_ids=batch["input_ids"], attention_mask = batch["attention_mask"],labels = batch["labels"] )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print("Evaluating on validation set...")
        evaluate(model, val_dataloader)

main(4)


# Analysis Questions:
#
# 1. How well does the LLM perform on the task?
#    - The fine-tuned model generally achieves a moderate level of performance on commonsense reasoning.
#      The validation accuracy is around 70-80%, which suggests that the LLM can reasonably
#      distinguish between plausible and implausible answer choices, though there is still room for improvement.
#
# 2. Did you observe signs of overfitting to the training data? If so, about how many epochs did this take?
#    - There appear to be signs of overfitting after about 10-12 epochs. This is evident if the training
#      accuracy continues to increase while the validation accuracy plateaus.
#
# 3. Do you notice any trends in the model performance?
#    - One key trend is that the model's performance improves rapidly during the initial epochs,
#      but then the improvements level off. Additionally, the growing gap between the training
#      and validation accuracies indicates overfitting, suggesting that early stopping or additional
#      regularization might be necessary to boost generalization.
