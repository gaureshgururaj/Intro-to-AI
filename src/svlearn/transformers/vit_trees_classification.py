
#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------
#

import numpy as np
from PIL import Image
from typing import Tuple
import joblib
from scipy.special import softmax

# torch
import torch

# svlearn
from svlearn.trees.preprocess import Preprocessor
from svlearn.config.configuration import ConfigurationMixin
from svlearn.train.visualization_utils import plot_roc_curve
from svlearn.common.utils import ensure_directory

# huggingface
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
import evaluate

# sklearn
from sklearn.preprocessing import LabelEncoder

#  -------------------------------------------------------------------------------------------------

# configurations
config = ConfigurationMixin().load_config()
data_dir = config['tree-classification']['data']
raw_dir = data_dir + '/raw'
processed_dir = data_dir + "/preprocessed"
ensure_directory(processed_dir)

results_dir = config['tree-classification']['results'] + "/" + 'vit'
ensure_directory(results_dir)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load('recall')
f1_metric = evaluate.load('f1')

#  -------------------------------------------------------------------------------------------------

def print_trainable_parameters(model):
    """prints the trainable parameters of a model
    Args:
        model (_type_): any huggingface transformer model
    """

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the percentage
    trainable_percentage = (trainable_params / total_params) * 100
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {trainable_percentage:.2f}%")

#  -------------------------------------------------------------------------------------------------

def transform(example_batch):
    """processes the image at runtime for VIT model

    Args:
        example_batch (_type_): a batch of the dataset

    Returns:
        Dict: a dictionary with keys (pixel_values , labels)
    """
    batch = [Image.open(sample_path).convert("RGB") for sample_path in example_batch['image_path']]

    # Take a list of PIL images and turn them to pixel values
    inputs = processor(batch, return_tensors='pt')

    inputs['labels'] = example_batch['label']
    return inputs

#  -------------------------------------------------------------------------------------------------

def collate_fn(batch):
    """process a list of samples into a torch batch

    Args:
        batch (_type_): a dataset batch with list of dicts

    Returns:
        _type_: a torch batch
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

#  -------------------------------------------------------------------------------------------------

def compute_metrics(eval_pred: EvalPrediction):
    """ does the computation necessary during evaluation

    Args:
        eval_pred (_type_): _description_

    Returns:
        _type_: _description_
    """

    predictions=np.argmax(eval_pred.predictions, axis=1)
    probs = softmax(eval_pred.predictions, axis=1)[:, 1]
    accuracy = accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    precision = precision_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    recall = recall_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    plot_roc_curve(eval_pred.label_ids , probs, f"{results_dir}/roc.png", False)

    return {'accuracy': accuracy['accuracy'] , 
            'precision': precision['precision'] , 
            'recall': recall['recall'], 
            'f1': f1['f1']}

#  -------------------------------------------------------------------------------------------------

def prepare_datasets(save_json: bool = False) -> Tuple[Dataset , Dataset , LabelEncoder]:
    """prepares the train and test datasets for VIT models

    Args:
        save_json (bool, optional): save the train and validation dataframes. Defaults to False.

    Returns:
        Tuple[Dataset , Dataset , LabelEncoder]: train_dataset , test_dataset , label_encoder
    """
    preprocessor = Preprocessor()

    # load train_df
    train_df, val_df, label_encoder = preprocessor.preprocess(raw_dir)

    if save_json:
        train_df.to_json(f"{processed_dir}/train.json", orient='records', index=False)
        val_df.to_json(f"{processed_dir}/validation.json", orient='records', index=False)

    # create a dataset suitable for training a VIT model
    train_dataset = Dataset.from_pandas(train_df).with_transform(transform)
    val_dataset = Dataset.from_pandas(val_df).with_transform(transform)

    return train_dataset, val_dataset, label_encoder

#  -------------------------------------------------------------------------------------------------
# MAIN
#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    train_dataset , val_dataset, label_encoder = prepare_datasets()
    joblib.dump(label_encoder, f"{results_dir}/label_encoder.joblib")

    #  -------------------------------------------------------------------------------------------------

    labels = label_encoder.classes_

    model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )


    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head (last layer)
    for param in model.classifier.parameters():
        param.requires_grad = True

    print_trainable_parameters(model)

    #  -------------------------------------------------------------------------------------------------

    training_args = TrainingArguments(
    output_dir=results_dir,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='none',
    load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_model()
    trainer.save_state()


    metrics = trainer.evaluate(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
#  -------------------------------------------------------------------------------------------------