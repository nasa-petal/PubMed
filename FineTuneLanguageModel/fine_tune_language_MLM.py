'''
    Finetuning a language model. Taking an existing model GPT2 (small version) and running it with a larger dataset.
        This example covers the mask language model distilroberta-base 
        https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
'''

import transformers
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling 
import math

def show_random_elements(dataset, num_examples=10):
    """Show random dataset elements

    Args:
        dataset ([type]): [description]
        num_examples (int, optional): [description]. Defaults to 10.
    """

    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    return df


def group_texts(examples):
    block_size = 128
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    # Handling the data 
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    df_random = show_random_elements(datasets["train"])

    # Load the tokenizer for distilroberta-base
    model_checkpoint = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    print(tokenized_datasets["train"][1])          # Look at elements in the dataset 

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    tokenizer.decode(lm_datasets["train"][1]["input_ids"])
    # Load the model for distilgpt2
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="test-clm",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.01
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # In general, perplexity is a measurement of how well a probability model predicts a sample. In the context of Natural Language Processing, perplexity is one way to evaluate language models. inverse of perplexity is how well it's able to predict the next word. lower the perplexity the better. I think. 

if __name__=="__main__":
    main()
    
