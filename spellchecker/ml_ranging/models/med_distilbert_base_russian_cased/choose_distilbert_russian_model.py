import math
import random

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, DataCollatorForWholeWordMask, \
    set_seed

from gpu_utils import set_device, print_gpu_memory_stats

MULTILANG_DISTILBERT_CHECKPOINT = "distilbert-base-multilingual-cased"
OWN_RUSSIAN_DISTILBERT_CHECKPOINT = "DmitryPogrebnoy/distilbert-base-russian-cased"
GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT = "Geotrend/distilbert-base-ru-cased"

PATH_TO_PREPROCESSED_ANAMNESIS = "../../../../data/anamnesis/processed/all_anamnesis.csv"

TRAIN_TOKENIZER_DATA_MIN_COUNT_WORD = 2
TRAIN_TOKENIZER_DATA_MIN_LENGTH_WORD = 2
TRAIN_TOKENIZER_BATCH_SIZE = 32

CHECK_MODEL_NUMBER_CANDIDATE = 20

TRAIN_PART_ANAMNESIS = 0.8

# model.max_position_embeddings = 514 (but it's a bug and real value is 512)
GROUPING_TEXT_CHUNK_SIZE = 512

MLM_PROBABILITY = 0.15

BATCH_SIZE_TRAINING_ARG = 256
PER_DEVICE_EVAL_BATCH_SIZE_TRAINING_ARG = 1
FP16_TRAINING_ARG = True


def setup_random():
    random_state = 100
    random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    set_seed(random_state)

def get_anamnesis():
    processed_anamnesis = pd.read_csv(PATH_TO_PREPROCESSED_ANAMNESIS, header=None, names=["anamnesis"])
    print(processed_anamnesis[processed_anamnesis['anamnesis'].isnull()])
    print(f"\nLoaded anamnesis.")
    print(f"Anamnesis number: {len(processed_anamnesis)}")
    print(f"Anamnesis head:")
    print(processed_anamnesis.head())
    return processed_anamnesis


def prepare_datasets(anamnesis):
    np.random.shuffle(anamnesis)
    test_dataset = Dataset.from_dict({"text": anamnesis})
    anamnesis_dataset = DatasetDict({"test": test_dataset})
    return anamnesis_dataset


def tokenize_dataset(tokenizer, dataset):
    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        # if tokenizer.is_fast:
        #    result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    return tokenized_datasets


def group_datasets_text(tokenized_dataset):
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples['input_ids'])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // GROUPING_TEXT_CHUNK_SIZE) * GROUPING_TEXT_CHUNK_SIZE
        # Split by chunks of max_len
        result = {
            k: [t[i: i + GROUPING_TEXT_CHUNK_SIZE] for i in range(0, total_length, GROUPING_TEXT_CHUNK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_dataset.map(group_texts, batched=True, batch_size=len(tokenized_dataset["test"]))
    lm_datasets.set_format("pt")
    print(f"\nGroupped text dataset - {lm_datasets}")
    return lm_datasets


def get_model_metrics(model, accelerator, test_dataloader):
    progress_bar = tqdm(range(len(test_dataloader)))
    model.eval()
    test_losses = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        test_losses.append(accelerator.gather(loss.repeat(BATCH_SIZE_TRAINING_ARG)))
        progress_bar.update(1)

    test_losses = torch.cat(test_losses)
    test_losses = test_losses[: len(test_dataloader)]
    test_mean_loss = torch.mean(test_losses).cpu().detach().numpy()

    try:
        test_perplexity = math.exp(torch.mean(test_losses))
    except OverflowError:
        test_perplexity = float("inf")

    return test_perplexity, test_mean_loss


def compute_model_metrics(model, tokenizer, dataset, is_gpu_used, name):
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)
    lm_datasets = group_datasets_text(tokenized_dataset)
    print(f"Example of decoded first text block for {name} tokenizer")
    print(tokenizer.decode(lm_datasets["test"][0]["input_ids"]))

    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=MLM_PROBABILITY)

    test_dataloader = DataLoader(lm_datasets["test"], batch_size=PER_DEVICE_EVAL_BATCH_SIZE_TRAINING_ARG,
                                 collate_fn=data_collator)

    accelerator = Accelerator(mixed_precision="fp16" if FP16_TRAINING_ARG else None,
                              cpu=False if is_gpu_used else True)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    return get_model_metrics(model, accelerator, test_dataloader)


def compute_metric_for_each_model():
    setup_random()

    multilang_model = DistilBertForMaskedLM.from_pretrained(MULTILANG_DISTILBERT_CHECKPOINT)
    print(f"Model {MULTILANG_DISTILBERT_CHECKPOINT} loaded.")
    multilang_tokenizer = DistilBertTokenizer.from_pretrained(MULTILANG_DISTILBERT_CHECKPOINT)
    print(f"Tokenizer {MULTILANG_DISTILBERT_CHECKPOINT} loaded.")

    own_model = DistilBertForMaskedLM.from_pretrained(OWN_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Model {OWN_RUSSIAN_DISTILBERT_CHECKPOINT} loaded.")
    own_tokenizer = DistilBertTokenizer.from_pretrained(OWN_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Tokenizer {OWN_RUSSIAN_DISTILBERT_CHECKPOINT} loaded.")

    geotrend_model = DistilBertForMaskedLM.from_pretrained(GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Model {GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT} loaded.")
    geotrend_tokenizer = DistilBertTokenizer.from_pretrained(GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Tokenizer {GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT} loaded.")

    anamnesis = get_anamnesis()

    anamnesis_list = anamnesis["anamnesis"].values
    dataset = prepare_datasets(anamnesis_list)

    is_gpu_used = set_device()
    print_gpu_memory_stats()

    multilang_perplexity, multilang_loss = compute_model_metrics(multilang_model, multilang_tokenizer, dataset,
                                                                 is_gpu_used, MULTILANG_DISTILBERT_CHECKPOINT)
    print(f"Miltilang-distilbert metrics on the anamnesis dataset")
    print(f"Preplexity {multilang_perplexity}")
    print(f"Mean loss {multilang_loss}")

    own_perplexity, own_loss = compute_model_metrics(own_model, own_tokenizer, dataset,
                                                     is_gpu_used, OWN_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Own-distilbert metrics on the anamnesis dataset")
    print(f"Preplexity {own_perplexity}")
    print(f"Mean loss {own_loss}")

    geotrend_perplexity, geotrend_loss = compute_model_metrics(geotrend_model, geotrend_tokenizer, dataset,
                                                               is_gpu_used, GEOTREND_RUSSIAN_DISTILBERT_CHECKPOINT)
    print(f"Geotrend-distilbert metrics on the anamnesis dataset")
    print(f"Preplexity {geotrend_perplexity}")
    print(f"Mean loss {geotrend_loss}")


if __name__ == '__main__':
    """
    Show perplexity and loss for multilang-distilbert, own russian-distilbert and Geotrend russian-distilbert
    """
    compute_metric_for_each_model()
