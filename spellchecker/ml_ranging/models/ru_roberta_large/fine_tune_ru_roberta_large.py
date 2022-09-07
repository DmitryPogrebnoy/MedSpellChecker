import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import pynvml
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

MODEL_CHECKPOINT = "sberbank-ai/ruRoberta-large"
PATH_TO_PREPROCESSED_ANAMNESIS = "../../../../data/anamnesis/processed/all_anamnesis.csv"

TRAIN_TOKENIZER_DATA_MIN_COUNT_WORD = 2
TRAIN_TOKENIZER_DATA_MIN_LENGTH_WORD = 2
TRAIN_TOKENIZER_BATCH_SIZE = 32

PATH_TO_NEW_TOKENIZER = "../../../../data/ml/ru_roberta_large_finetuned/tokenizer"

CHECK_MODEL_NUMBER_CANDIDATE = 20

TRAIN_PART_ANAMNESIS = 0.8

# model.max_position_embeddings = 514 (but it's a bug and real value is 512)
GROUPING_TEXT_CHUNK_SIZE = 512

MLM_PROBABILITY = 0.15

OUTPUT_DIR_TRAINING_ARG = f"{MODEL_CHECKPOINT}-finetuned"
OVERWRITE_OUTPUT_DIR_TRAINING_ARG = True
BATCH_SIZE_TRAINING_ARG = 256
NUM_EPOCH_TRAINING_ARG = 25
LR_TRAINING_ARG = 5e-5
WEIGHT_DECAY_TRAINING_ARG = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE_TRAINING_ARG = 1
GRADIENT_CHECKPOINTING_TRAINING_ARG = True
PER_DEVICE_EVAL_BATCH_SIZE_TRAINING_ARG = 1
FP16_TRAINING_ARG = True

PATH_TO_SAVE_FINETUNED_MODEL = "../../../../data/ml/ru_roberta_large_finetuned/model"


def setup_random():
    random_state = 100
    random.seed(random_state)


def set_device():
    if torch.cuda.is_available():
        gpus_free_mem_list = []
        for device_num in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_num)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus_free_mem_list.append((info.total - info.used) // 1024 ** 3)
        selected_device_number = np.argmax(gpus_free_mem_list)
        print(selected_device_number)
        torch.cuda.set_device(torch.device(selected_device_number))
        print(f"Selected GPU number: {torch.cuda.current_device()}")
        print(
            f"Will use device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Device has {np.max(gpus_free_mem_list)} Gb free memory")
    else:
        print(f"We will use device: CPU")


def print_gpu_memory_stats():
    current_device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"All GPU memory occupied: {info.used // 1024 ** 3}/{info.total // 1024 ** 3}  Gb.\n")
    print(f"Torch GPU {current_device} memory stats: {torch.cuda.memory_allocated(current_device) // 1024 ** 3} Gb")


def check_tokenizer_behaviour(tokenizer):
    print("\nTokenizer behaviour\nExample text:")
    text = f"ультразвуковой исследование {tokenizer.mask_token} полость"
    print(text)
    print("Decoded text:")
    print(tokenizer.decode(tokenizer(text)["input_ids"]))
    # Tokenizer should return [1, index, 2], but it didn't by default
    word = "хронический"
    print(f"Tokenizer `input_ids` for word '{word}'")
    print(tokenizer(word)["input_ids"])


def get_anamnesis():
    processed_anamnesis = pd.read_csv(PATH_TO_PREPROCESSED_ANAMNESIS, header=None, names=["anamnesis"])
    print(processed_anamnesis[processed_anamnesis['anamnesis'].isnull()])
    # ru-Roberta-large didn't work fine with `ё`, let's replace it with `е`
    processed_anamnesis["anamnesis"] = processed_anamnesis["anamnesis"].map(lambda x: x.replace('ё', 'е'))
    print(f"\nLoaded anamnesis.")
    print(f"Anamnesis number: {len(processed_anamnesis)}")
    print(f"Anamnesis head:")
    print(processed_anamnesis.head())
    return processed_anamnesis


def train_tokenizer(tokenizer, anamnesis):
    def prepare_words(anamnesis):
        print(f"\nPrepare words data for tokenizer training")
        anamnesis_list = anamnesis["anamnesis"].values
        anamnesis_words = np.concatenate(list(map(lambda x: x.split(), anamnesis_list)))
        print(f"All anamnesis words number: {len(anamnesis_words)}")
        unique_anamnesis_words = [val for val, i in Counter(anamnesis_words).items() if
                                  i > TRAIN_TOKENIZER_DATA_MIN_COUNT_WORD and
                                  len(val) > TRAIN_TOKENIZER_DATA_MIN_LENGTH_WORD]
        anamnesis_words = [word for word in anamnesis_words if word in unique_anamnesis_words]
        print(f"Filtered anamnesis words number: {len(anamnesis_words)}")
        return anamnesis_words, len(unique_anamnesis_words)

    def batch_iterator(data):
        for i in range(0, len(data), TRAIN_TOKENIZER_BATCH_SIZE):
            yield data[i: i + TRAIN_TOKENIZER_BATCH_SIZE]

    print(f"\nTrain tokenizer")
    print(f"Default tokenizer `vocab_size`: {tokenizer.vocab_size}")
    words, unique_words_number = prepare_words(anamnesis)
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(words), vocab_size=unique_words_number)
    print(f"New tokenizer `vocab_size`: {new_tokenizer.vocab_size}")
    return new_tokenizer


def check_model_prediction(model, tokenizer, text):
    print("\nCheck model prediction")
    print(f"Text: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Inputs ids: {inputs['input_ids']}")
    print(inputs)
    # Find the location of <mask> and extract its logits
    token_logits = model(**inputs).logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_logits = torch.softmax(mask_token_logits, dim=1)
    # Pick the <mask> candidates with the highest logits
    top_n = torch.topk(mask_token_logits, CHECK_MODEL_NUMBER_CANDIDATE, dim=1)
    top_n_tokens = zip(top_n.indices[0].tolist(), top_n.values[0].tolist())
    print(f"Top {CHECK_MODEL_NUMBER_CANDIDATE} candidates")
    for token, score in top_n_tokens:
        print(f"{text.replace(tokenizer.mask_token, tokenizer.decode([token]))}, score: {score}")


def prepare_datasets(anamnesis):
    np.random.shuffle(anamnesis)
    train = anamnesis[:int((len(anamnesis) + 1) * TRAIN_PART_ANAMNESIS)]
    test = anamnesis[int((len(anamnesis) + 1) * TRAIN_PART_ANAMNESIS):]
    print(f"\nTrain dataset - {len(train)}")
    print(f"Test dataset - {len(test)}")
    train_dataset = Dataset.from_dict({"text": train})
    test_dataset = Dataset.from_dict({"text": test})
    anamnesis_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    print(f"Result dataset - {anamnesis_dataset}")
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

    lm_datasets = tokenized_dataset.map(group_texts, batched=True, batch_size=len(tokenized_dataset["train"]))
    lm_datasets.set_format("pt")
    print(f"\nGroupped text dataset - {lm_datasets}")
    return lm_datasets


def build_training_arguments():
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_TRAINING_ARG,
        overwrite_output_dir=OVERWRITE_OUTPUT_DIR_TRAINING_ARG,
        num_train_epochs=NUM_EPOCH_TRAINING_ARG,
        learning_rate=LR_TRAINING_ARG,
        weight_decay=WEIGHT_DECAY_TRAINING_ARG,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE_TRAINING_ARG,
        gradient_accumulation_steps=BATCH_SIZE_TRAINING_ARG,
        gradient_checkpointing=GRADIENT_CHECKPOINTING_TRAINING_ARG,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE_TRAINING_ARG,
        # no_cuda=True,
        fp16=FP16_TRAINING_ARG
    )

    return training_args


def train_model(model, optimizer, accelerator, train_dataloader, test_dataloader, training_args):
    for epoch in range(training_args.num_train_epochs):
        progress_bar = tqdm(range(len(train_dataloader)))

        print(f"TRAIN EPOCH {epoch}")
        model.train()
        for step, batch in enumerate(train_dataloader, start=1):
            loss = model(**batch).loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            progress_bar.update(1)

        print_gpu_memory_stats()
        print(f"EVAL EPOCH {epoch}")
        progress_bar = tqdm(range(len(test_dataloader)))
        model.eval()
        losses = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(BATCH_SIZE_TRAINING_ARG)))
            progress_bar.update(1)

        losses = torch.cat(losses)
        losses = losses[: len(test_dataloader)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")


def fine_tune_model():
    setup_random()
    pynvml.nvmlInit()

    model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)
    print(f"Model {MODEL_CHECKPOINT} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print(f"Tokenizer {MODEL_CHECKPOINT} loaded.")

    check_tokenizer_behaviour(tokenizer)

    # Let's train our tokenizer to correct tokenize medical words
    # Source https://github.com/huggingface/notebooks/blob/main/examples/tokenizer_training.ipynb

    # We can't use all known words from dictionary,
    # because then the embedding in model become too large
    # and don't fit into the GPU memory(4 GB for NVIDIA GeForce GTX 1650 or 24 GB for Tesla K80 on Colab).
    # So let's use all words from existed anamnesis, and each word should appear at least two times.

    anamnesis = get_anamnesis()
    tokenizer = train_tokenizer(tokenizer, anamnesis)
    check_tokenizer_behaviour(tokenizer)

    # Save new tokenizer to dir
    tokenizer.save_pretrained(PATH_TO_NEW_TOKENIZER)

    # Resize model token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Check model prediction candidates
    check_model_test_text = f"ультразвуковой исследование {tokenizer.mask_token} полость"
    check_model_prediction(model, tokenizer, check_model_test_text)

    anamnesis_list = anamnesis["anamnesis"].values
    dataset = prepare_datasets(anamnesis_list)
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)
    lm_datasets = group_datasets_text(tokenized_dataset)

    print("Example of decoded first text block")
    print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MLM_PROBABILITY)
    training_args = build_training_arguments()

    train_dataloader = DataLoader(lm_datasets["train"],
                                  batch_size=training_args.per_device_train_batch_size,
                                  collate_fn=data_collator)

    test_dataloader = DataLoader(lm_datasets["test"],
                                 batch_size=training_args.per_device_train_batch_size,
                                 collate_fn=data_collator)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    set_device()
    print_gpu_memory_stats()

    accelerator = Accelerator(fp16=training_args.fp16)
    adam_w_optim = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, adam_w_optim, train_dataloader,
                                                                              test_dataloader)
    print_gpu_memory_stats()

    train_model(model, optimizer, accelerator, train_dataloader, test_dataloader, training_args)

    check_model_prediction(model, tokenizer, check_model_test_text)

    model.save_pretrained(PATH_TO_SAVE_FINETUNED_MODEL)


if __name__ == '__main__':
    """
    Fine-tuning Ru-Roberta-large for MLM task on the russian medical text 
    """
    fine_tune_model()
