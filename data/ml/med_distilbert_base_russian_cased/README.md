---
language:

- ru
  license: apache-2.0

---

# Model DmitryPogrebnoy/MedDistilBertBaseRuCased

# Model Description

This model is fine-tuned version
of [DmitryPogrebnoy/distilbert-base-russian-cased](https://huggingface.co/DmitryPogrebnoy/distilbert-base-russian-cased)
.
The code for the fine-tuned process can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/spellchecker/ml_ranging/models/med_distilbert_base_russian_cased/fine_tune_distilbert_base_russian_cased.py)
.
The model is fine-tuned on a specially collected dataset of over 30,000 medical anamneses in Russian.
The collected dataset can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/data/anamnesis/processed/all_anamnesis.csv).

This model was created as part of a master's project to develop a method for correcting typos
in medical histories using BERT models as a ranking of candidates.
The project is open source and can be found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker).

# How to Get Started With the Model

You can use the model directly with a pipeline for masked language modeling:

```python
>> > from transformers import pipeline
>> > pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/MedDistilBertBaseRuCased')
>> > pipeline("У пациента [MASK] боль в грудине.")
[{'score': 0.1733243614435196,
  'token': 6880,
  'token_str': 'имеется',
  'sequence': 'У пациента имеется боль в грудине.'},
 {'score': 0.08818087726831436,
  'token': 1433,
  'token_str': 'есть',
  'sequence': 'У пациента есть боль в грудине.'},
 {'score': 0.03620537742972374,
  'token': 3793,
  'token_str': 'особенно',
  'sequence': 'У пациента особенно боль в грудине.'},
 {'score': 0.03438418731093407,
  'token': 5168,
  'token_str': 'бол',
  'sequence': 'У пациента бол боль в грудине.'},
 {'score': 0.032936397939920425,
  'token': 6281,
  'token_str': 'протекает',
  'sequence': 'У пациента протекает боль в грудине.'}]
```

Or you can load the model and tokenizer and do what you need to do:

```python
>> > from transformers import AutoTokenizer, AutoModelForMaskedLM
>> > tokenizer = AutoTokenizer.from_pretrained("DmitryPogrebnoy/MedDistilBertBaseRuCased")
>> > model = AutoModelForMaskedLM.from_pretrained("DmitryPogrebnoy/MedDistilBertBaseRuCased")
```


