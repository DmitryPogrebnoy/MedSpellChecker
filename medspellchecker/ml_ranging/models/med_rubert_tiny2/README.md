---
language:

- ru
  license: apache-2.0

---

# Model DmitryPogrebnoy/MedRuBertTiny2

# Model Description

This model is fine-tuned version
of [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)
.
The code for the fine-tuned process can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/spellchecker/ml_ranging/models/med_rubert_tiny2/fine_tune_rubert_tiny2.py)
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
>> > pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/MedRuBertTiny2')
>> > pipeline("У пациента [MASK] боль в грудине.")
[{'score': 0.4527082145214081,
  'token': 29626,
  'token_str': 'боль',
  'sequence': 'У пациента боль боль в грудине.'},
 {'score': 0.05768931284546852,
  'token': 46275,
  'token_str': 'головной',
  'sequence': 'У пациента головной боль в грудине.'},
 {'score': 0.02957102842628956,
  'token': 4674,
  'token_str': 'есть',
  'sequence': 'У пациента есть боль в грудине.'},
 {'score': 0.02168550342321396,
  'token': 10030,
  'token_str': 'нет',
  'sequence': 'У пациента нет боль в грудине.'},
 {'score': 0.02051634155213833,
  'token': 60730,
  'token_str': 'болит',
  'sequence': 'У пациента болит боль в грудине.'}]
```

Or you can load the model and tokenizer and do what you need to do:

```python
>> > from transformers import AutoTokenizer, AutoModelForMaskedLM
>> > tokenizer = AutoTokenizer.from_pretrained("DmitryPogrebnoy/MedRuBertTiny2")
>> > model = AutoModelForMaskedLM.from_pretrained("DmitryPogrebnoy/MedRuBertTiny2")
```


