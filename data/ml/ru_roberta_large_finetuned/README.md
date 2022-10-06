---
language:

- ru
  license: apache-2.0

---

# Model MedRuRobertaLarge

# Model Description

This model is fine-tuned version of [ruRoberta-large](sberbank-ai/ruRoberta-large).
The code for the fine-tuned process can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/spellchecker/ml_ranging/models/med_ru_roberta_large/fine_tune_ru_roberta_large.py)
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
>> > pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/MedRuRobertaLarge')
>> > pipeline("У пациента <mask> боль в грудине.")
[{'score': 0.2467374950647354,
  'token': 9233,
  'token_str': ' сильный',
  'sequence': 'У пациента сильный боль в грудине.'},
 {'score': 0.16476310789585114,
  'token': 27876,
  'token_str': ' постоянный',
  'sequence': 'У пациента постоянный боль в грудине.'},
 {'score': 0.07211139053106308,
  'token': 19551,
  'token_str': ' острый',
  'sequence': 'У пациента острый боль в грудине.'},
 {'score': 0.0616639070212841,
  'token': 18840,
  'token_str': ' сильная',
  'sequence': 'У пациента сильная боль в грудине.'},
 {'score': 0.029712719842791557,
  'token': 40176,
  'token_str': ' острая',
  'sequence': 'У пациента острая боль в грудине.'}]
```

Or you can load the model and tokenizer and do what you need to do:

```python
>> > from transformers import AutoTokenizer, AutoModelForMaskedLM
>> > tokenizer = AutoTokenizer.from_pretrained("DmitryPogrebnoy/MedRuRobertaLarge")
>> > model = AutoModelForMaskedLM.from_pretrained("DmitryPogrebnoy/MedRuRobertaLarge")
```


