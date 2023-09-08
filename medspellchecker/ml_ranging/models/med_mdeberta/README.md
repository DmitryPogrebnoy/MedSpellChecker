---
language:

  - license: apache-2.0

---

# Model MedMDebertaV3

# Model Description

This model is fine-tuned version
of [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base/tree/main).
The code for the fine-tuned process can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/spellchecker/ml_ranging/models/med_mdeberta/fine_tune_mdebert_colab.ipynb)
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
>> > pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/MedMDebertaV3')
>> > pipeline("У пациента [MASK] боль в грудине.")
[{'score': 0.05280596762895584,
  'token': 4595,
  'token_str': 'суд',
  'sequence': 'У пациента суд боль в грудине.'},
 {'score': 0.050577640533447266,
  'token': 19157,
  'token_str': 'времени',
  'sequence': 'У пациента времени боль в грудине.'},
 {'score': 0.02754475176334381,
  'token': 19174,
  'token_str': 'препарат',
  'sequence': 'У пациента препарат боль в грудине.'},
 {'score': 0.027341477572917938,
  'token': 125009,
  'token_str': 'рошен',
  'sequence': 'У пациентарошен боль в грудине.'},
 {'score': 0.022251157090067863,
  'token': 19441,
  'token_str': 'енный',
  'sequence': 'У пациентаенный боль в грудине.'}]
```

Or you can load the model and tokenizer and do what you need to do:

```python
>> > from transformers import AutoTokenizer, AutoModelForMaskedLM
>> > tokenizer = AutoTokenizer.from_pretrained("DmitryPogrebnoy/MedMDebertaV3")
>> > model = AutoModelForMaskedLM.from_pretrained("DmitryPogrebnoy/MedMDebertaV3")
```


