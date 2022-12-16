---
language:

- ru
  license: apache-2.0

---

# Model DmitryPogrebnoy/distilbert-base-russian-cased

# Model Description

This model is russian version
of [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased).
The code for the transforming process can be
found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/spellchecker/ml_ranging/models/distilbert_base_russian_cased/distilbert_from_multilang_to_ru.ipynb)
.

This model give exactly the same representations produced by the original model which preserves the original accuracy.
There is a similar model
of [Geotrend/distilbert-base-ru-cased](https://huggingface.co/Geotrend/distilbert-base-ru-cased).
However, our model is derived from a slightly different approach.
Instead of using wikipedia's Russian dataset to pick the necessary tokens,
we used regular expressions in this model to select only Russian tokens, punctuation marks, numbers and other service
tokens.
Thus, our model contains several hundred tokens, which have been filtered out
in [Geotrend/distilbert-base-ru-cased](https://huggingface.co/Geotrend/distilbert-base-ru-cased).

This model was created as part of a master's project to develop a method for correcting typos
in medical histories using BERT models as a ranking of candidates.
The project is open source and can be found [here](https://github.com/DmitryPogrebnoy/MedSpellChecker).

# How to Get Started With the Model

You can use the model directly with a pipeline for masked language modeling:

```python
>> > from transformers import pipeline
>> > pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/distilbert-base-russian-cased')
>> > pipeline("Я [MASK] на заводе.")
[{'score': 0.11498937010765076,
  'token': 1709,
  'token_str': 'работал',
  'sequence': 'Я работал на заводе.'},
 {'score': 0.07212855666875839,
  'token': 12375,
  'token_str': '##росла',
  'sequence': 'Яросла на заводе.'},
 {'score': 0.03575785085558891,
  'token': 4059,
  'token_str': 'находился',
  'sequence': 'Я находился на заводе.'},
 {'score': 0.02496381290256977,
  'token': 5075,
  'token_str': 'работает',
  'sequence': 'Я работает на заводе.'},
 {'score': 0.020675526931881905,
  'token': 5774,
  'token_str': '##дро',
  'sequence': 'Ядро на заводе.'}]
```

Or you can load the model and tokenizer and do what you need to do:

```python
>> > from transformers import AutoTokenizer, AutoModelForMaskedLM
>> > tokenizer = AutoTokenizer.from_pretrained("DmitryPogrebnoy/distilbert-base-russian-cased")
>> > model = AutoModelForMaskedLM.from_pretrained("DmitryPogrebnoy/distilbert-base-russian-cased")
```


