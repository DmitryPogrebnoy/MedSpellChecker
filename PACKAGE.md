[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Test](https://github.com/DmitryPogrebnoy/MedSpellChecker/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/DmitryPogrebnoy/MedSpellChecker/actions/workflows/python-test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/LICENSE)

# MedSpellChecker

Fast and effective tool for correcting spelling errors in Russian medical texts.
The tool takes the raw medical text and returns the corrected text in lemmatized form.

This project is in its final stages.
If you have found a bug or would like to improve something, please create an issue or pull request.

## How to install

The project is written in Python 3.9 and works on Python version 3.9 or higher.

- From the [pip repository](https://pypi.org/project/medspellchecker/)
    ```bash
    pip install medspellchecker
    ```

- Or from the sources
    - Install requirements
      ```bash
      pip install -r requirements.txt
      ```
    - Install the package
      ```bash
      pip install .
      ```

> **Note!**
> MedSpellChecker has the [editdistpy]() package as a dependency.
> To install it you will need [Microsoft Visual C++ 14](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (for
> Windows)
> or [build-essential](https://packages.ubuntu.com/focal/build-essential) package (for Linux).

## How to use

An example of using the `medspellchecker' is shown below.

```python
from medspellchecker.tool.medspellchecker import MedSpellchecker
from medspellchecker.tool.distilbert_candidate_ranker import RuDistilBertCandidateRanker

candidate_ranker = RuDistilBertCandidateRanker()
spellchecker = MedSpellchecker(candidate_ranker)
fixed_text = spellchecker.fix_text(
    "У больногодиагностирован инфркт и тубркулез"
)

print(fixed_text)
# -> "У больной диагностировать инфаркт и тубркулез"
```

1) The first two lines import the main class of the package.
2) Lines 3 and 4 then import one of the three available classes to rank the edit candidates.
3) Line 6 creates an instance of the candidate ranking class based on the fine-tuned DistilBert model.
4) Line 7 creates an instance of the class to correct spelling errors.
5) On line 8, the `fix_text` method is called, which takes the raw text and returns the corrected text.
6) Finally line 10 prints the corrected result, which looks like "У больной диагностировать инфаркт и тубркулез".
7) In this way, the package can be used to correct spelling errors in Russian medical texts in a few lines.

### Tips for using the tool

1) Initiate and create MedSpellchecker class once. And then use it many times to correct Russian medical texts.
2) The tool must be used before all the preprocessing steps. This way the various preprocessing steps cannot affect the
   result of the tool correction.
3) Nevertheless, the input raw text must be cleared of all sorts of tags, technical symbols and other trash. The input
   text should be plain text without any structure.
4) Before using the tool on the whole dataset, test how it works on a small chunk to choose the best available candidate
   ranker.
5) If the available candidate rankers do not perform well enough, consider fine-tuning the BERT-based model on your
   dataset and using the candidate ranker on the fine-tuned model. The quality of the corrections should improve.

## Internals

**MedSpellChecker** uses the SymDel algorithm to speed up the generation of correction candidates,
and a fine-tuned BERT-based machine learning model to rank candidates and select the best fit.

This architecture allows each component to be developed almost independently and
the correction process to be implemented flexibly.

* **Spellchecker Manager** - responsible for coordinating other components and implementing high-level logic.
* **Preprocessor** and **PostProcessor** - responsible for splitting the incoming text and assembling the result.
* **Dictionary** - contains a dictionary of correct words, which allows to check the correct word or not.
* **Edit Distance Index** - allows to optimize and speed up the calculation of the editing distance required to generate
  candicates for fixing an incorrect word.
* **Error Model** - responsible for generating candidates for fixing incorrect words.
* **Language Model** - based on the fine-tuned BERT-based model, ranks candidates for fixing and selects the most
  suitable word for correction.

## Supported Candidate Rankers

Candidate rankers rank the candidates to replace the incorrect word and choose the most appropriate one.

The tool contains several rankers based on different BERT-based models.
However, the list of supported rankers can easily be extended with rankers based on any other approach and models.
It is sufficient to implement the `AbstractCandidateRanker` interface.

Available rankers out of the box:

- `RuRobertaCandidateRanker` based
  on [DmitryPogrebnoy/MedRuRobertaLarge](https://huggingface.co/DmitryPogrebnoy/MedRuRobertaLarge) model
- `RuDistilBertCandidateRanker` based
  on [DmitryPogrebnoy/MedDistilBertBaseRuCased](https://huggingface.co/DmitryPogrebnoy/MedDistilBertBaseRuCased) model
- `RuBertTiny2CandidateRanker` based
  on [DmitryPogrebnoy/MedRuBertTiny2](https://huggingface.co/DmitryPogrebnoy/MedRuBertTiny2) model
- `RuBioRobertCandidateRanker` based on [alexyalunin/RuBioRoBERTa](https://huggingface.co/alexyalunin/RuBioRoBERTa)
  model
- `RuBioBertCandidateRanker` based on [alexyalunin/RuBioBERT](https://huggingface.co/alexyalunin/RuBioBERT) model

## More information

This project is part of master's thesis. The current state is the result of the first year of work.
More details about **MedSpellCHecker** you can find in the text of the
[term report](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/presentation_materials/pre_pre_defense/Dmitry_Pogrebnoy_pre_pre_defence.pdf)
.
