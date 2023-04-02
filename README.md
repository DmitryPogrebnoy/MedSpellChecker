[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/LICENSE)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Test](https://github.com/DmitryPogrebnoy/MedSpellChecker/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/DmitryPogrebnoy/MedSpellChecker/actions/workflows/python-test.yml)

# MedSpellChecker

Fast and effective tool for correcting spelling errors in Russian medical texts.
The tool takes the raw medical text and returns the corrected text in lemmatized form.

This project is under active development and is gradually improving.

## Demo

Here is an example of how to correct a spelling mistake with MedSpellChecker.

![Demo](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/presentation_materials/readme/demo/demo_correct_message.gif)

Steps for reproducing the demo:

1. Clone the project
2. Install all requirements
3. Go to `demo` folder
3. Run demo Flask server
4. Open demo website and enjoy!

## Supported errors

**MedSpellChecker** supports fixing the following types of errors.

![Supported errors](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/presentation_materials/figures/misspelling_types.drawio.png)

## Internals

**MedSpellChecker** uses the SymDel algorithm to speed up the generation of correction candidates,
and a fine-tuned BERT-based machine learning model to rank candidates and select the best fit.

The architecture of the **MedSpellChecker** tool is shown below.

![Arch](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/presentation_materials/figures/arch.png)

This architecture allows each component to be developed almost independently and
the correction process to be implemented flexibly.

* **Spellchecker Manager** - responsible for coordinating other components and implementing high-level logic.
* **Preprocessor** and **PostProcessor** - responsible for splitting the incoming text and assembling the result.
* **Dictionary** - contains a dictionary of correct words, which allows to check the correct word or not.
* **Edit Distance Index** - allows to optimize and speed up the calculation of the editing distance required to generate
  candicates for fixing an incorrect word.
* **Error Model** - responsible for generating candidates for fixing incorrect words.
* **Language Model** - based on the fine-tuned RuRoberta model, ranks candidates for fixing and selects the most
  suitable word for correction.

## More information

This project is part of master's thesis. The current state is the result of the first year of work.
More details about **MedSpellCHecker** you can find in the text of the
[term report](https://github.com/DmitryPogrebnoy/MedSpellChecker/blob/main/presentation_materials/summer-report/Dmitry_Pogrebnoy_term_work.pdf)
.
