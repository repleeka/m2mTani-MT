# m2mTani-MT

**Many-to-many machine translation for five Tani languages of Arunachal Pradesh, Hindi and English.**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-m2mTani--MT-yellow)](https://huggingface.co/repleeka/m2mTani-MT)
[![Demo Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-m2mTani--MT--app-blue)](https://huggingface.co/spaces/repleeka/m2mTani-Translator)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

m2mTani-MT fine-tunes [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) to translate
in **all 42 directions** among **Adi, Apatani, Galo, Nyishi, Tagin** (languages NLLB-200 has never seen),
**Hindi** and **English**, with a single model.

This repository has the **complete fine-tuning notebook**, an inference notebook and a **2,000-pair non-Bible sample**
of the training data. The model and the demo are on Hugging Face:

| | Where | Access |
|---|---|---|
| 🤗 **Model** | [huggingface.co/repleeka/m2mTani-MT](https://huggingface.co/repleeka/m2mTani-MT) | Gated: request access on the model page |
| 🤗 **Demo** | [huggingface.co/spaces/repleeka/m2mTani-MT-app](https://huggingface.co/spaces/repleeka/m2mTani-Translator) | Sign in with a Hugging Face account approved on the model page |
| 📓 **Training** | [`notebooks/01_finetune_m2mTani_MT.ipynb`](notebooks/01_finetune_m2mTani_MT.ipynb) | This repository |

## Languages

| Language | Code | Family | Data used for training |
|---|---|---|---|
| Adi | `adi_Latn` | Tani | Bible (Old and New Testament) |
| Apatani | `apt_Latn` | Tani | Bible (New Testament), dictionary, everyday conversation |
| Galo | `adl_Latn` | Tani | Bible (New Testament), dictionary with example sentences |
| Nyishi | `njz_Latn` | Tani | Bible (Old and New Testament), dictionary, everyday phrases, news |
| Tagin | `tgj_Latn` | Tani | Bible (New Testament), GinLish corpus, dictionary, conversation, news, synthetic |
| Hindi | `hin_Deva` | Indo-Aryan | Bible (Old and New Testament) |
| English | `eng_Latn` | Germanic | Pivot: every text above is paired with English |

The five Tani codes are new tokens added to the NLLB-200 tokenizer.

## Repository layout

```
m2mTani-MT/
├── README.md
├── notebooks/
│   ├── 01_finetune_m2mTani_MT.ipynb   complete training and evaluation pipeline
│   └── 02_inference.ipynb             translate with the released model
├── data/
│   ├── README.md                      data description and Bible source links
│   └── sample_2k_non_bible.parquet    2,000 non-Bible pairs (Apatani, Galo, Nyishi, Tagin)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quick start: translate

```bash
pip install -U torch "transformers>=5,<6" huggingface_hub
hf auth login          # with a token from an account that has been granted access to the model
```

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "repleeka/m2mTani-MT"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()

def translate(texts, src_lang, tgt_lang, num_beams=4):
    tokenizer.src_lang = src_lang
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.inference_mode():
        out = model.generate(**batch, num_beams=num_beams, max_new_tokens=256,
                             forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
    return tokenizer.batch_decode(out, skip_special_tokens=True)

print(translate(["Where are you going?"], "eng_Latn", "tgj_Latn"))   # English -> Tagin
```

See [`notebooks/02_inference.ipynb`](notebooks/02_inference.ipynb) for more examples, including Tani ↔ Tani.

## Fine-tuning

[`notebooks/01_finetune_m2mTani_MT.ipynb`](notebooks/01_finetune_m2mTani_MT.ipynb) is the full experiment in one
notebook. All settings are in its first code cell.

| Step | What it does |
|---|---|
| 1. Settings | Paths, seed, split fractions, sampling and optimisation hyper-parameters |
| 2. Environment | Seeds, library versions, SHA-256 checksums of every input file |
| 3. Load corpora | Groups each source text with all its English translations. Finds Bible verses through `tani_corpus.json` |
| 4. Split | 90 / 5 / 5 by **whole Bible chapter**, the same chapters in every language. Non-Bible text is split by groups linked through shared English |
| 5. Leakage audit | Flags held-out items whose English is in training, or whose source nearly matches a training text (3-gram consonant-skeleton Jaccard ≥ 0.8). Headline scores use clean items only |
| 6. Cache | Tokenizes the training split once and reuses it on restart |
| 7. Batches | Per epoch: up to 3 English versions per source text, both directions. Language temperature α = 0.3. Tani ↔ Tani / Hindi pairs from shared verses |
| 8. Model | Adds the five Tani language tokens, initialised from the mean of `lus_Latn` and `kac_Latn` |
| 9–11. Train | Adafactor, lr 1e-4, 2K warm-up, inverse-sqrt decay, label smoothing 0.1, bf16, ≈16K tokens/step. Dev chrF++ over all 42 directions every 2K steps, best checkpoint kept, early stopping. Resumes after interruption |
| 12. Test | Beam 4. chrF++ and BLEU with 95% bootstrap CIs for every direction, plus an untouched NLLB-200 baseline for Hindi ↔ English |
| 13–14. Report | Training curves, chrF++ heat map and a Markdown report |

The released checkpoint `m2m-tani-nllb-32k-steps` was taken at 32,000 optimizer steps on one NVIDIA RTX 4000 Ada
(20 GB).

### Run it

```bash
git clone https://github.com/repleeka/m2mTani-MT.git
cd m2mTani-MT
pip install -r requirements.txt
# put your corpus in ./mtani-corpora (layout in data/README.md), then:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output experiment_run.ipynb notebooks/01_finetune_m2mTani_MT.ipynb
```

If the kernel stops, run all cells again. Data preparation is cached and training resumes from the last
checkpoint. If the GPU runs out of memory, lower `TOKENS_PER_BATCH` and raise `ACCUM`. Outputs (configuration,
split manifest, leakage audit, logs, predictions and report) are written to `runs/`.

## Data

The full corpus is **not** in this repository: most of it is Bible text, which belongs to its publishers. Instead,
[`data/README.md`](data/README.md) links to every Bible source used, and
[`data/sample_2k_non_bible.parquet`](data/sample_2k_non_bible.parquet) has **2,000 non-Bible pairs** (500 each for
Apatani, Galo, Nyishi and Tagin; all Adi data is Bible text) to show the format and the non-Bible domains. The
sample is filtered against every Bible verse and every English Bible version in the corpus, plus a Biblical
vocabulary filter, so it contains no Bible text.

## Limitations

- Most of the training text is Bible translation, so the model is strongest in that register.
- Tani ↔ Tani and Tani ↔ Hindi directions come from verse-aligned Bible text only.
- Quality differs a lot between languages, because the amount and kind of data differ.
- Tani spelling is not fully standardised. The model follows the spelling of its sources.
- Have output checked by a fluent speaker before using it for anything that matters.

## Citation

```bibtex
@unpublished{dugi2026m2mtanimt,
  title  = {m2m-Tani-MT: Finetuning Multilingual Machine Translation Model for 5 Tani languages of Arunachal Pradesh},
  author = {Dugi, Tungon},
  note   = {Department of Computer Science and Engineering, National Institute of Technology. Manuscript under peer review},
  year   = {2026}
}
```

## License

Code, notebooks, documentation and the data sample: [CC BY-NC 4.0](LICENSE). The model weights on Hugging Face are
also CC BY-NC 4.0, following the base model NLLB-200. Bible texts remain the property of their publishers and are
not redistributed here.

## Acknowledgements

- [Meta AI's NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M), the base model.
- [eBible.org](https://ebible.org), for openly licensed Bible translations used as English and Hindi references.
- The speakers, translators and communities of the Adi, Apatani, Galo, Nyishi and Tagin languages, whose work made
  these corpora possible.
# m2mTani-MT
