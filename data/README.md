# Data

m2mTani-MT was trained on ≈2.95M parallel pairs (each text in a Tani language or Hindi paired with one or more
English translations). **Most of it is Bible text, which is not redistributed here.** This folder has:

1. [`sample_2k_non_bible.parquet`](sample_2k_non_bible.parquet): **2,000 non-Bible pairs** in Apache Parquet
   format, to show the data format and the non-Bible domains.
2. Links to every **Bible source** used, so you can obtain the texts from their publishers.

## 1. Non-Bible sample: `sample_2k_non_bible.parquet`

Load it with pandas or Hugging Face `datasets`:

```python
import pandas as pd
df = pd.read_parquet("data/sample_2k_non_bible.parquet")      # needs pyarrow: pip install pyarrow

from datasets import load_dataset
ds = load_dataset("parquet", data_files="data/sample_2k_non_bible.parquet", split="train")
```

| Column | Description |
|---|---|
| `language` | Apatani, Galo, Nyishi or Tagin |
| `lang_code` | Model language code (`apt_Latn`, `adl_Latn`, `njz_Latn`, `tgj_Latn`) |
| `source_text` | Text in the Tani language |
| `english` | English translation |

| Language | Pairs | Content |
|---|---:|---|
| Apatani | 500 | Dictionary entries, everyday sentences, travel and flight-booking conversation |
| Galo | 500 | Dictionary example sentences and phrases, everyday sentences |
| Nyishi | 500 | Words, phrases and short everyday sentences |
| Tagin | 500 | Sentences with names, dates and places, dictionary entries with sense numbers, everyday and travel conversation, emotion sentences, traditional-knowledge terms |
| **Total** | **2,000** | |

**Adi is not in the sample:** every Adi pair in the training corpus is Bible text.

How the sample was drawn (seed 13, 500 pairs per language):

- Rows were dropped if their source text matched any Bible verse in the corpus, or their English matched any of the
  ≈586K English Bible verse texts (all versions).
- Rows were dropped if they contained verse references (`3:16`) or Biblical vocabulary (God, Lord, Jesus, Israel,
  disciples, thee, unto, behold, Gospel book names…).
- Files that mix Bible verses with other text were excluded entirely, as were machine-generated (KIFS),
  AI-generated and WMT shared-task data.
- Flight-booking templates whose flight number differs between the two sides (misaligned pairs) were dropped.
- Duplicates on either side were removed, and both sides were kept between 2 and 300 characters.

Spelling is kept as in the sources. Apatani uses `’` for the glottal stop. Tagin dictionary entries carry a sense
number, for example `ajar ( 2 )`.

### Full training-data layout

The fine-tuning notebook expects one CSV per language, with columns `<language>, english, source`, plus
`tani_corpus.json` with Bible verse references:

```
mtani-corpora/
├── Adi/adi_english_parallel.csv
├── Apatani/apatani_english_parallel.csv
├── Galo/galo_english_parallel.csv
├── Hindi/hindi_english_parallel.csv
├── Nyishi/nyishi_english_parallel.csv
├── Tagin/tagin_english_parallel.csv
└── tani_corpus.json
```

To try the pipeline on the sample, write one CSV per language in this layout, with the Tani text in a column named
after the language (e.g. `tagin`), `english`, and any value in `source`.

| Language | Pairs in the full training corpus |
|---|---:|
| Adi | 565,218 |
| Apatani | 176,388 |
| Galo | 171,894 |
| Nyishi | 585,364 |
| Tagin | 363,606 |
| Hindi | 1,089,496 |

## 2. Bible sources (links only, not redistributed)

Bible verses give the corpus its multi-way alignment: the same verse in several languages yields Tani ↔ Tani and
Tani ↔ Hindi pairs. Please obtain the texts from their publishers and respect their licences.

### Tani and Hindi Bibles

| Language | Text | Where to find it |
|---|---|---|
| Adi | Adi Bible (Old and New Testament) | [bible.com: Adi](https://www.bible.com/languages/adi) |
| Apatani | Apatani New Testament (*Niti Gontu*, Bible Society of India) | [bible.com: Apatani NT](https://www.bible.com/versions/1529) |
| Galo | Galo New Testament | [bible.com: Galo](https://www.bible.com/languages/adl) |
| Nyishi | Nyishi Bible (Old and New Testament) | [bible.com: Nyishi](https://www.bible.com/languages/njz) |
| Tagin | *Anwnv Gamlv*, Tagin New Testament © 2018 The Word for the World International NECTAR, CC BY 4.0 | [ebible.org/tgj](https://ebible.org/tgj/) |
| Hindi | Indian Revised Version (IRV) Hindi Bible 2019, CC BY-SA 4.0 | [ebible.org/hin2017](https://ebible.org/hin2017/) |
| Hindi | Hindi Contemporary Version © Biblica | [ebible.org/hincv](https://ebible.org/hincv/) |

### English Bibles (pivot side)

Each Tani and Hindi verse was paired with openly licensed English versions from [eBible.org](https://ebible.org).
Main versions:

| Version | Link |
|---|---|
| American Standard Version (1901) | [eng-asv](https://ebible.org/eng-asv/) |
| Berean Standard Bible | [engbsb](https://ebible.org/engbsb/) |
| World English Bible | [eng-web](https://ebible.org/eng-web/) |
| Bible in Basic English | [engBBE](https://ebible.org/engBBE/) |
| Geneva Bible 1599 | [enggnv](https://ebible.org/enggnv/) |
| Douay-Rheims 1899 | [engDRA](https://ebible.org/engDRA/) |
| Darby Translation | [engDBY](https://ebible.org/engDBY/) |
| Young's Literal Translation | [engylt](https://ebible.org/engylt/) |
| King James Version | [eng-kjv](https://ebible.org/eng-kjv/) |
| Translation for Translators | [eng-t4t](https://ebible.org/eng-t4t/) |
| Free Bible Version | [engfbv](https://ebible.org/engfbv/) |
| Literal Standard Version | [englsv](https://ebible.org/englsv/) |
| NET Bible | [engnet](https://ebible.org/engnet/) |
| Unlocked Literal Bible | [engULB](https://ebible.org/engULB/) |
| Orthodox Jewish Bible | [engojb](https://ebible.org/engojb/) |
| Wycliffe Bible (modern spelling) | [engwyc2017](https://ebible.org/engwyc2017/) |
| Open English Bible | [engoebcw](https://ebible.org/engoebcw/) |

The full list is at [ebible.org/find](https://ebible.org/find/). An English version that was 90% or more identical,
word for word, to one already kept was left out.

## Licence

The sample is released under [CC BY-NC 4.0](../LICENSE). Bible texts remain the property of their publishers.
