# Cognitive Warrant: Public Research Materials

Public, curated repository for my LIS 4901 capstone project, *Cognitive Warrant: A Theoretical Framework for Evaluating Semantic Alignment in Knowledge Organization Systems*, as part of my requirements for the Master of Library and Information Science (MLIS) degree at the University of Denver. It examines whether the categorical and hierarchical structure of the Library of Congress Subject Headings (LCSH) corresponds to measurable patterns in human semantic cognition, adapting Trott and Bergen's (2023) hybrid meaning theory to a professionally imposed controlled vocabulary.

**Principal Investigator:** Nicholas Meister

**Institution:** University of Denver, MLIS Program

## Live site

The rendered site (research document, IRB packet, presentation, and the executable analysis pipeline below) is published at **[nicholasmeister.github.io/cognitive-warrant](https://nicholasmeister.github.io/cognitive-warrant/)**.

## What's in this repository

This is a deliberately curated subset of a larger private working repository. It contains the finished, documented materials meant for outside readers — not the full iteration history, raw participant data, or in-progress scripts.

```
.
├── index.qmd, pages/*.qmd   # Quarto source for the site above
├── assets/
│   ├── pdf/                 # research document, IRB packet, presentation, summaries
│   ├── data/                # analysis-ready CSVs (see Data below)
│   └── img/                 # survey mockups, favicon
└── src/analysis/
    ├── cognitive_warrant_pipeline_decluttered_version_of_update2.R
    │                         # the R source behind pages/cognitive-warrant-pipeline.qmd
    └── get_distances.py     # builds the LCSH-augmented distance CSV (see note below)
```

### Data

`assets/data/s2_trials_lcsh_distances.csv` merges Trott and Bergen's (2023) Experiment 2 stimulus materials with a newly constructed mapping of those materials onto LCSH authority records; it is the analytic dataset for [the proof-of-concept pipeline](https://nicholasmeister.github.io/cognitive-warrant/pages/cognitive-warrant-pipeline.html). The `tb2023_*` files are Trott and Bergen's original published materials, included here for convenience.

`get_distances.py` is included for methodological transparency — it documents how `s2_trials_lcsh_distances.csv` was built — but is **not runnable as-is** in this repository: its upstream inputs (the raw LCSH RDF dump, the full stimulus-to-heading mapping pipeline, and Trott and Bergen's raw trial data) live in the private working repository and are not republished here.

The R pipeline (`cognitive_warrant_pipeline_decluttered_version_of_update2.R` / `pages/cognitive-warrant-pipeline.qmd`), by contrast, **is** fully reproducible from this repository alone — it reads only `assets/data/s2_trials_lcsh_distances.csv`.

## Reproducing the analysis pipeline

```r
install.packages(c("tidyverse", "logistf", "lme4", "splines", "patchwork", "knitr"))
```

Then knit `pages/cognitive-warrant-pipeline.qmd` (or `Rscript src/analysis/cognitive_warrant_pipeline_decluttered_version_of_update2.R` for a plain console transcript). See `requirements.txt` for the Python environment `get_distances.py` was written against.

## Building the site locally

```
quarto render
```

requires [Quarto](https://quarto.org) and R with the packages above installed.

## License

Text and code in this repository are licensed under [CC BY 4.0](LICENSE) — reuse with attribution. The included Trott and Bergen (2023) materials (`assets/pdf/trott_bergen_2023*.pdf`, `assets/data/tb2023_*.csv`) remain the authors' own work, reproduced here for reference; see their [published article](https://doi.org/10.1037/rev0000420) for the original terms.

## Questions

Nicholas Meister — [nicholas.meister@du.edu](mailto:nicholas.meister@du.edu)
