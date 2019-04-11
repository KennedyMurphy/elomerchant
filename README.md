ELOMerchant
==============================

Category recommendation competition from [Kaggle](https://www.kaggle.com/c/elo-merchant-category-recommendation).

Commands
==============================

`make requirements`
------------
Installs the required python libraries.

`make data`
------------
Before running `make data`, make sure that the required data sets for this competition are installed in `data/raw` using [kaggle](https://github.com/Kaggle/kaggle-api) API. 

Once data sets are downloaded, `make data` can be run. Unsurprisingly, blindly using [Featuretools](https://docs.featuretools.com/) to generate model features does not provide great results. The shotgun approach to Featuretools does not seem appropriate for this competition given that the dataset is sensitive to the selection of features amongst other things (see [this kaggle discussion](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/75935#latest-476146)).

With that in mind, `make data` generates 8 kinds of datasets as outlined by one of the [top 10 participant(s) of the competition](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82055#latest-483943):

```
1. Only historical data
2. Only historical data with authorized_flag=1
3. Only new data
4. Merge of historical data with authorized_flag=1 and new data
5. Merge of historical data and new data
6. Merge of historical data and merchants data
7. Merge of new data and merchants data
8. Merge of historical data and new data and merchants data
```

Project Organization
==============================

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
