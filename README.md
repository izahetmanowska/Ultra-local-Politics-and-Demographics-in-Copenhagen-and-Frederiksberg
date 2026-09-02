# Investigating Ultra-Local Politics and Demographics in Copenhagen and Frederiksberg

## About

This project investigates the relationship between local politics and demographic patterns across polling areas in Copenhagen and Frederiksberg.

Using municipal election data from **2009–2025** alongside demographic, socioeconomic, migration, geospatial, and social media data, the project explores four main areas:

- Predicting election outcomes
- Identifying and characterising electoral swing areas
- Clustering neighbourhoods based on demographic and socioeconomic characteristics
- Exploring the relationship between migration, citizenship composition, and voter turnout

The project combines data from multiple public sources and applies data wrangling, feature engineering, statistical analysis, machine learning, clustering, and geospatial analysis.

## Repository Structure

```text
├── data-wrangling/
│   └── exploration/
├── models/
├── processed-data/
│   └── geodata/
├── scrapers/
└── Report.pdf
```

### `data-wrangling/`

Contains notebooks and scripts used for data cleaning, preprocessing, integration, feature engineering, and exploratory analysis.

### `models/`

Contains the modelling and analytical workflows used throughout the project, including predictive models, clustering, classification, and model evaluation.

### `processed-data/`

Contains processed datasets produced during the data preparation pipeline, including prepared geospatial data.

### `scrapers/`

Contains custom data collection scripts, including the Playwright-based Instagram scraper used to collect publicly available follower and post information.

### `Report.pdf`

The full project report describing the research questions, methodology, modelling approaches, results, limitations, and conclusions.

## Data

The project combines data from several sources:

- Den Danske Valgdatabase
- valg.dk
- Danmarks Statistik
- DAGI geodata
- Public Instagram profiles

Due to the size and structure of the original datasets, raw data is not included in this repository.

**Raw data can be provided upon request.**

## Methods

The project uses a combination of data mining and machine learning techniques, including:

- Data cleaning and schema integration
- Feature engineering and feature selection
- ElasticNet regression
- Random Forest models
- Principal Component Analysis (PCA)
- K-Means clustering
- Hierarchical clustering
- Gaussian Mixture Models (GMM)
- DBSCAN
- SHAP analysis
- Geospatial analysis and visualisation

## Authors

- Gry Højagergaard Lipczak
- Johanne Auener
- Izabela Ewa Hetmanowska
- Louise Holst Andersen
