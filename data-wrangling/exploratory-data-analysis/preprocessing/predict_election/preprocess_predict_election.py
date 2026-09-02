
#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder

# Configuration
ROOT = '../../../'
dem_path = 'processed-data/demographics/'
elec_path = 'processed-data/elections/'

#_____________________________________#
#AGGREGATED GROUPS
#_____________________________________#

AGE_GROUPS = {
    "Children_0_17": [
        "Age_0-4 years", "Age_5-9 years", "Age_10-14 years", "Age_15-17 years",
    ],
    "Youth_18_29": [
        "Age_18-19 years", "Age_20-24 years", "Age_25-29 years",
    ],
    "WorkingAge_30_64": [
        "Age_30-34 years", "Age_35-39 years", "Age_40-44 years",
        "Age_45-49 years", "Age_50-54 years", "Age_55-59 years",
        "Age_60-64 years",
    ],
    "Seniors_65_plus": [
        "Age_65-69 years", "Age_70- years",
    ]
}

EDUCATION_GROUPS = {
    "Education_Long_higher_education": [
        "Education_18-29 years_Long higher education",
        "Education_30-70+ years_Long higher education",
    ],
    "Education_Primary_lower_secondary": [
        "Education_18-29 years_Primary and lower secondary",
        "Education_30-70+ years_Primary and lower secondary"
    ],
    "Education_Vocational_training": [
        "Education_18-29 years_Vocational training",
        "Education_30-70+ years_Vocational training"
    ]
}

CAR_GROUP = {
    "Households_with_car": [
        "Car_Households with 1 car",
        "Car_Households with 2+ cars"
    ]
}

BENEFIT_GROUP = {
    "People_on_benefits": [
        'Benefit type_Foertidspension', 
        'Benefit type_Kontanthjaelp',
    ]
}

DROP_COLUMNS = [
    # Area codes
    'ValgstedId','Valgsted navn','KredsNr','Kreds navn','Kommune navn',
    # Sex
    'Sex_Male',

    # Homes
    'Housing type_Homes_Apartment buildings',
    'Housing type_Homes_Detached house',
    'Housing type_Homes_Terraced house',
    'Housing tenure_Homes_Almen','Housing tenure_Homes_Andel',
    'Housing tenure_Homes_Ejer','Housing tenure_Homes_Private rental',
    'Housing tenure_Total homes',

    # People-housing dropped
    'Housing type_People_Detached house','Housing type_People_Terraced house',

    # Year built
    'Year built_Homes_1960-2009','Year built_Homes_2010-9999',
    'Year built_Homes_Before 1960','Year built_People_1960-2009',
    'Year built_People_2010-9999','Year built_People_Before 1960',

    # Citizenship
    'Citizenship_Africa','Citizenship_Asia and Oceania','Citizenship_Denmark',
    'Citizenship_Former Yugoslavia','Citizenship_New EU countries',
    'Citizenship_Nordic','Citizenship_North America','Citizenship_Old EU countries',
    'Citizenship_Other Europe','Citizenship_South and Central America',
    'Citizenship_Turkey', 'Citizenship_Not specified',

    # Ages (raw)
    "Age_0-4 years","Age_5-9 years","Age_10-14 years","Age_15-17 years",
    "Age_18-19 years","Age_20-24 years","Age_25-29 years","Age_30-34 years",
    "Age_35-39 years","Age_40-44 years","Age_45-49 years","Age_50-54 years",
    "Age_55-59 years","Age_60-64 years","Age_65-69 years","Age_70- years",

    # Education raw
    "Education_18-29 years_Long higher education",
    "Education_30-70+ years_Long higher education",
    "Education_18-29 years_Primary and lower secondary",
    "Education_30-70+ years_Primary and lower secondary",
    "Education_18-29 years_Vocational training",
    "Education_30-70+ years_Vocational training",
    'Education_18-29 years_Not specified',
    'Education_30-70+ years_Not specified',
    'Education_Total 18-29 years','Education_Total 30-70+ years',

    # Cars
    "Car_Households with 1 car","Car_Households with 2+ cars",
    'Car_Households with no car',

    # Benefits
    "Benefit type_Foertidspension", 
    "Benefit type_Kontanthjaelp","Benefit type_Modtager ikke ydelser",

    # Immigrant subgroups
    'Immigrants_Africa','Immigrants_Asia and Oceania','Immigrants_Former Yugoslavia',
    'Immigrants_New EU countries','Immigrants_Nordic','Immigrants_North America',
    'Immigrants_Not specified','Immigrants_Old EU countries','Immigrants_Other Europe',
    'Immigrants_South and Central America','Immigrants_Turkey',

    # Income
    'Income_Median household income'
]

# For election preprocessing
# Parties that has won in polling areas across the years 2009-2021
KEEP_PARTIES = ['A', 'C', 'F', 'Ø', 'V']


#Agregation functions for demografic data

def combine_columns(df, groups: dict):
    """
    Combine multiple sets of columns into aggregated features.
    
    Parameters:
        df (DataFrame)
        groups (dict):
            {
              "NewFeatureName": ["col1", "col2", ...],
            }
    
    Returns:
        DataFrame with aggregated columns added.
    """
    for new_col, cols in groups.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for '{new_col}': {missing}")
        
        df[new_col] = df[cols].sum(axis=1)

    return df

def normalize_income(dfs_dict, income_col='Income_Median household income'):
    """
    Normalize median household income across multiple years using log transform 
    and min-max scaling to 0-100 range (so it matches other features).
    
    Parameters:
        dfs_dict (dict): Dictionary of {year: DataFrame} containing income data
        income_col (str): Name of the income column to normalize
    
    Returns:
        dict: Dictionary of DataFrames with normalized income column added
    
    Example:
        dfs = {
            2009: df_2009,
            2013: df_2013,
            2017: df_2017,
            2021: df_2021
        }
        dfs = normalize_income(dfs)
    """
    # Collecting income values across all years
    all_incomes = pd.concat([
        df[income_col] for df in dfs_dict.values()
    ], ignore_index=True)
    
    # Removing any missing values for min/max calculation
    all_incomes = all_incomes.dropna()
    
    if len(all_incomes) == 0:
        raise ValueError(f"No valid income data found in column '{income_col}'")
    
    # Calculating global log min/max across all years
    all_incomes_log = np.log(all_incomes)
    global_log_min = all_incomes_log.min()
    global_log_max = all_incomes_log.max()
    
    print(f"Global income range: {all_incomes.min():.2f} - {all_incomes.max():.2f}")
    print(f"Global log range: {global_log_min:.4f} - {global_log_max:.4f}")
    
    # Applying normalization to each year
    normalized_dfs = {}
    for year, df in dfs_dict.items():
        df = df.copy()
        
        # Log transform and normalize to 0-100
        df['Income_Median_normalized'] = (
            (np.log(df[income_col]) - global_log_min) / 
            (global_log_max - global_log_min)
        ) * 100
        
        print(f"Year {year}: Normalized income range {df['Income_Median_normalized'].min():.2f} - {df['Income_Median_normalized'].max():.2f}")
        
        normalized_dfs[year] = df
    
    return normalized_dfs


def preprocess_demographics(df):
    """
    Performs all demographic feature engineering:
      - Age groups
      - Education groups
      - Car ownership
      - Benefit groups
      - Drops unwanted raw columns
    """
    # Combined groups
    GROUP_MAPPINGS = {
        **AGE_GROUPS,
        **EDUCATION_GROUPS,
        **CAR_GROUP,
        **BENEFIT_GROUP
    }
    
    # Add aggregated columns
    df = combine_columns(df, GROUP_MAPPINGS)

    # Drop raw leakage columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    return df


# Preprocessing for election data

def get_party_columns(df, start_after='Valid votes'):
    cols = df.columns.tolist()
    if start_after not in cols:
        raise ValueError(f"Column '{start_after}' not found")

    start_idx = cols.index(start_after) + 1
    return cols[start_idx:]

def add_winning_party(df, start_after='Valid votes'):
    df = df.copy()
    party_cols = get_party_columns(df, start_after)

    if not party_cols:
        raise ValueError("No party columns found after 'Valid votes'")

    df['winning_party'] = df[party_cols].idxmax(axis=1)
    df['winning_votes'] = df[party_cols].max(axis=1)

    return df


#REDUNDANT after regression shift:
"""def winning_party_encoder(df, party_categories = KEEP_PARTIES):
    df = df.copy()

    df["winning_party"] = pd.Categorical(
        df["winning_party"],
        categories=party_categories
    )
    df["winning_party_encoded"] = df["winning_party"].cat.codes

    if (df["winning_party_encoded"] == -1).any():
        unseen = df.loc[df["winning_party_encoded"] == -1, "winning_party"].unique()
        raise ValueError(f"Unseen parties detected: {unseen}")

    return df"""

def preprocess_election(df):
    start_after='Valid votes'

    party_cols = get_party_columns(df, start_after)

    df = add_winning_party(df, start_after=start_after)

    #df = winning_party_encoder(df, KEEP_PARTIES)

    party_cols_drop = [p for p in party_cols if p not in KEEP_PARTIES]

    drop_columns = [*party_cols_drop,
        'PollingAreaID',
        'Name',
        'DistrictNo',
        'District',
        'Eligible voters',
        'Other invalid votes',
        'Valid votes',
        'Voter turnout',
        'Blank votes'
        ]
    
    # Drop raw columns
    df = df.drop(columns=drop_columns, errors="ignore")

    df = df.rename(columns={p: f"{p}_vote_share" for p in KEEP_PARTIES})

    return df
