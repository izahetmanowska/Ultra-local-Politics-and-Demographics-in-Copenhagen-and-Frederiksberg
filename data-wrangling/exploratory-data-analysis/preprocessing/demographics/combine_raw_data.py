import pandas as pd
import numpy as np
import re

# Configuration
ROOT = '../../../'
PARENT_PATH = ROOT + 'raw-data/historical-elections/'
PATH = PARENT_PATH + 'demographics/Population_'
OUTPUT_PATH = ROOT + 'processed-data/demographics/'
YEARS = [2001, 2005, 2009, 2013, 2017, 2021] # local election years
MUNICIPALITY_IDS = [101, 147] # Copenhagen and Frederiksberg
COMMON_COLS = ['Gruppe', 'ValgstedId', 'KredsNr', 'StorKredsNr', 'LandsdelsNr']
PREFIX = 'KV'

def find_polling_areas():
    df_geo = pd.read_csv(PARENT_PATH + 'election-results/Geography.csv', sep=';')
    cph_areas = set(df_geo[df_geo['KommuneNr'].isin(MUNICIPALITY_IDS)]['Valgsted Id'].astype(str))
    return cph_areas

CPH_AREAS = find_polling_areas()


# ============================================================================
# SIMPLE
# ============================================================================

# Dict of dicts to define what columns to get from each demographics CSV
SIMPLE_CSVS = {
    'PersonsByBenefitType.csv': {
        'pattern': ' - Personer efter forsørgelsestype_',
        'columns': {
            'Kontanthjaelp': '05. Kontanthjælp',
            'Foertidspension': '08. Førtidspension',
            'Folkepension': '10. Folkepension',
            'Modtager ikke ydelser': '11. Modtager ikke ydelser',
            'Antal personer i alt': '12. Antal personer i alt'
        },
        'prefix': 'Benefit type'
    },
    'CrimeArrests.csv': {
        'pattern': ' - Kriminalitet B. Anholdelser_',
        'columns': {
            'Number of arrests': 'Antal anholdelser'
        },
        'prefix': 'Crime'
    },
    'HouseholdIncomes.csv': {
        'pattern': ' - Husstandsindkomster fordelt på afstemningsområder_',
        'columns': {
            'Median household income': '50%-percentil for husstandsindkomst',
            'Total households': 'Antal husstande i alt',
            'Gross income top20pct': 'Bruttoindkomst for de 20% der tjener mest'
        },
        'prefix': 'Income'
    },
    'HouseholdsCarOwnership.csv': {
        'pattern': ' - Husstande efter bilrådighed_',
        'columns': {
            'Households with no car': '1. Husstande uden bil',
            'Households with 1 car': '2. Husstande med 1 bil',
            'Households with 2+ cars': '3. Husstande med 2 eller flere biler'
        },
        'prefix': 'Car'
    }
}


# ============================================================================
# SOCIOECONOMY AND INDUSTRY
# ============================================================================

def inspect_socioeconomy_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in socioeconomy data to understand differences.
    Prints which columns have data and how they differ.
    """
    csv_path = PATH + 'PersonsBySocioeconomyIndustry.csv'
    pattern = f'{PREFIX}{year} - Socio-økonomisk status og brancher fordelt på afstemningsområder_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = {
        '01. Selvstændig og medhj.': [],
        '02. Topledere': [],
        '07. Arbejdsløse': []
    }

    # get relevant columns
    for col in df.columns:
        if pattern in col:
            for category in categories.keys():
                if f'{pattern}{category}_' in col:
                    categories[category].append(col)
        
    print(f"\n{'='*80}")
    print(f"SOCIOECONOMY DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    for category, cols in categories.items():
        if not cols:
            print(f"{category}: No columns found")
            continue
                
        print(f"\n{category}:")
        print(f"  Found {len(cols)} columns")

        # normalise industry name and group by industry names
        industry_groups = {}
        for col in cols:
            suffix = col.split(f'{pattern}{category}_')[1]
            normalised = re.sub(r'^\d+\.\s*', '', suffix) # regex - 1 or more digits followed by . and 0 or more whitespace chars
            if normalised not in industry_groups:
                industry_groups[normalised] = []
            industry_groups[normalised].append(col)

        # find duplicates
        duplicates = {k: v for k, v in industry_groups.items() if len(v) > 1}
        singles = {k: v for k, v in industry_groups.items() if len(v) == 1}
        print(f"  Industries with duplicates: {len(duplicates)}")
        print(f"  Industries with single column: {len(singles)}")

        if duplicates:
            print(f"\n  Analyzing {len(duplicates)} duplicate pairs:\n")
            
            for industry, duplicate_cols in duplicates.items():
                print(f"  Industry: {industry}")
                
                # compare the two columns
                for i, col in enumerate(duplicate_cols, 1):
                    # Get the suffix to show which version this is
                    suffix = col.split(f'{pattern}{category}_')[1]
                    is_numbered = bool(re.match(r'^\d+\.', suffix))
                    
                    # stats
                    non_null_mask = df[col].notna() & (df[col].astype(str).str.strip() != '-') # '-' is treated as null
                    non_null = non_null_mask.sum()
                    non_null = df[col].notna().sum()

                    vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), 
                                       errors='coerce')
                    non_zero = (vals != 0).sum()
                    total = vals.sum()
                    mean = vals.mean()
                    
                    print(f"    Version {i} ({'numbered' if is_numbered else 'no number'}):")
                    print(f"      Non-null: {non_null}/{len(df)}, Non-zero: {non_zero}, Sum: {total:.2f}, Mean: {mean:.2f}")
                    print(f"      Sample values: {vals.head(sample_size).tolist()[:3]}")
                
                # check if values are identical
                if len(duplicate_cols) == 2:
                    vals1 = pd.to_numeric(df[duplicate_cols[0]].astype(str).str.replace(',', '.'), 
                                         errors='coerce').fillna(0)
                    vals2 = pd.to_numeric(df[duplicate_cols[1]].astype(str).str.replace(',', '.'), 
                                         errors='coerce').fillna(0)
                    
                    identical = (vals1 == vals2).all()

                    # only calculate correlation if there's variation in the data
                    if vals1.std() > 0 and vals2.std() > 0:
                        correlation = vals1.corr(vals2)
                        print(f"    Comparison:")
                        print(f"      Identical: {identical}")
                        print(f"      Correlation: {correlation:.4f}")
                        print(f"      Sum difference: {abs(vals1.sum() - vals2.sum()):.2f}")
                    else:
                        print(f"    Comparison:")
                        print(f"      Identical: {identical}")
                        print(f"      Correlation: Cannot calculate (no variation in data)")
                        print(f"      Sum difference: {abs(vals1.sum() - vals2.sum()):.2f}")
            
                print()
        else:
            print(f"  No duplicates found - all industries have unique columns")
    return categories

def inspect_citizenship_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in age/citizenship data to understand differences.
    Prints which columns have data and how they differ.
    """
    csv_path = PATH + 'PersonsByCitizenshipSexAge.csv'
    pattern = f'{PREFIX}{year} - Antal personer opgjort efter statsborgerskab køn og aldersgrupper_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = {
        '01.': [], # Danmark
        '02.': [], # Nordiske lande
        '03.': [], # Tyrkiet
        '04.': [], # Tidligere Jugoslavien
        '05.': [], # Gamle EU-lande
        '06.': [], # Nye EU-lande
        '07.': [], # Øvrige Europa
        '08.': [], # Afrika
        '09.': [], # Nordamerika
        '10.': [], # Syd- og Mellemamerika
        '11.': [], # Asien og Oceanien
        '12.': [] # Uoplyst/statsløse
    }

    # get relevant columns (structure is pattern + sex + space + age + underscore + category + citizenship)
    citizenship_cols = {}
    for category in categories.keys():
        citizenship_cols[category] = []
    
    for col in df.columns:
        if pattern in col:
            # check if it has the category number somewhere after the age group:
            for category in categories.keys():
                if f'_{category}' in col:
                    citizenship_cols[category].append(col)
                    break
        
    print(f"\n{'='*80}")
    print(f"CITIZENSHIP DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    for category, cols in citizenship_cols.items():
        if not cols:
            print(f"{category} {categories[category]}: No columns found")
            continue
                
        print(f"\n{category} {categories[category]}:")
        print(f"  Found {len(cols)} columns")

        # normalise citizenship name and group by citizenship
        citizenship_groups = {}
        for col in cols:
            parts = col.split('_')
            suffix = parts[-1] # only get e.g. 'Danmark' and ' Danmark'
            normalised = suffix.replace('. ', '.') # remove space
            if normalised not in citizenship_groups:
                citizenship_groups[normalised] = []
            citizenship_groups[normalised].append(col)

        # find duplicates
        duplicates = {k: v for k, v in citizenship_groups.items() if len(v) > 1}
        singles = {k: v for k, v in citizenship_groups.items() if len(v) == 1}
        print(f"  Citizenship variants with duplicates: {len(duplicates)}")
        print(f"  Citizenship variants with single column: {len(singles)}")

        if duplicates:
            print(f"\n  Analyzing {len(duplicates)} duplicate pairs:\n")
            
            for citizenship, duplicate_cols in duplicates.items():
                print(f"  Citizenship: {citizenship}")
                
                # compare duplicate columns
                for i, col in enumerate(duplicate_cols, 1):
                    suffix = col.split('_')[-1]
                    has_space = '. ' in suffix
   
                    # stats
                    non_null_mask = df[col].notna() & (df[col].astype(str).str.strip() != '-') # '-' is treated as null
                    non_null = non_null_mask.sum()

                    vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    non_zero = (vals != 0).sum()
                    total = vals.sum()
                    mean = vals.mean()
                    
                    print(f"    Version {i} ({'with space' if has_space else 'no space'}):")
                    print(f"      Non-null: {non_null}/{len(df)}, Non-zero: {non_zero}, Sum: {total:.2f}, Mean: {mean:.2f}")
                    print(f"      Sample values: {vals.head(sample_size).tolist()[:3]}")
                
                # check if values are identical
                if len(duplicate_cols) == 2:
                    vals1 = pd.to_numeric(df[duplicate_cols[0]].astype(str).str.replace(',', '.'), 
                                         errors='coerce').fillna(0)
                    vals2 = pd.to_numeric(df[duplicate_cols[1]].astype(str).str.replace(',', '.'), 
                                         errors='coerce').fillna(0)
                    
                    identical = (vals1 == vals2).all()

                    # only calculate correlation if there's variation in the data
                    if vals1.std() > 0 and vals2.std() > 0:
                        correlation = vals1.corr(vals2)
                        print(f"    Comparison:")
                        print(f"      Identical: {identical}")
                        print(f"      Correlation: {correlation:.4f}")
                        print(f"      Sum difference: {abs(vals1.sum() - vals2.sum()):.2f}")
                    else:
                        print(f"    Comparison:")
                        print(f"      Identical: {identical}")
                        print(f"      Correlation: Cannot calculate (no variation in data)")
                        print(f"      Sum difference: {abs(vals1.sum() - vals2.sum()):.2f}")
            
                print()
        else:
            print(f"  No duplicates found")
    return categories

def inspect_citizenship_duplicates_simple(year, sample_size=10):
    """
    Inspect duplicate columns in age/citizenship data to understand differences.
    Prints which columns have data and how they differ, but with less detail.
    """
    csv_path = PATH + 'PersonsByCitizenshipSexAge.csv'
    pattern = f'{PREFIX}{year} - Antal personer opgjort efter statsborgerskab køn og aldersgrupper_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = {
        '01.': 'Danmark',
        '02.': 'Nordiske lande',
        '03.': 'Tyrkiet',
        '04.': 'Tidligere Jugoslavien',
        '05.': 'Gamle EU-lande',
        '06.': 'Nye EU-lande',
        '07.': 'Øvrige Europa',
        '08.': 'Afrika',
        '09.': 'Nordamerika',
        '10.': 'Syd- og Mellemamerika',
        '11.': 'Asien og Oceanien',
        '12.': 'Uoplyst/statsløse'
    }

    print(f"\n{'='*80}")
    print(f"CITIZENSHIP DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    # Get all relevant columns
    citizenship_cols = []
    for col in df.columns:
        if pattern in col:
            for category in categories.keys():
                if f'_{category}' in col:
                    citizenship_cols.append(col)
                    break
    
    # Separate columns by space vs no-space
    with_space = []
    without_space = []
    
    for col in citizenship_cols:
        suffix = col.split('_')[-1]
        if '. ' in suffix:  # Has space after dot
            with_space.append(col)
        else:
            without_space.append(col)
    
    print(f"Total columns found: {len(citizenship_cols)}")
    print(f"  With space (e.g. '01. Danmark'): {len(with_space)}")
    print(f"  Without space (e.g. '01.Danmark'): {len(without_space)}\n")
    
    # Check if columns with space have any data
    with_space_has_data = 0
    with_space_total_sum = 0
    for col in with_space:
        vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if vals.notna().sum() > 0:
            with_space_has_data += 1
            with_space_total_sum += vals.sum()
    
    # Check if columns without space have data
    without_space_has_data = 0
    without_space_total_sum = 0
    for col in without_space:
        vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if vals.notna().sum() > 0:
            without_space_has_data += 1
            without_space_total_sum += vals.sum()
    
    print("SUMMARY:")
    print(f"  Columns WITH space:")
    print(f"    Columns with data: {with_space_has_data}/{len(with_space)}")
    print(f"    Total sum across all: {with_space_total_sum:.2f}")
    print(f"  Columns WITHOUT space:")
    print(f"    Columns with data: {without_space_has_data}/{len(without_space)}")
    print(f"    Total sum across all: {without_space_total_sum:.2f}")
    
    print(f"\nCONCLUSION:")
    if with_space_has_data == 0 and without_space_has_data > 0:
        print("  USE columns WITHOUT space (no space after dot)")
        print("  IGNORE columns WITH space (they are empty)")
    elif with_space_has_data > 0 and without_space_has_data == 0:
        print("  USE columns WITH space")
        print("  IGNORE columns WITHOUT space (they are empty)")
    else:
        print("  ⚠ Both types have data - need further investigation")

def inspect_education_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in education data to understand differences.
    Focuses on age group naming variations (e.g., '70- år' vs '70 år -').
    """
    csv_path = PATH + 'HighestCompletedEducationByAge.csv'
    pattern = f'{PREFIX}{year} - Højst fuldførte erhvervsuddannelse og aldersgrupper_'
    
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]
    
    age_groups = ['18-19 år', '20-24 år', '25-29 år', '30-34 år', '35-39 år', 
                  '40-44 år', '45-49 år', '50-54 år', '55-59 år', '60-64 år', 
                  '65-69 år', '70- år', '70 år -']
    
    education_levels = [
        '01.Grundskole',
        '02.Almengymnasial',
        '03.Erhvervsgymnaisal',
        '04.Erhvervsfaglig',
        '05.Kort videregående',
        '06.Mellemlang videregående',
        '07.Lang videregående',
        '08.Uoplyst'
    ]
    
    print(f"\n{'='*80}")
    print(f"EDUCATION DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")
    
    # Check each age group variant
    for age_variant in ['70- år', '70 år -']:
        cols_found = []
        for edu in education_levels:
            col_name = f'{pattern}{age_variant}_{edu}'
            if col_name in df.columns:
                cols_found.append(col_name)
        
        if cols_found:
            print(f"Age variant '{age_variant}': {len(cols_found)} columns found")
            
            # Check if they have data
            total_sum = 0
            cols_with_data = 0
            for col in cols_found:
                vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), 
                                   errors='coerce')
                if vals.notna().sum() > 0:
                    cols_with_data += 1
                    total_sum += vals.sum()
            
            print(f"  Columns with data: {cols_with_data}/{len(cols_found)}")
            print(f"  Total sum: {total_sum:.2f}")
        else:
            print(f"Age variant '{age_variant}': 0 columns found")
    
    print(f"\nCONCLUSION:")
    
    # Check which variant to use
    variant_70_minus = f'{pattern}70- år_{education_levels[0]}'
    variant_70_space = f'{pattern}70 år -_{education_levels[0]}'
    
    has_70_minus = variant_70_minus in df.columns
    has_70_space = variant_70_space in df.columns
    
    if has_70_minus and has_70_space:
        # Check which has data
        vals_minus = pd.to_numeric(df[variant_70_minus].astype(str).str.replace(',', '.'), 
                                  errors='coerce')
        vals_space = pd.to_numeric(df[variant_70_space].astype(str).str.replace(',', '.'), 
                                  errors='coerce')
        
        if vals_minus.notna().sum() > 0 and vals_space.notna().sum() == 0:
            print(f"  ✓ Use '70- år' format (has data)")
            print(f"  ✗ Ignore '70 år -' format (empty)")
        elif vals_space.notna().sum() > 0 and vals_minus.notna().sum() == 0:
            print(f"  ✓ Use '70 år -' format (has data)")
            print(f"  ✗ Ignore '70- år' format (empty)")
        else:
            print(f"  ⚠ Both formats exist - need to check data")
    elif has_70_minus:
        print(f"  ✓ Use '70- år' format (only one available)")
    elif has_70_space:
        print(f"  ✓ Use '70 år -' format (only one available)")
    else:
        print(f"  ✗ Neither format found")

def inspect_immigrants_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in immigrant/descendant data to understand differences.
    Prints which columns have data and how they differ.
    """
    csv_path = PATH + 'ImmigrantsDescendantsByCountryOfOrigin.csv'
    pattern = f'{PREFIX}{year} - Indvandrere og efterkommere fordelt efter oprindelsesland_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = {
        '01.': 'Danmark',
        '02.': 'Nordiske lande',
        '03.': 'Tyrkiet',
        '04.': 'Tidligere Jugoslavien',
        '05.': 'Gamle EU-lande',
        '06.': 'Nye EU-lande',
        '07.': 'Øvrige Europa',
        '08.': 'Afrika',
        '09.': 'Nordamerika',
        '10.': 'Syd- og Mellemamerika',
        '11.': 'Asien og Oceanien',
        '12.': 'Uoplyst/statsløse'
    }

    print(f"\n{'='*80}")
    print(f"IMMIGRANTS/DESCENDANTS DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    # Get all relevant columns
    immigrants_cols = []
    for col in df.columns:
        if pattern in col:
            for category in categories.keys():
                if f'_{category}' in col:
                    immigrants_cols.append(col)
                    break
    
    # Separate columns by space vs no-space
    with_space = []
    without_space = []
    
    for col in immigrants_cols:
        suffix = col.split('_')[-1]
        if '. ' in suffix:  # Has space after dot
            with_space.append(col)
        else:
            without_space.append(col)
    
    print(f"Total columns found: {len(immigrants_cols)}")
    print(f"  With space (e.g. '01. Danmark'): {len(with_space)}")
    print(f"  Without space (e.g. '01.Danmark'): {len(without_space)}\n")
    
    # Check if columns with space have any data
    with_space_has_data = 0
    with_space_total_sum = 0
    for col in with_space:
        vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if vals.notna().sum() > 0:
            with_space_has_data += 1
            with_space_total_sum += vals.sum()
    
    # Check if columns without space have data
    without_space_has_data = 0
    without_space_total_sum = 0
    for col in without_space:
        vals = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        if vals.notna().sum() > 0:
            without_space_has_data += 1
            without_space_total_sum += vals.sum()
    
    print("SUMMARY:")
    print(f"  Columns WITH space:")
    print(f"    Columns with data: {with_space_has_data}/{len(with_space)}")
    print(f"    Total sum across all: {with_space_total_sum:.2f}")
    print(f"  Columns WITHOUT space:")
    print(f"    Columns with data: {without_space_has_data}/{len(without_space)}")
    print(f"    Total sum across all: {without_space_total_sum:.2f}")
    
    print(f"\nCONCLUSION:")
    if with_space_has_data == 0 and without_space_has_data > 0:
        print("  USE columns WITHOUT space (no space after dot)")
        print("  IGNORE columns WITH space (they are empty)")
    elif with_space_has_data > 0 and without_space_has_data == 0:
        print("  USE columns WITH space")
        print("  IGNORE columns WITHOUT space (they are empty)")
    else:
        print("  ⚠ Both types have data - need further investigation")

def inspect_housingtenure_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in housing tenure data to understand differences.
    Prints which columns have data and how they differ.
    """
    csv_path = PATH + 'HousingTenure.csv'
    pattern = f'{PREFIX}{year} - Ejerforhold_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    data_categories = [
        ('Antal boliger', 'Antal_boliger'),
        ('Antal personer', 'Antal_personer')
    ]
    tenure_categories = [
        '01. Ejerbolig',
        '02. Andelsbolig',
        '03. Almennyttig bolig',
        '04. Lejer i offentligt ejet',
        '05. Privat lejerbolig',
        '06. Øvrige beboede boliger',
        '07. Ingen BBR-oplysninger'
    ]

    print(f"\n{'='*80}")
    print(f"HOUSING TENURE DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    # Structure is '{pattern}_{data_category}_{tenure_category}'

    # Get all relevant columns
    relevant_cols = []
    for col in df.columns:
        if not col.startswith(pattern):
            continue
        for spaced, underscored in data_categories:
            # check: 'Antal boliger' or 'Antal_boliger'
            if spaced in col or underscored in col:
                for tenure in tenure_categories:
                    if col.endswith(tenure):
                        relevant_cols.append(col)
                        break
    relevant_cols = sorted(set(relevant_cols))
    print(f"Total columns found: {len(relevant_cols)}\n")

    spaced = []       # e.g. 'Antal boliger'
    underscored = []  # e.g. 'Antal_boliger'
    for col in relevant_cols:
        mid = col[len(pattern):].split('0')[0]  # extract the 'Antal...' part
        if ' ' in mid:  # spaced
            spaced.append(col)
        else:
            underscored.append(col)

    print(f"  Spaced columns     (e.g. 'Antal boliger'):   {len(spaced)}")
    print(f"  Underscore columns (e.g. 'Antal_boliger'):  {len(underscored)}\n")

    def check(cols):
        non_empty = 0
        total_sum = 0

        for col in cols:
            vals = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )
            if vals.notna().sum() > 0:
                non_empty += 1
                total_sum += vals.sum()

        return non_empty, total_sum

    spaced_count, spaced_sum = check(spaced)
    under_count, under_sum = check(underscored)

    print("SUMMARY:")
    print(f"  Columns using SPACE ('Antal boliger' or 'Antal personer'):")
    print(f"    Columns with data: {spaced_count}/{len(spaced)}")
    print(f"    Total sum across all: {spaced_sum:.2f}")
    print(f"  Columns using UNDERSCORE ('Antal_boliger' or 'Antal_personer'):")
    print(f"    Columns with data: {under_count}/{len(underscored)}")
    print(f"    Total sum across all: {under_sum:.2f}\n")

    print("CONCLUSION:")
    if spaced_count == 0 and under_count > 0:
        print("  USE columns with UNDERSCORES ('Antal_boliger', 'Antal_personer').")
        print("  IGNORE columns with spaces (they appear empty).")
    elif spaced_count > 0 and under_count == 0:
        print("  USE columns with SPACES ('Antal boliger', 'Antal personer').")
        print("  IGNORE columns with underscores (they appear empty).")
    elif spaced_count == 0 and under_count == 0:
        print("  ⚠ Neither type contains data — unexpected, investigate.")
    else:
        print("  ⚠ Both have data — check dataset manually.")

    print("\n" + "="*80 + "\n")

def inspect_housingtype_duplicates(year, sample_size=10):
    """
    Inspect duplicate columns in housing type data to understand differences.
    Prints which columns have data and how they differ.
    """
    csv_path = PATH + 'HousingType.csv'
    pattern = f'{PREFIX}{year} - Boligtype_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = [
        ('Antal boliger', 'Antal_boliger'),
        ('Antal personer', 'Antal_personer')
    ]
    types = [
        '01. Stuehuse og parcelhuse',
        '02. Række- og kædehuse',
        '03. Flerfamiliehuse',
        '04. Kollegier',
        '05. Sommerhuse',
        '06. Øvrige boliger',
        '07. Ingen BBR-oplysninger'
    ]

    print(f"\n{'='*80}")
    print(f"HOUSING TYPE DUPLICATE INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    # Structure is '{pattern}{data_category}_{type_category}'

    # Get all relevant columns
    relevant_cols = []
    for col in df.columns:
        if not col.startswith(pattern):
            continue
        for spaced, underscored in categories:
            # check: 'Antal boliger' or 'Antal_boliger'
            if spaced in col or underscored in col:
                for type in types:
                    if col.endswith(type):
                        relevant_cols.append(col)
                        break
    relevant_cols = sorted(set(relevant_cols))
    print(f"Total columns found: {len(relevant_cols)}\n")

    spaced = []       # e.g. 'Antal boliger'
    underscored = []  # e.g. 'Antal_boliger'
    for col in relevant_cols:
        mid = col[len(pattern):].split('0')[0]  # extract the 'Antal...' part
        if ' ' in mid:  # spaced
            spaced.append(col)
        else:
            underscored.append(col)

    print(f"  Spaced columns     (e.g. 'Antal boliger'):   {len(spaced)}")
    print(f"  Underscore columns (e.g. 'Antal_boliger'):  {len(underscored)}\n")

    def check(cols):
        non_empty = 0
        total_sum = 0

        for col in cols:
            vals = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            )
            if vals.notna().sum() > 0:
                non_empty += 1
                total_sum += vals.sum()

        return non_empty, total_sum

    spaced_count, spaced_sum = check(spaced)
    under_count, under_sum = check(underscored)

    print("SUMMARY:")
    print(f"  Columns using SPACE ('Antal boliger' or 'Antal personer'):")
    print(f"    Columns with data: {spaced_count}/{len(spaced)}")
    print(f"    Total sum across all: {spaced_sum:.2f}")
    print(f"  Columns using UNDERSCORE ('Antal_boliger' or 'Antal_personer'):")
    print(f"    Columns with data: {under_count}/{len(underscored)}")
    print(f"    Total sum across all: {under_sum:.2f}\n")

    print("CONCLUSION:")
    if spaced_count == 0 and under_count > 0:
        print("  USE columns with UNDERSCORES ('Antal_boliger', 'Antal_personer').")
        print("  IGNORE columns with spaces (they appear empty).")
    elif spaced_count > 0 and under_count == 0:
        print("  USE columns with SPACES ('Antal boliger', 'Antal personer').")
        print("  IGNORE columns with underscores (they appear empty).")
    elif spaced_count == 0 and under_count == 0:
        print("  ⚠ Neither type contains data — unexpected, investigate.")
    else:
        print("  ⚠ Both have data — check dataset manually.")

    print("\n" + "="*80 + "\n")

def inspect_constructionyear_duplicates(year, sample_size=10):
    """
    Inspect all construction year columns to see which have data.
    Shows exactly which columns exist and contain data.
    """
    csv_path = PATH + 'HousingByYearOfConstruction.csv'
    pattern = f'{PREFIX}{year} - Opførelsesår_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    print(f"\n{'='*80}")
    print(f"CONSTRUCTION YEAR INSPECTION - Year {year}")
    print(f"{'='*80}\n")

    # Get ALL columns that match the pattern
    relevant_cols = [col for col in df.columns if col.startswith(pattern)]
    print(f"Total columns found: {len(relevant_cols)}\n") # should get 30 cols for 2021

    # Analyse each column
    results = []
    for col in sorted(relevant_cols):
        suffix = col.replace(pattern, '')
        vals = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False).replace('-', ''),
            errors='coerce'
        )
        non_null = vals.notna().sum()
        non_zero = (vals != 0).sum()
        total = vals.sum()
        has_data = non_null > 0 and total > 0
        results.append({
            'column': suffix,
            'has_data': has_data,
            'non_null': non_null,
            'non_zero': non_zero,
            'sum': total
        })

    # Separate (spaced or underscored)
    spaced = [r for r in results if ' ' in r['column']]
    underscored = [r for r in results if '_' in r['column'] and ' ' not in r['column']]

    print(f"Columns with SPACE (e.g., 'Antal boliger_...'):")
    print(f"  Total: {len(spaced)}")
    print(f"  With data: {sum(1 for r in spaced if r['has_data'])}")
    for r in spaced:
        status = "✓" if r['has_data'] else "✗"
        print(f"    {status} {r['column'][:60]:60s} sum={r['sum']:.0f}")
    
    print(f"\nColumns with UNDERSCORE (e.g., 'Antal_boliger_...'):")
    print(f"  Total: {len(underscored)}")
    print(f"  With data: {sum(1 for r in underscored if r['has_data'])}")
    for r in underscored:
        status = "✓" if r['has_data'] else "✗"
        print(f"    {status} {r['column'][:60]:60s} sum={r['sum']:.0f}")
    
    print(f"\nCONCLUSION:")
    spaced_with_data = sum(1 for r in spaced if r['has_data'])
    under_with_data = sum(1 for r in underscored if r['has_data'])
    
    if under_with_data > spaced_with_data:
        print(f"  ✓ USE underscore columns (Antal_boliger/Antal_personer)")
        print(f"    {under_with_data} columns have data vs {spaced_with_data} spaced columns")
    elif spaced_with_data > under_with_data:
        print(f"  ✓ USE spaced columns (Antal boliger/Antal personer)")
        print(f"    {spaced_with_data} columns have data vs {under_with_data} underscore columns")
    else:
        print(f"  ⚠ Both types have similar data - investigate manually")
    
    print("\n" + "="*80 + "\n")


def preprocess_socioeconomy(year):
    """
    Extract socioeconomy data: totals for Topledere, Selvstændig, and Arbejdsløse.
    Only uses unnumbered columns (without "01. ", "02. " etc. prefix) to avoid duplicates.
    """
    csv_path = PATH + 'PersonsBySocioeconomyIndustry.csv'
    pattern = f'{PREFIX}{year} - Socio-økonomisk status og brancher fordelt på afstemningsområder_'
    
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]
    
    # Define the three categories we want
    categories = {
        'Selvstaendige': '01. Selvstændig og medhj.',
        'Topledere': '02. Topledere',
        'Arbejdsloese': '07. Arbejdsløse'
    }

    translations = {
        'Selvstaendige': 'Self-employed',
        'Topledere': 'Top executives',
        'Arbejdsloese': 'Unemployed'
    }

    # Start df with common columns
    result_df = df[[col for col in COMMON_COLS if col in df.columns]].copy()

    print(year)
    for clean_name, category_code in categories.items():
        # Find all columns for this category
        category_cols = [col for col in df.columns 
                        if f'{pattern}{category_code}_' in col]
        
        if not category_cols:
            print(f"    Warning: No columns found for {category_code}")
            result_df[f'Socioeconomy_{translations[clean_name]}'] = np.nan
            continue

        # Only get unnumbered columns
        unnumbered_cols = []
        for col in category_cols:
            suffix = col.split(f'{pattern}{category_code}_')[1]
            # Check if suffix starts with number pattern like "01. "
            if not re.match(r'^\d+\.', suffix):
                unnumbered_cols.append(col)
        
        if not unnumbered_cols:
            print(f"    Warning: No unnumbered columns found for {category_code}")
            result_df[f'Socioeconomy_{translations[clean_name]}'] = np.nan
            continue
        
        print(f"    {category_code}: Using {len(unnumbered_cols)} unnumbered columns")

        # Process each column; convert to numeric, treat '-' and empty as NaN
        processed_cols = []
        for col in unnumbered_cols:
            # Replace '-' with NaN, then convert to numeric
            col_numeric = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace('-', '', regex=False)
                .str.replace(',', '.', regex=False)
                .pipe(pd.to_numeric, errors='coerce')
            )

            if col_numeric.notna().sum() == 0:
                # all vals are null
                processed_cols.append(pd.Series(np.nan, index=df.index))
            else:
                # column has SOME data, so fill NaN with 0
                processed_cols.append(col_numeric.fillna(0))

        if processed_cols:
            combined = pd.concat(processed_cols, axis=1)

            # Sum row-wise
            category_total = combined.sum(axis=1)

            # If ALL original values in a row were NaN, keep NaN instead of 0
            all_nan_rows = combined.isna().all(axis=1)
            category_total.loc[all_nan_rows] = np.nan

            result_df[f'Socioeconomy_{clean_name}'] = category_total
        else:
            result_df[f'Socioeconomy_{clean_name}'] = np.nan
    result_df['Socioeconomy_Total'] = result_df.filter(like='Socioeconomy_').sum(axis=1)
    return result_df

def preprocess_socioeconomy2(year):
    """
    Extract socioeconomy data: totals for Topledere, Selvstændig, and Arbejdsløse.
    And total across columns.
    Only uses unnumbered columns (without "01. ", "02. " etc. prefix) to avoid duplicates.
    """
    csv_path = PATH + 'PersonsBySocioeconomyIndustry.csv'
    pattern = f'{PREFIX}{year} - Socio-økonomisk status og brancher fordelt på afstemningsområder_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    all_categories = [
        '01. Selvstændig og medhj.',
        '02. Topledere',
        '03. Lønmodtagere på højt niveau',
        '04. Lønmodtagere på mellem niveau',
        '05. Lønmodtagere på grundniveau', 
        '06. Øvrige lønmodtagere',
        '07. Arbejdsløse',
        '08. Uddannelsessøgende',
        '09. Folke- og tjenestemandspension',
        '10. Efterløn',
        '11. Børn',
        '12. Øvrige (offentligt forsørgede)'
    ]

    wanted = {
        '01. Selvstændig og medhj.': 'Self-employed',
        '02. Topledere': 'Top executives',
        '07. Arbejdsløse': 'Unemployed'
    }

    # Helper functions
    def clean_column(col_name):
        if col_name not in df.columns:
            return None
        col_data = df[col_name].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.', regex=False)
        col_data = col_data.replace('-', '', regex=False)
        return pd.to_numeric(col_data, errors='coerce')
    
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)

    # Start df with common columns
    result_df = df[[col for col in COMMON_COLS if col in df.columns]].copy()

    all_processed_cols = []
    print(year)
    for category in all_categories:
        regex = re.compile(rf"^{re.escape(pattern)}{re.escape(category)}_[^\d].*")
        sub_cols = [c for c in df.columns if regex.match(c)]
        if not sub_cols:
            print(f"    Warning: No unnumbered columns found for {category}")
            continue

        processed = []
        for col in sub_cols:
            series = (
                df[col].astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            series = series.replace('-', '', regex=False)
            numeric = pd.to_numeric(series, errors="coerce")
            if (numeric.fillna(0) == 0).all():
                processed.append(pd.Series(np.nan, index=df.index))
            else:
                processed.append(numeric.fillna(0))
        if processed:
            combined = pd.concat(processed, axis=1)
            all_processed_cols.append(combined.sum(axis=1))
            if category in wanted:
                label = "Socioeconomy_" + wanted[category]
                result_df[label] = combined.sum(axis=1)
        else:
            if category in wanted:
                result_df["Socioeconomy_" + wanted[category]] = np.nan

    if all_processed_cols:
        total = pd.concat(all_processed_cols, axis=1).sum(axis=1)
        result_df["Socioeconomy_Total"] = total
    else:
        result_df["Socioeconomy_Total"] = np.nan
    return result_df

def preprocess_citizenship_sex_age(year):
    """
    Extract citizenship and age data per year.
    Creates three sets of columns:
    1. Sex (total male, total female)
    2. Age groups (combined male + female)
    3. Citizenship totals (all ages, both sexes)
    """
    csv_path = PATH + 'PersonsByCitizenshipSexAge.csv'
    pattern = f'{PREFIX}{year} - Antal personer opgjort efter statsborgerskab køn og aldersgrupper_'
    
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]
    use_no_space = year in [2001, 2005]

    # column structure is '{pattern}{sex} {age_group}_{category number} {citizenship}' (with space) OR '{pattern}{sex} {age_group}_{category number}{citizenship}' (no space)
    sexes = [('Mænd', 'Male'), ('Kvinder', 'Female')]
    age_groups = ['0-4 år', '5-9 år', '10-14 år', '15-17 år', '18-19 år', 
                  '20-24 år', '25-29 år', '30-34 år', '35-39 år', '40-44 år',
                  '45-49 år', '50-54 år', '55-59 år', '60-64 år', '65-69 år', '70- år']
    citizenships = [
        ('01. Danmark', '01.Danmark'),
        ('02. Nordiske lande', '02.Nordiske lande'),
        ('03. Tyrkiet', '03.Tyrkiet'),
        ('04. Tidligere Jugoslavien', '04.Tidligere Jugoslavien'),
        ('05. Gamle EU-lande', '05.Øvrige gamle EU-lande'),
        ('06. Nye EU-lande', '06.Nye EU-lande'),
        ('07. Øvrige Europa', '07.Øvrige Europa'),
        ('08. Afrika', '08.Afrika'),
        ('09. Nordamerika', '09.Nordamerika'),
        ('10. Syd- og Mellemamerika', '10.Syd-og Mellemam.'),
        ('11. Asien og Oceanien', '11.Asien og oceanien'),
        ('12. Uoplyst', '12.Uoplyst/statsløse')
    ]
    translations = {
        'Danmark': 'Denmark',
        'Nordiske lande': 'Nordic',
        'Tyrkiet': 'Turkey',
        'Tidligere Jugoslavien': 'Former Yugoslavia',
        'Gamle EU-lande': 'Old EU countries',
        'Nye EU-lande': 'New EU countries',
        'Øvrige Europa': 'Other Europe',
        'Afrika': 'Africa',
        'Nordamerika': 'North America',
        'Syd- og Mellemamerika': 'South and Central America',
        'Asien og Oceanien': 'Asia and Oceania',
        'Uoplyst': 'Not specified'
    }

    # helper function to clean numeric columns, treat '-' as NaN
    def clean_column(col_name):
        if col_name in df.columns:
            target_col = col_name
        else:
            # Try to find with stripped whitespace
            matching = [c for c in df.columns if c.strip() == col_name.strip()]
            if not matching:
                return None
            target_col = matching[0]
        col_data = df[target_col].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')  # Replace '-' with empty string instead of NaN
        return pd.to_numeric(col_data, errors='coerce')  # Empty strings become NaN automatically
    
    # helper function to deal with NaN/0 - if all vals in a col are null(0), return NaN series, otherwise replace NaN with 0
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    result_df = df[COMMON_COLS].copy()
    
    # Get total male and total female
    for sex in sexes:
        sex_total = pd.Series(0.0, index=df.index)
        found_data = False
        
        for age in age_groups:
            for citizenship_variants in citizenships:
                variant_to_use = citizenship_variants[1] if use_no_space else citizenship_variants[0]
                col_name = f'{pattern}{sex[0]} {age}_{variant_to_use}'
                vals = clean_column(col_name)
                if vals is not None:
                    sex_total += vals.fillna(0)
                    found_data = True

        if found_data:
            result_df[f'Sex_{sex[1]}'] = finalise_column(sex_total)
        else:
            result_df[f'Sex_{sex[1]}'] = np.nan

    # Get total within each age group
    for age in age_groups:
        age_total = pd.Series(0.0, index=df.index)
        found_data = False
        
        for sex in sexes:
            for citizenship_variants in citizenships:
                variant_to_use = citizenship_variants[1] if use_no_space else citizenship_variants[0]
                col_name = f'{pattern}{sex[0]} {age}_{variant_to_use}'
                vals = clean_column(col_name)
                if vals is not None:
                    age_total += vals.fillna(0)
                    found_data = True
        
        if found_data:
            clean_age_name = age.replace(' år', ' years')
            result_df[f'Age_{clean_age_name}'] = finalise_column(age_total)
        else:
            result_df[f'Age_{age.replace(" ", "_")}'] = np.nan

    # Get citizenships
    for citizenship_variants in citizenships:
        citizenship_sum = pd.Series(0.0, index=df.index)
        citizenship_found = False
        
        # Select correct variant based on year
        variant_to_use = citizenship_variants[1] if use_no_space else citizenship_variants[0]
        
        for age in age_groups:
            for sex in sexes:
                col = f'{pattern}{sex[0]} {age}_{variant_to_use}'
                vals = clean_column(col)
                if vals is not None:
                    citizenship_sum += vals.fillna(0)
                    citizenship_found = True
        
        if citizenship_found:
            citizenship_name = citizenship_variants[0].split('. ')[1]
            result_df[f'Citizenship_{translations[citizenship_name]}'] = finalise_column(citizenship_sum)
        else:
            citizenship_name = citizenship_variants[0].split('. ')[1]
            result_df[f'Citizenship_{translations[citizenship_name]}'] = np.nan
    
    print(f"    Citizenship data: Using {'no-space' if use_no_space else 'with-space'} format for year {year}")
    result_df['Citizenship_Total'] = result_df.filter(like='Citizenship_').sum(axis=1)
    return result_df

def preprocess_education(year):
    """
    Extract education data, grouping into Under 30 and 30+ age groups.
    Returns totals for each education level within each age group.
    """
    csv_path = PATH + 'HighestCompletedEducationByAge.csv'
    pattern = f'{PREFIX}{year} - Højst fuldførte erhvervsuddannelse og aldersgrupper_'
    
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]
    
    under_30 = ['18-19 år', '20-24 år', '25-29 år']
    over_30 = ['30-34 år', '35-39 år', '40-44 år', '45-49 år', '50-54 år', 
               '55-59 år', '60-64 år', '65-69 år']
    age_70_format = ''
    if year in [2001, 2005, 2009]:
        age_70_format = '70 år -'
    else:
        age_70_format = '70- år'
    over_30.append(age_70_format)
    
    education_levels = {
        '01.Grundskole': 'Primary and lower secondary',
        '02.Almengymnasial': 'General upper secondary',
        '03.Erhvervsgymnaisal': 'Vocational upper secondary',
        '04.Erhvervsfaglig': 'Vocational training',
        '05.Kort videregående': 'Short higher education',
        '06.Mellemlang videregående': 'Medium higher education',
        '07.Lang videregående': 'Long higher education',
        '08.Uoplyst': 'Not specified'
    }
    
    result_df = df[COMMON_COLS].copy()
    
    # Helper function
    def clean_column(col_name):
        if col_name not in df.columns:
            return None
        col_data = df[col_name].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')
        return pd.to_numeric(col_data, errors='coerce')
    
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    # Process each education level for each age group
    for edu_code, edu_name in education_levels.items():
        # Under 30
        under_30_total = pd.Series(0.0, index=df.index)
        found_data = False
        
        for age in under_30:
            col_name = f'{pattern}{age}_{edu_code}'
            vals = clean_column(col_name)
            if vals is not None:
                under_30_total += vals.fillna(0)
                found_data = True
        
        if found_data:
            result_df[f'Education_18-29 years_{edu_name}'] = finalise_column(under_30_total)
        else:
            result_df[f'Education_18-29 years_{edu_name}'] = np.nan
        
        # 30 and over
        over_30_total = pd.Series(0.0, index=df.index)
        found_data = False
        
        for age in over_30:
            col_name = f'{pattern}{age}_{edu_code}'
            vals = clean_column(col_name)
            if vals is not None:
                over_30_total += vals.fillna(0)
                found_data = True
        
        if found_data:
            result_df[f'Education_30-70+ years_{edu_name}'] = finalise_column(over_30_total)
        else:
            result_df[f'Education_30-70+ years_{edu_name}'] = np.nan
    
    print(f"    Education data: Using '{age_70_format}' for 70+ age group")
    result_df['Education_Total 18-29 years'] = result_df.filter(like='Education_18-29 years_').sum(axis=1)
    result_df['Education_Total 30-70+ years'] = result_df.filter(like='Education_30-70+ years_').sum(axis=1)
    result_df['Education_Total'] = result_df['Education_Total 18-29 years'] + result_df['Education_Total 30-70+ years']

    return result_df

def preprocess_immigrants(year):
    """
    Extract immigrants/descendants data per year.
    Creates columns for totals for each place.
    """    
    csv_path = PATH + 'ImmigrantsDescendantsByCountryOfOrigin.csv'
    pattern = f'{PREFIX}{year} - Indvandrere og efterkommere fordelt efter oprindelsesland_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]
    use_no_space = year in [2001, 2005]

    # column structure is '{pattern}{sex} {age_group}_{category number} {citizenship}' (with space) OR '{pattern}{sex} {age_group}_{category number}{citizenship}' (no space)
    sexes = ['Mænd', 'Kvinder']
    age_groups = ['0-4 år', '5-9 år', '10-14 år', '15-17 år', '18-19 år', 
                  '20-24 år', '25-29 år', '30-34 år', '35-39 år', '40-44 år',
                  '45-49 år', '50-54 år', '55-59 år', '60-64 år', '65-69 år', '70- år']
    country_of_origin = [
        ('01. Danmark', '01.Danmark'),
        ('02. Nordiske lande', '02.Nordiske lande'),
        ('03. Tyrkiet', '03.Tyrkiet'),
        ('04. Tidligere Jugoslavien', '04.Tidligere Jugoslavien'),
        ('05. Gamle EU-lande', '05.Øvrige gamle EU-lande'),
        ('06. Nye EU-lande', '06.Nye EU-lande'),
        ('07. Øvrige Europa', '07.Øvrige Europa'),
        ('08. Afrika', '08.Afrika'),
        ('09. Nordamerika', '09.Nordamerika'),
        ('10. Syd- og Mellemamerika', '10.Syd-og Mellemam.'),
        ('11. Asien og Oceanien', '11.Asien og oceanien'),
        ('12. Uoplyst', '12.Uoplyst/statsløse')
    ]
    translations = {
        'Danmark': 'Denmark',
        'Nordiske lande': 'Nordic',
        'Tyrkiet': 'Turkey',
        'Tidligere Jugoslavien': 'Former Yugoslavia',
        'Gamle EU-lande': 'Old EU countries',
        'Nye EU-lande': 'New EU countries',
        'Øvrige Europa': 'Other Europe',
        'Afrika': 'Africa',
        'Nordamerika': 'North America',
        'Syd- og Mellemamerika': 'South and Central America',
        'Asien og Oceanien': 'Asia and Oceania',
        'Uoplyst': 'Not specified'
    }

    # helper function to clean numeric columns, treat '-' as NaN
    def clean_column(col_name):
        if col_name in df.columns:
            target_col = col_name
        else:
            # Try to find with stripped whitespace
            matching = [c for c in df.columns if c.strip() == col_name.strip()]
            if not matching:
                return None
            target_col = matching[0]
        col_data = df[target_col].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')  # Replace '-' with empty string instead of NaN
        return pd.to_numeric(col_data, errors='coerce')  # Empty strings become NaN automatically
    
    # helper function to deal with NaN/0 - if all vals in a col are null(0), return NaN series, otherwise replace NaN with 0
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    result_df = df[COMMON_COLS].copy()

    # Get totals for each country of origin
    for country in country_of_origin:
        country_sum = pd.Series(0.0, index=df.index)
        country_found = False
        variant_to_use = country[1] if use_no_space else country[0]
        
        for age in age_groups:
            for sex in sexes:
                col = f'{pattern}{sex} {age}_{variant_to_use}'
                vals = clean_column(col)
                if vals is not None:
                    country_sum += vals.fillna(0)
                    country_found = True
        
        if country_found:
            country_name = country[0].split('. ')[1]
            result_df[f'Immigrants_{translations[country_name]}'] = finalise_column(country_sum)
        else:
            country_name = country[0].split('. ')[1]
            result_df[f'Immigrants_{translations[country_name]}'] = np.nan
    
    print(f"    Immigrants/descendants data: Using {'no-space' if use_no_space else 'with-space'} format for year {year}")
    result_df['Immigrants_Total'] = result_df.filter(like='Immigrants_').sum(axis=1)
    return result_df

def preprocess_housingtenure(year):
    """
    Extract housing tenure data per year.

    Only get ejerbolig, andelsbolig, almenbolig and privat lejerbolig.
    """    
    csv_path = PATH + 'HousingTenure.csv'
    pattern = f'{PREFIX}{year} - Ejerforhold_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    data_categories = [
        ('Antal boliger', 'Antal_boliger', 'Homes'),
        ('Antal personer', 'Antal_personer', 'People')
    ]
    tenure_categories = {
        '01. Ejerbolig': 'Ejer', # owner-occupied housing
        '02. Andelsbolig': 'Andel', # cooperative housing
        '03. Almennyttig bolig': 'Almen', # social housing
        '05. Privat lejerbolig': 'Private rental',
    }

    # helper function to clean numeric columns, treat '-' as NaN
    def clean_column(col_name):
        if col_name in df.columns:
            target_col = col_name
        else:
            # Try to find with stripped whitespace
            matching = [c for c in df.columns if c.strip() == col_name.strip()]
            if not matching:
                return None
            target_col = matching[0]
        col_data = df[target_col].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')  # Replace '-' with empty string instead of NaN
        return pd.to_numeric(col_data, errors='coerce')  # Empty strings become NaN automatically
    
    # helper function to deal with NaN/0 - if all vals in a col are null(0), return NaN series, otherwise replace NaN with 0
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    result_df = df[COMMON_COLS].copy()
    use_underscore = year in [2013, 2017, 2021]

    # Get totals for each country of origin
    for spaced, underscored, var_name in data_categories:
        variant = underscored if use_underscore else spaced
        for tenure, english in tenure_categories.items():
            col = f'{pattern}{variant}_{tenure}'
            vals = clean_column(col)
            if vals is None:
                result_df[f"Housing tenure_{var_name}_{english}"] = np.nan
                continue
            result_df[f"Housing tenure_{var_name}_{english}"] = finalise_column(vals)
    print(f"    Housing tenure data: Using {'underscored' if use_underscore else 'spaced'} format for year {year}")

    people_cols = [c for c in result_df.columns if c.startswith("Housing tenure_People_")]
    homes_cols  = [c for c in result_df.columns if c.startswith("Housing tenure_Homes_")]
    result_df["Housing tenure_Total people"] = result_df[people_cols].sum(axis=1, min_count=1)
    result_df["Housing tenure_Total homes"]  = result_df[homes_cols].sum(axis=1, min_count=1)

    return result_df

def preprocess_housingtype(year):
    """
    Extract housing type data per year.
    """    
    csv_path = PATH + 'HousingType.csv'
    pattern = f'{PREFIX}{year} - Boligtype_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = [
        ('Antal boliger', 'Antal_boliger', 'Homes'),
        ('Antal personer', 'Antal_personer', 'People')
    ]

    types = {
        '01. Stuehuse og parcelhuse': 'Detached house',
        '02. Række- og kædehuse': 'Terraced house',
        '03. Flerfamiliehuse': 'Apartment buildings',
        '04. Kollegier': 'Student halls',
        '05. Sommerhuse': 'Holiday homes',
        '06. Øvrige boliger': 'Other homes',
        '07. Ingen BBR-oplysninger': 'No BBR information'
    }

    # helper function to clean numeric columns, treat '-' as NaN
    def clean_column(col_name):
        if col_name in df.columns:
            target_col = col_name
        else:
            # Try to find with stripped whitespace
            matching = [c for c in df.columns if c.strip() == col_name.strip()]
            if not matching:
                return None
            target_col = matching[0]
        col_data = df[target_col].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')  # Replace '-' with empty string instead of NaN
        return pd.to_numeric(col_data, errors='coerce')  # Empty strings become NaN automatically
    
    # helper function to deal with NaN/0 - if all vals in a col are null(0), return NaN series, otherwise replace NaN with 0
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    result_df = df[COMMON_COLS].copy()
    use_underscore = year in [2013, 2017, 2021]

    # Get totals for each country of origin
    for spaced, underscored, english in categories:
        variant = underscored if use_underscore else spaced
        for type in types.keys():
            col = f'{pattern}{variant}_{type}'
            vals = clean_column(col)
            if vals is None:
                result_df[f"Housing type_{english}_{types[type]}"] = np.nan
                continue
            result_df[f"Housing type_{english}_{types[type]}"] = finalise_column(vals)
    print(f"    Housing type data: Using {'underscored' if use_underscore else 'spaced'} format for year {year}")

    people_cols = [c for c in result_df.columns if c.startswith("Housing type_People_")]
    homes_cols  = [c for c in result_df.columns if c.startswith("Housing type_Homes_")]
    result_df["Housing type_Total people"] = result_df[people_cols].sum(axis=1, min_count=1)
    result_df["Housing type_Total homes"]  = result_df[homes_cols].sum(axis=1, min_count=1)

    return result_df

def preprocess_builtyear(year):
    """
    Extract housing type data per year.
    """    
    csv_path = PATH + 'HousingByYearOfConstruction.csv'
    pattern = f'{PREFIX}{year} - Opførelsesår_'
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df[df['Gruppe'].isin(CPH_AREAS)]

    categories = [
        ('Antal boliger', 'Antal_boliger', 'Homes'),
        ('Antal personer', 'Antal_personer', 'People')
    ]

    periods = ['Før 1940', '1940-1959', '1960-1969', '1970-1979', 
               '1980-1989', '1990-1999', '2000-2009', '2010-9999', '2000-']
    
        # Helper functions
    def clean_column(col_name):
        if col_name not in df.columns:
            return None
        col_data = df[col_name].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')
        return pd.to_numeric(col_data, errors='coerce')
    
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    result_df = df[COMMON_COLS].copy()
    use_underscore = year in [2013, 2017, 2021]

    # Get totals for each country of origin
    for spaced, underscored, english in categories:
        variant = underscored if use_underscore else spaced
        for period in periods:
            col = f'{pattern}{variant}_{period}'
            vals = clean_column(col)
            if vals is None:
                result_df[f"Year built_{english}_{period}"] = np.nan
                continue
            result_df[f"Year built_{english}_{period}"] = finalise_column(vals)
    print(f"    Construction year data: Using {'underscored' if use_underscore else 'spaced'} format for year {year}")

    people_cols = [c for c in result_df.columns if c.startswith("Year built_People_")]
    homes_cols  = [c for c in result_df.columns if c.startswith("Year built_Homes_")]
    result_df["Year built_Total people"] = result_df[people_cols].sum(axis=1, min_count=1)
    result_df["Year built_Total homes"]  = result_df[homes_cols].sum(axis=1, min_count=1)
    return result_df
    
def preprocess_simple(year):
    """
    Extract data from simple CSVs (benefit types, crime, income, car ownership).
    Returns a single dataframe with all simple demographic data.
    """
    
    # Load first CSV to get the common columns and index
    first_csv = list(SIMPLE_CSVS.keys())[0]
    first_df = pd.read_csv(PATH + first_csv, sep=';', dtype=str)
    result_df = first_df[first_df['Gruppe'].isin(CPH_AREAS)][COMMON_COLS].copy()
    
    # Helper functions
    def clean_column(df, col_name):
        if col_name not in df.columns:
            return None
        col_data = df[col_name].astype(str).str.strip()
        col_data = col_data.str.replace(',', '.')
        col_data = col_data.replace('-', '')
        return pd.to_numeric(col_data, errors='coerce')
    
    def finalise_column(series):
        if (series.fillna(0) == 0).all():
            return pd.Series(np.nan, index=series.index)
        return series.fillna(0)
    
    # Process each simple CSV
    for csv_name, config in SIMPLE_CSVS.items():
        csv_path = PATH + csv_name
        print(f"    Extracting from {csv_name}...")
        df = pd.read_csv(csv_path, sep=';', dtype=str)
        df = df[df['Gruppe'].isin(CPH_AREAS)]
        pattern = f'{PREFIX}{year}{config["pattern"]}'
        prefix = config['prefix']
        
        for new_name, col_suffix in config['columns'].items():
            full_col_name = f'{pattern}{col_suffix}'
            vals = clean_column(df, full_col_name)
            
            if vals is not None:
                # Merge on Gruppe to align rows
                temp_df = pd.DataFrame({'Gruppe': df['Gruppe'], 'temp_col': vals})
                result_df = result_df.merge(temp_df, on='Gruppe', how='outer')
                result_df.rename(columns={'temp_col': f'{prefix}_{new_name}'}, inplace=True)
                result_df[f'{prefix}_{new_name}'] = finalise_column(result_df[f'{prefix}_{new_name}'])
            else:
                print(f"      Warning: Column '{full_col_name}' not found")
                result_df[f'{prefix}_{new_name}'] = np.nan
    
    return result_df