import pandas as pd

PROJECT_ROOT = ''

# Read the files
kbh = pd.read_csv(PROJECT_ROOT + 'raw-data/election-2025/Kommunalvalg_2025_København_25-11-2025 14.54.11.csv', sep=';', encoding='utf-8')
fberg = pd.read_csv(PROJECT_ROOT + 'raw-data/election-2025/Kommunalvalg_2025_Frederiksberg_25-11-2025 14.54.22.csv', sep=';', encoding='utf-8')
geografi = pd.read_csv(PROJECT_ROOT + 'raw-data/historical-elections/election-results/Geography.csv', sep=';', encoding='utf-8')
valgData = pd.read_csv(PROJECT_ROOT + 'raw-data/historical-elections/election-results/ElectionData.csv', sep=';', encoding='utf-8')

# Clean column names
geografi.columns = geografi.columns.str.strip().str.replace('"', '')
valgData.columns = valgData.columns.str.strip().str.replace('"', '')

# Create expanded ValgData by splitting semicolon-separated ValgstedId values
expanded_rows = []
for _, row in valgData.iterrows():
    valgsted_ids = str(row['ValgstedId']).split(';')
    kreds_nrs = str(row['KredsNr']).split(';')
    stor_kreds_nrs = str(row['StorKredsNr']).split(';')
    
    # Use the first value for KredsNr and StorKredsNr if multiple exist
    kreds_nr = int(kreds_nrs[0].strip()) if kreds_nrs and kreds_nrs[0].strip() else None
    stor_kreds_nr = int(stor_kreds_nrs[0].strip()) if stor_kreds_nrs and stor_kreds_nrs[0].strip() else None
    
    # Create a row for each ValgstedId
    for valgsted_id in valgsted_ids:
        valgsted_id = valgsted_id.strip()
        if valgsted_id:
            expanded_rows.append({
                'ValgstedId': int(valgsted_id),
                'Gruppe': int(row['Gruppe']),
                'KredsNr': kreds_nr,
                'StorKredsNr': stor_kreds_nr
            })

valgData_expanded = pd.DataFrame(expanded_rows)

print(f"Expanded ValgData to {len(valgData_expanded)} rows from {len(valgData)} original rows")
print(f"Sample expanded rows:\n{valgData_expanded.head(10)}\n")

# Do a RIGHT join to include ALL ValgstedIds from ElectionData, even if not in Geography
geografi_with_gruppe = geografi.merge(
    valgData_expanded,
    left_on='Valgsted Id',
    right_on='ValgstedId',
    how='right'  # Changed from 'left' to 'right' to keep all ValgstedIds
)

# For ValgstedIds not in Geography, use ValgstedId as the name
geografi_with_gruppe['Valgsted navn'] = geografi_with_gruppe['Valgsted navn'].fillna(
    geografi_with_gruppe['ValgstedId'].astype(str)
)

# Ensure Valgsted Id is populated
geografi_with_gruppe['Valgsted Id'] = geografi_with_gruppe['Valgsted Id'].fillna(
    geografi_with_gruppe['ValgstedId']
)

# Get KommuneNr from the Gruppe (first 3 digits)
geografi_with_gruppe['KommuneNr'] = geografi_with_gruppe['KommuneNr'].fillna(
    geografi_with_gruppe['Gruppe'].astype(str).str[:3].astype(int)
)

print(f"Total ValgstedIds in expanded data: {len(geografi_with_gruppe)}")
print(f"ValgstedIds from Geography: {geografi['Valgsted Id'].nunique()}")
print(f"ValgstedIds from ElectionData: {valgData_expanded['ValgstedId'].nunique()}")

# Function to extract valgsted name from afstemningsområde
def extract_valgsted_name(name):
    """Remove the first number prefix (e.g., '1. ' from '1. 1. Østerbro')"""
    parts = name.split('. ', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return name.strip()

# Create normalized columns for matching
kbh['valgsted_for_match'] = kbh['Afstemningsområde'].apply(extract_valgsted_name)
fberg['valgsted_for_match'] = fberg['Afstemningsområde'].apply(extract_valgsted_name)

# Merge with Geografi (which now has Gruppe, KredsNr, StorKredsNr)
kbh_merged = kbh.merge(
    geografi_with_gruppe[geografi_with_gruppe['KommuneNr'] == 101][
        ['Valgsted Id', 'Valgsted navn', 'Gruppe', 'KredsNr', 'StorKredsNr']
    ],
    left_on='valgsted_for_match',
    right_on='Valgsted navn',
    how='left'
)

fberg_merged = fberg.merge(
    geografi_with_gruppe[geografi_with_gruppe['KommuneNr'] == 147][
        ['Valgsted Id', 'Valgsted navn', 'Gruppe', 'KredsNr', 'StorKredsNr']
    ],
    left_on='valgsted_for_match',
    right_on='Valgsted navn',
    how='left'
)

print(f"\nKøbenhavn matched: {kbh_merged['Gruppe'].notna().sum()} of {len(kbh_merged)}")
print(f"Frederiksberg matched: {fberg_merged['Gruppe'].notna().sum()} of {len(fberg_merged)}")

# Drop helper columns and rename
kbh_merged = kbh_merged.drop(columns=['valgsted_for_match'])
fberg_merged = fberg_merged.drop(columns=['valgsted_for_match'])

kbh_merged = kbh_merged.rename(columns={
    'Valgsted Id': 'ValgstedID',
    'Valgsted navn': 'ValgstedNavn'
})

fberg_merged = fberg_merged.rename(columns={
    'Valgsted Id': 'ValgstedID',
    'Valgsted navn': 'ValgstedNavn'
})

# Manual mapping for new 2025 locations
new_locations_kbh = {
    '9. 2. Ørestad': {'Gruppe': 101060, 'ValgstedID': 101060, 'ValgstedNavn': '2. Ørestad', 'KredsNr': 2, 'StorKredsNr': 1},
    '33. 7. Rødkilde': {'Gruppe': 101061, 'ValgstedID': 101061, 'ValgstedNavn': '7. Rødkilde', 'KredsNr': 7, 'StorKredsNr': 1},
    '60. 1. Nordhavn': {'Gruppe': 101062, 'ValgstedID': 101062, 'ValgstedNavn': '1. Nordhavn', 'KredsNr': 1, 'StorKredsNr': 1},
    '61. 2. Kalvebod Fælled': {'Gruppe': 101063, 'ValgstedID': 101063, 'ValgstedNavn': '2. Kalvebod Fælled', 'KredsNr': 2, 'StorKredsNr': 1},
    '62. 2. Peder Lykke': {'Gruppe': 101064, 'ValgstedID': 101064, 'ValgstedNavn': '2. Peder Lykke', 'KredsNr': 2, 'StorKredsNr': 1},
    '63. 3. Midt': {'Gruppe': 101065, 'ValgstedID': 101065, 'ValgstedNavn': '3. Midt', 'KredsNr': 3, 'StorKredsNr': 1},
    '64. 3. Sølvgade': {'Gruppe': 101066, 'ValgstedID': 101066, 'ValgstedNavn': '3. Sølvgade', 'KredsNr': 3, 'StorKredsNr': 1},
    '65. 4. Nordøst': {'Gruppe': 101067, 'ValgstedID': 101067, 'ValgstedNavn': '4. Nordøst', 'KredsNr': 4, 'StorKredsNr': 1},
    '66. 9. Sluseholmen': {'Gruppe': 101068, 'ValgstedID': 101068, 'ValgstedNavn': '9. Sluseholmen', 'KredsNr': 9, 'StorKredsNr': 1},
}

new_locations_fberg = {
    '2. 10. Kreds, Grundtvigsvej': {'Gruppe': 147009, 'ValgstedID': 147009, 'ValgstedNavn': '10. Kreds, Grundtvigsvej', 'KredsNr': 10, 'StorKredsNr': 1},
    '6. 11. Kreds, Frederiksberghallerne': {'Gruppe': 147010, 'ValgstedID': 147010, 'ValgstedNavn': '11. Kreds, Frederiksberghallerne', 'KredsNr': 11, 'StorKredsNr': 1},
}

# Fill missing values for new locations
for afstemning, values in new_locations_kbh.items():
    mask = kbh_merged['Afstemningsområde'] == afstemning
    kbh_merged.loc[mask, 'Gruppe'] = values['Gruppe']
    kbh_merged.loc[mask, 'ValgstedID'] = values['ValgstedID']
    kbh_merged.loc[mask, 'ValgstedNavn'] = values['ValgstedNavn']
    kbh_merged.loc[mask, 'KredsNr'] = values['KredsNr']
    kbh_merged.loc[mask, 'StorKredsNr'] = values['StorKredsNr']

for afstemning, values in new_locations_fberg.items():
    mask = fberg_merged['Afstemningsområde'] == afstemning
    fberg_merged.loc[mask, 'Gruppe'] = values['Gruppe']
    fberg_merged.loc[mask, 'ValgstedID'] = values['ValgstedID']
    fberg_merged.loc[mask, 'ValgstedNavn'] = values['ValgstedNavn']
    fberg_merged.loc[mask, 'KredsNr'] = values['KredsNr']
    fberg_merged.loc[mask, 'StorKredsNr'] = values['StorKredsNr']

# Convert to integers
kbh_merged['Gruppe'] = kbh_merged['Gruppe'].astype(int)
kbh_merged['ValgstedID'] = kbh_merged['ValgstedID'].astype(int)
kbh_merged['KredsNr'] = kbh_merged['KredsNr'].astype(int)
kbh_merged['StorKredsNr'] = kbh_merged['StorKredsNr'].astype(int)

fberg_merged['Gruppe'] = fberg_merged['Gruppe'].astype(int)
fberg_merged['ValgstedID'] = fberg_merged['ValgstedID'].astype(int)
fberg_merged['KredsNr'] = fberg_merged['KredsNr'].astype(int)
fberg_merged['StorKredsNr'] = fberg_merged['StorKredsNr'].astype(int)

# Show sample of final result
print(f"\nSample København result:")
print(kbh_merged[['Afstemningsområde', 'ValgstedID', 'ValgstedNavn', 'Gruppe', 'KredsNr', 'StorKredsNr']].head())

# Save results
kbh_merged.to_csv(PROJECT_ROOT + 'processed-data/elections/CPH_absolute_2025.csv', sep=';', index=False, encoding='utf-8')
fberg_merged.to_csv(PROJECT_ROOT + 'processed-data/elections/FRB_absolute_2025.csv', sep=';', index=False, encoding='utf-8')

print("\nFiles saved successfully!")