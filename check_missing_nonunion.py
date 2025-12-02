#!/usr/bin/env python3
"""
Critical Finding: Identify why 3 Scolaro nonunion studies weren't extracted
"""

import pandas as pd

# Load data
mapping = pd.read_csv('/Users/elias/Documents/ScienceAI/scienceai_db/Scolaro Papers/csv_files/FractureSmokingMapping_v2_2025-11-27_22_27_16.csv')
nonunion_csv = pd.read_csv('/Users/elias/Documents/ScienceAI/scienceai_db/Scolaro Papers/csv_files/SmokingNonunion2x2Extraction_2025-11-28_04_17_41.csv')

# The 3 papers that should have nonunion but don't
missing_nonunion = {
    '1e482c3c3a': 'Castillo 2005 (LEAP)',
    'c0e963edee': 'Giannoudis 2000 - Nonunion femoral diaphysis',
    'f7e1801e7a': 'Ristiniemi 2007 - Two-ring Hybrid Fixation'
}

print('='*80)
print('CRITICAL FINDING: 3 SCOLARO NONUNION STUDIES NOT EXTRACTED')
print('='*80)
print()

for paper_id, ref in missing_nonunion.items():
    target = mapping[mapping['id'] == paper_id]
    
    if len(target) == 0:
        print(f'❌ {paper_id}: NOT IN MAPPING FILE')
        continue
    
    row = target.iloc[0]
    
    print(f'\n📄 {paper_id}: {ref}')
    print(f'   Title: {row["paper_title"]}')
    print()
    
    # Check all nonunion-related fields
    print('   NONUNION FLAGS IN MAPPING:')
    print(f'     nonunion_reported: {row.get("nonunion_reported_value", "N/A")}')
    print(f'     nonunion_by_smoking: {row.get("nonunion_by_smoking_value", "N/A")}')
    print(f'     nonunion_analysis_reported: {row.get("nonunion_analysis_reported_value", "N/A")}')
    print()
    
    # Get quotes
    if pd.notna(row.get('nonunion_by_smoking_source_quote')):
        print(f'   QUOTE:')
        print(f'     {str(row["nonunion_by_smoking_source_quote"])[:300]}...')
        print(f'     Location: {row.get("nonunion_by_smoking_source_location", "N/A")}')
        print()
    
    # Check smoking exposure
    print(f'   SMOKING COMPARISON: {row.get("smoking_exposure_comparison_value", "N/A")}')
    print()

print('='*80)
print('EXPLANATION OF DISCREPANCY')
print('='*80)
print()
print('If nonunion_by_smoking is NOT flagged as "yes", then the paper was')
print('not sent for 2x2 table extraction, explaining the gap.')
print()
print('Scolaro likely:')
print('1. Manually extracted nonunion data from these papers')
print('2. Used data from tables/text not captured by our automated system')
print('3. Had access to unpublished data or author correspondence')
