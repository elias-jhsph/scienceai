#!/usr/bin/env python3
"""
Complete accounting of all papers by ID
"""

import pandas as pd

pub_years = pd.read_csv('/Users/elias/Documents/ScienceAI/scienceai_db/Scolaro Papers/csv_files/PublicationYearsAndTitles_2025-11-27_20_39_14.csv')
nonunion = pd.read_csv('/Users/elias/Documents/ScienceAI/scienceai_db/Scolaro Papers/csv_files/SmokingNonunion2x2Extraction_2025-11-28_04_17_41.csv')

all_paper_ids = set(pub_years['id'].tolist())
nonunion_ids = set(nonunion['id'].tolist())

# Scolaro's 8 mapped nonunion studies (user provided)
scolaro_nonunion_mapped = {
    'f9f2375ade': 'Adams 2001',
    '1e482c3c3a': 'Castillo 2005 (LEAP)',
    '920dc3d62b': 'Enninghorst 2011',
    'c0e963edee': 'Giannoudis 2000',
    'f3b3fb9a95': 'Moghaddam 2011',
    'f7e1801e7a': 'Ristiniemi 2007',
    'bc42862567': 'Taitsman 2009',
    'a154656aa1': 'Ziran 2007'
}

print('='*100)
print('COMPLETE ACCOUNTING OF SCOLARO 2014 PAPERS')
print('='*100)
print()

print('GLOBAL TOTALS:')
print(f'  Scolaro total papers: 19')
print(f'  Papers in our database: 16')
print(f'  Papers missing from database: 3')
print()

print('NONUNION TOTALS:')
print(f'  Scolaro nonunion studies (refs 13-22): 10')
print(f'  Scolaro nonunion mapped to IDs: 8')
print(f'  Scolaro nonunion NOT mapped: 2 (among the 3 missing papers)')
print()

print('='*100)
print('BREAKDOWN OF 8 MAPPED NONUNION STUDIES')
print('='*100)
print()

for paper_id, scolaro_ref in sorted(scolaro_nonunion_mapped.items(), key=lambda x: x[1]):
    in_db = paper_id in all_paper_ids
    extracted = paper_id in nonunion_ids
    
    # Get title
    title_row = pub_years[pub_years['id'] == paper_id]
    title = title_row.iloc[0]['paper_title'][:60] if len(title_row) > 0 else "NOT IN DB"
    
    db_status = '✅ IN DB' if in_db else '❌ MISSING'
    extract_status = '✅ EXTRACTED' if extracted else '❌ NOT EXTRACTED'
    
    print(f'{paper_id}: {scolaro_ref}')
    print(f'  {db_status} | {extract_status}')
    print(f'  Title: {title}...')
    print()

print('='*100)
print('SUMMARY BY STATUS')
print('='*100)
print()

# Group by status
in_and_extracted = [pid for pid in scolaro_nonunion_mapped.keys() 
                    if pid in all_paper_ids and pid in nonunion_ids]
in_not_extracted = [pid for pid in scolaro_nonunion_mapped.keys() 
                    if pid in all_paper_ids and pid not in nonunion_ids]
not_in_db = [pid for pid in scolaro_nonunion_mapped.keys() 
             if pid not in all_paper_ids]

print(f'✅ IN DATABASE + EXTRACTED: {len(in_and_extracted)}')
for pid in in_and_extracted:
    print(f'   - {pid}: {scolaro_nonunion_mapped[pid]}')
print()

print(f'⚠️ IN DATABASE + NOT EXTRACTED: {len(in_not_extracted)}')
for pid in in_not_extracted:
    print(f'   - {pid}: {scolaro_nonunion_mapped[pid]}')
print()

print(f'❌ NOT IN DATABASE: {len(not_in_db)}')
for pid in not_in_db:
    print(f'   - {pid}: {scolaro_nonunion_mapped[pid]}')
print()

print('='*100)
print('FINAL ACCOUNTING')
print('='*100)
print()
print('From Scolaro nonunion studies (10 total):')
print(f'  - {len(in_and_extracted)} in DB and extracted')
print(f'  - {len(in_not_extracted)} in DB but NOT extracted (flagging issue)')
print(f'  - {len(not_in_db)} NOT in DB (from 8 mapped)')
print(f'  - 2 not yet mapped to IDs (also NOT in DB)')
print()
print(f'Total accounted for: {len(in_and_extracted)} + {len(in_not_extracted)} + {len(not_in_db)} + 2 = {len(in_and_extracted) + len(in_not_extracted) + len(not_in_db) + 2}/10')
print()
print('From our database (16 papers):')
print(f'  - {len(in_and_extracted)} are Scolaro nonunion studies (extracted)')
print(f'  - {len(in_not_extracted)} are Scolaro nonunion studies (not extracted)')
print(f'  - {16 - len(in_and_extracted) - len(in_not_extracted)} are other Scolaro papers (not nonunion)')
