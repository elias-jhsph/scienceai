#!/usr/bin/env python3
"""
Generate discrepancy summary tables in CSV format
"""

import pandas as pd
import json

# Generate Discrepancy Matrix
discrepancy_matrix = pd.DataFrame([
    {
        "Outcome": "Nonunion",
        "Endpoint": "Study count",
        "Scolaro_Target": "10 studies",
        "System_Replicate": "6 studies",
        "Discrepancy_Type": "Data availability",
        "Root_Cause": "4 studies not extracted (40% data loss)",
        "Impact": "HIGH"
    },
    {
        "Outcome": "Nonunion",
        "Endpoint": "Pooled OR",
        "Scolaro_Target": "2.32 (1.76-3.06)",
        "System_Replicate": "2.52 (1.49-4.25) crude",
        "Discrepancy_Type": "Methodology + subset",
        "Root_Cause": "Crude vs adjusted + different study mix",
        "Impact": "MEDIUM"
    },
    {
        "Outcome": "Nonunion",
        "Endpoint": "Pooled RR",
        "Scolaro_Target": "N/A",
        "System_Replicate": "1.85 (1.28-2.68)",
        "Discrepancy_Type": "Effect measure",
        "Root_Cause": "RR vs OR (expected difference)",
        "Impact": "MEDIUM"
    },
    {
        "Outcome": "Nonunion",
        "Endpoint": "Publication bias",
        "Scolaro_Target": "Egger p=0.06",
        "System_Replicate": "Egger p=0.012",
        "Discrepancy_Type": "Statistical test",
        "Root_Cause": "Missing studies (likely larger, conservative ones)",
        "Impact": "MEDIUM"
    },
    {
        "Outcome": "Healing Time",
        "Endpoint": "Study count",
        "Scolaro_Target": "8 studies",
        "System_Replicate": "7 studies (with valid data)",
        "Discrepancy_Type": "Data availability",
        "Root_Cause": "1 study missing data",
        "Impact": "LOW"
    },
    {
        "Outcome": "Healing Time",
        "Endpoint": "Smokers - Nonsmokers",
        "Scolaro_Target": "+6.1 weeks",
        "System_Replicate": "+6.0 weeks",
        "Discrepancy_Type": "Calculation",
        "Root_Cause": "Excellent match (±0.1 weeks)",
        "Impact": "LOW"
    },
    {
        "Outcome": "Global",
        "Endpoint": "Total studies",
        "Scolaro_Target": "19 papers",
        "System_Replicate": "16 papers",
        "Discrepancy_Type": "Data availability",
        "Root_Cause": "3 papers not in database",
        "Impact": "MEDIUM"
    },
    {
        "Outcome": "Global",
        "Endpoint": "Total samples",
        "Scolaro_Target": "6,374 fractures",
        "System_Replicate": "6,068 samples",
        "Discrepancy_Type": "Data availability",
        "Root_Cause": "Missing studies + unit heterogeneity",
        "Impact": "MEDIUM"
    }
])

# Generate Study Extraction Status Table
studies_extracted = pd.DataFrame([
    {"Paper_ID": "1e482c3c3a", "Title": "LEAP Study - Impact of Smoking", "Nonunion": "❌", "Healing_Time": "✅", "Infection": "?", "Notes": "Healing time extracted"},
    {"Paper_ID": "3b3475937c", "Title": "Ankle Fractures Age>80", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "Hip fracture, not relevant"},
    {"Paper_ID": "620fb1773e", "Title": "SER II Conservative vs Operative", "Nonunion": "❌", "Healing_Time": "✅", "Infection": "?", "Notes": "Healing time extracted"},
    {"Paper_ID": "6d465f919c", "Title": "Effect of Smoking on Tibial Shaft", "Nonunion": "✅", "Healing_Time": "✅", "Infection": "?", "Notes": "Both outcomes extracted"},
    {"Paper_ID": "6dfe436834", "Title": "Smoking and hip fracture (3617 cases)", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "Hip fracture, not in Scolaro"},
    {"Paper_ID": "830bf21cf4", "Title": "Host Classification Infection", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "Infection focus"},
    {"Paper_ID": "920dc3d62b", "Title": "Open Tibia Timely Debridement", "Nonunion": "✅", "Healing_Time": "❌", "Infection": "?", "Notes": "Nonunion extracted"},
    {"Paper_ID": "9ac67a783a", "Title": "Ankle Fractures (906 patients)", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "Ankle fractures"},
    {"Paper_ID": "a154656aa1", "Title": "Allograft+DBM Osseous Healing", "Nonunion": "✅", "Healing_Time": "❌", "Infection": "?", "Notes": "Nonunion extracted"},
    {"Paper_ID": "bc42862567", "Title": "Risk Factors Femoral Nonunion", "Nonunion": "✅", "Healing_Time": "❌", "Infection": "?", "Notes": "Nonunion extracted"},
    {"Paper_ID": "c0e963edee", "Title": "Nonunion of femoral diaphysis", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "Not extracted yet"},
    {"Paper_ID": "c6c5269e19", "Title": "Alcohol Abusers Delayed Healing", "Nonunion": "❌", "Healing_Time": "✅", "Infection": "?", "Notes": "Alcohol confound"},
    {"Paper_ID": "da73e6f003", "Title": "Tibial shaft circular fixator", "Nonunion": "❌", "Healing_Time": "✅", "Infection": "?", "Notes": "Healing time extracted"},
    {"Paper_ID": "f3b3fb9a95", "Title": "Cigarette smoking tibial clinical", "Nonunion": "✅", "Healing_Time": "✅", "Infection": "?", "Notes": "Both outcomes extracted"},
    {"Paper_ID": "f7e1801e7a", "Title": "Two-ring Hybrid External Fixation", "Nonunion": "❌", "Healing_Time": "❌", "Infection": "?", "Notes": "No smoking data"},
    {"Paper_ID": "f9f2375ade", "Title": "Cigarette smoking open tibial", "Nonunion": "✅", "Healing_Time": "✅", "Infection": "?", "Notes": "Both outcomes extracted"}
])

# Generate Per-Study Nonunion Comparison
nonunion_comparison = pd.DataFrame([
    {"Paper_ID": "6d465f919c", "Title": "Tibial Shaft Healing", "Crude_RR": 6.67, "Crude_OR": 7.00, "Scolaro_OR": "Unknown", "In_Scolaro": "Yes (likely)"},
    {"Paper_ID": "920dc3d62b", "Title": "Open Tibia Debridement", "Crude_RR": 1.71, "Crude_OR": 2.26, "Scolaro_OR": "Unknown", "In_Scolaro": "Yes (likely)"},
    {"Paper_ID": "a154656aa1", "Title": "Allograft+DBM", "Crude_RR": 3.92, "Crude_OR": 5.06, "Scolaro_OR": "Unknown", "In_Scolaro": "Yes (likely)"},
    {"Paper_ID": "bc42862567", "Title": "Femoral Nonunion", "Crude_RR": 2.14, "Crude_OR": 3.25, "Scolaro_OR": "Unknown", "In_Scolaro": "Yes (likely)"},
    {"Paper_ID": "f3b3fb9a95", "Title": "Tibial Shaft Outcome", "Crude_RR": 16.17, "Crude_OR": 20.01, "Scolaro_OR": "~20.01 (Fig 4)", "In_Scolaro": "Yes (Moghaddam 2011)"},
    {"Paper_ID": "f9f2375ade", "Title": "Open Tibial Fractures", "Crude_RR": 1.32, "Crude_OR": 1.48, "Scolaro_OR": "~1.48 (Fig 4)", "In_Scolaro": "Yes (Adams 2001)"}
])

# Save tables
output_dir = "/Users/elias/Documents/ScienceAI/scienceai_db/Scolaro Papers/investigation_report"
import os
os.makedirs(output_dir, exist_ok=True)

discrepancy_matrix.to_csv(f"{output_dir}/discrepancy_matrix.csv", index=False)
studies_extracted.to_csv(f"{output_dir}/study_extraction_status.csv", index=False)
nonunion_comparison.to_csv(f"{output_dir}/nonunion_per_study_comparison.csv", index=False)

print("✅ Generated 3 summary tables:")
print(f"   1. {output_dir}/discrepancy_matrix.csv")
print(f"   2. {output_dir}/study_extraction_status.csv")
print(f"   3. {output_dir}/nonunion_per_study_comparison.csv")

# Display tables
print("\n" + "="*80)
print("DISCREPANCY MATRIX")
print("="*80)
print(discrepancy_matrix.to_string(index=False))

print("\n" + "="*80)
print("STUDY EXTRACTION STATUS")
print("="*80)
print(studies_extracted.to_string(index=False))

print("\n" + "="*80)
print("NONUNION PER-STUDY COMPARISON")
print("="*80)
print(nonunion_comparison.to_string(index=False))

# Generate JSON summary
summary = {
    "investigation_date": "2025-11-29",
    "scolaro_citation": "Scolaro et al. 2014, JBJS-Am, DOI:10.2106/JBJS.M.00081",
    "database_summary": {
        "total_papers": 16,
        "target_papers": 19,
        "missing_papers": 3,
        "total_samples": 6068,
        "target_samples": 6374
    },
    "nonunion_outcome": {
        "studies_extracted": 6,
        "studies_target": 10,
        "data_loss_percent": 40,
        "pooled_rr": "1.85 (1.28-2.68)",
        "pooled_or_crude": "2.52 (1.49-4.25)",
        "scolaro_or_adjusted": "2.32 (1.76-3.06)",
        "egger_p_system": 0.012,
        "egger_p_scolaro": 0.06,
        "discrepancy_drivers": [
            "Missing 4/10 studies (40% data loss)",
            "Crude vs adjusted estimates",
            "RR vs OR effect measure",
            "Different study mix in subset"
        ]
    },
    "healing_time_outcome": {
        "studies_extracted": 7,
        "studies_target": 8,
        "data_loss_percent": 12.5,
        "difference_weeks_system": 6.0,
        "difference_weeks_scolaro": 6.1,
        "discrepancy_weeks": 0.1,
        "assessment": "EXCELLENT MATCH"
    },
    "key_conclusions": [
        "Primary driver: DATA AVAILABILITY (missing studies and incomplete extraction)",
        "Healing time analysis: Near-perfect replication (±0.1 weeks)",
        "Nonunion analysis: 40% data loss explains most discrepancies",
        "Extraction quality: High accuracy where data exists",
        "Effect measure: Crude RR vs adjusted OR reduces comparability",
        "Direction preserved: Smokers have worse outcomes across all metrics"
    ],
    "recommendations": [
        "Priority 1: Locate and add 3 missing papers to database",
        "Priority 2: Extract nonunion data from 4 additional studies",
        "Priority 3: Consider extracting adjusted ORs when available",
        "Priority 4: Implement automated verification checks",
        "Priority 5: Document extraction decision rules"
    ]
}

with open(f"{output_dir}/investigation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Generated JSON summary: {output_dir}/investigation_summary.json")
print("\n" + "="*80)
print("All investigation outputs saved successfully!")
print("="*80)
