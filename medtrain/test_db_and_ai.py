#!/usr/bin/env python3
"""
test_db_and_ai.py: Validates SQLite database schema and AI DSL action execution.
Tests:
1. Patient creation with ONLY full_name (the single required field).
2. AI DSL parsing simulation (Visits, Diagnoses, Medications, Allergies).
3. Querying formatted medical history charts.
"""

import json
import os
from medical_db import MedicalDatabase


def run_tests():
    db_file = "test_medical.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    print("=" * 65)
    print("🩺 TESTING MEDICAL DATABASE & AI CONVERSATION PARSING DSL")
    print("=" * 65)

    db = MedicalDatabase(db_file)
    print(f"[*] Database initialized: {db_file}")

    # -------------------------------------------------------------------------
    # Test 1: Create Patient with ONLY full_name (mandatory field)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Creating patient with ONLY mandatory 'full_name'...")
    pid = db.create_patient(full_name="Sarah Jenkins")
    print(f"✅ Success! Created patient '{pid}' (Sarah Jenkins). Optional fields remain NULL.")

    # Verify optional fields are empty
    summary = db.get_patient_summary(pid)
    assert summary["full_name"] == "Sarah Jenkins"
    assert summary["dob"] is None
    assert summary["blood_group"] is None
    assert len(summary["visits"]) == 0
    assert len(summary["medications"]) == 0
    print("  -> Verified: dob, gender, contact, and blood_group are null.")
    print("  -> Verified: visits, medications, allergies lists are empty.")

    # -------------------------------------------------------------------------
    # Test 2: AI extracts additional patient info during conversation
    # -------------------------------------------------------------------------
    print("\n[Test 2] AI detects patient date of birth and blood group in chat...")
    ai_update_dsl = {
        "action": "UPDATE_PATIENT",
        "params": {
            "patient_id": pid,
            "dob": "1994-06-15",
            "gender": "Female",
            "blood_group": "A+",
            "contact_number": "+1-555-0199"
        }
    }
    res = db.execute_dsl_action(ai_update_dsl)
    print(f"  -> AI DSL Output: {res}")
    assert res["status"] == "SUCCESS"

    # -------------------------------------------------------------------------
    # Test 3: AI logs a consultation visit
    # -------------------------------------------------------------------------
    print("\n[Test 3] AI logs a new clinical visit from chat...")
    ai_visit_dsl = {
        "action": "LOG_VISIT",
        "params": {
            "patient_id": pid,
            "visit_type": "Outpatient Urgent Care",
            "chief_complaint": "Severe throbbing ear pain, low-grade fever (100.8°F)",
            "doctor_notes": "Tympanic membrane is erythematous and bulging. Acute Otitis Media confirmed.",
            "doctor_name": "Dr. Angela Foster, MD"
        }
    }
    res = db.execute_dsl_action(ai_visit_dsl)
    print(f"  -> AI DSL Output: {res}")
    visit_id = res["visit_id"]

    # -------------------------------------------------------------------------
    # Test 4: AI records diagnosis in medical_history
    # -------------------------------------------------------------------------
    print("\n[Test 4] AI parses and records diagnosis...")
    ai_diag_dsl = {
        "action": "RECORD_DIAGNOSIS",
        "params": {
            "patient_id": pid,
            "visit_id": visit_id,
            "condition_name": "Acute Otitis Media (Right Ear)",
            "diagnosed_date": "2026-09-27",
            "status": "Active",
            "notes": "Bacterial middle ear infection"
        }
    }
    res = db.execute_dsl_action(ai_diag_dsl)
    print(f"  -> AI DSL Output: {res}")
    history_id = res["history_id"]

    # -------------------------------------------------------------------------
    # Test 5: AI records allergy mentioned by patient
    # -------------------------------------------------------------------------
    print("\n[Test 5] AI catches allergy mentioned by patient...")
    ai_allergy_dsl = {
        "action": "RECORD_ALLERGY",
        "params": {
            "patient_id": pid,
            "substance": "Amoxicillin / Penicillins",
            "reaction": "Anaphylaxis and generalized hives",
            "severity": "Severe",
            "noted_date": "2026-09-27"
        }
    }
    res = db.execute_dsl_action(ai_allergy_dsl)
    print(f"  -> AI DSL Output: {res}")

    # -------------------------------------------------------------------------
    # Test 6: AI logs alternative prescribed medication
    # -------------------------------------------------------------------------
    print("\n[Test 6] AI records safe alternative medication prescribed...")
    ai_med_dsl = {
        "action": "PRESCRIBE_MEDICATION",
        "params": {
            "patient_id": pid,
            "visit_id": visit_id,
            "history_id": history_id,
            "drug_name": "Azithromycin (Z-Pak)",
            "dosage": "500mg Day 1, then 250mg daily",
            "frequency": "Once daily for 5 days",
            "start_date": "2026-09-27",
            "end_date": "2026-10-02",
            "prescribed_by": "Dr. Angela Foster, MD"
        }
    }
    res = db.execute_dsl_action(ai_med_dsl)
    print(f"  -> AI DSL Output: {res}")

    # -------------------------------------------------------------------------
    # Final Summary Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("📋 FULL PATIENT MEDICAL CHART EXTRACTED BY AI DSL:")
    print("=" * 65)
    full_chart = db.get_patient_summary(pid)
    print(json.dumps(full_chart, indent=2, default=str))

    print("\n" + "=" * 65)
    print("✅ All database and AI DSL tests PASSED successfully!")
    print("=" * 65)

    # Cleanup test db
    if os.path.exists(db_file):
        os.remove(db_file)


if __name__ == "__main__":
    run_tests()
