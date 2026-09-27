#!/usr/bin/env python3
"""
medical_db.py: SQLite Database Manager for Medical Records and AI DSL Parsing.
Implements the flexible schema where full_name is the only mandatory field,
and visits, history, medications, treatments, and allergies are populated dynamically.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name       TEXT NOT NULL,
    dob             DATE,
    gender          TEXT,
    contact_number  TEXT,
    blood_group     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    visit_date      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    visit_type      TEXT,
    chief_complaint TEXT,
    doctor_notes    TEXT,
    doctor_name     TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS medical_history (
    history_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    visit_id        INTEGER,
    condition_name  TEXT NOT NULL,
    diagnosed_date  DATE,
    status          TEXT,
    notes           TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS treatments (
    treatment_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    history_id      INTEGER,
    visit_id        INTEGER,
    treatment_type  TEXT,
    description     TEXT,
    start_date      DATE,
    end_date        DATE,
    outcome         TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (history_id) REFERENCES medical_history(history_id) ON DELETE SET NULL,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS medications (
    medication_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    history_id      INTEGER,
    visit_id        INTEGER,
    drug_name       TEXT NOT NULL,
    dosage          TEXT,
    frequency       TEXT,
    start_date      DATE,
    end_date        DATE,
    prescribed_by   TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (history_id) REFERENCES medical_history(history_id) ON DELETE SET NULL,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS allergies (
    allergy_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      INTEGER NOT NULL,
    substance       TEXT NOT NULL,
    reaction        TEXT,
    severity        TEXT,
    noted_date      DATE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);
"""


class MedicalDatabase:
    """Manages SQLite operations and AI DSL actions for the medical system."""

    def __init__(self, db_path: str = "medical_records.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes tables and foreign key constraints."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)

    # -------------------------------------------------------------------------
    # PATIENT OPERATIONS
    # -------------------------------------------------------------------------

    def create_patient(
        self,
        full_name: str,
        dob: Optional[str] = None,
        gender: Optional[str] = None,
        contact_number: Optional[str] = None,
        blood_group: Optional[str] = None,
    ) -> int:
        """
        Creates a new patient. 'full_name' is the only mandatory field.
        Returns the auto-assigned patient_id.
        """
        if not full_name or not full_name.strip():
            raise ValueError("full_name is strictly required to create a patient.")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO patients (full_name, dob, gender, contact_number, blood_group)
                VALUES (?, ?, ?, ?, ?)
                """,
                (full_name.strip(), dob, gender, contact_number, blood_group),
            )
            return cursor.lastrowid

    def update_patient_details(self, patient_id: int, **fields) -> bool:
        """Updates optional patient fields (dob, gender, contact, blood_group)."""
        valid_fields = {"full_name", "dob", "gender", "contact_number", "blood_group"}
        updates = {k: v for k, v in fields.items() if k in valid_fields and v is not None}
        if not updates:
            return False

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [patient_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE patients SET {set_clause} WHERE patient_id = ?", values)
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # CLINICAL ENTITY OPERATIONS
    # -------------------------------------------------------------------------

    def add_visit(
        self,
        patient_id: int,
        visit_type: Optional[str] = "Consultation",
        chief_complaint: Optional[str] = None,
        doctor_notes: Optional[str] = None,
        doctor_name: Optional[str] = None,
    ) -> int:
        """Records a new clinical consultation/visit."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO visits (patient_id, visit_type, chief_complaint, doctor_notes, doctor_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (patient_id, visit_type, chief_complaint, doctor_notes, doctor_name),
            )
            return cursor.lastrowid

    def add_medical_history(
        self,
        patient_id: int,
        condition_name: str,
        visit_id: Optional[int] = None,
        diagnosed_date: Optional[str] = None,
        status: Optional[str] = "Active",
        notes: Optional[str] = None,
    ) -> int:
        """Logs a diagnosed medical condition."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO medical_history (patient_id, visit_id, condition_name, diagnosed_date, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (patient_id, visit_id, condition_name, diagnosed_date, status, notes),
            )
            return cursor.lastrowid

    def add_medication(
        self,
        patient_id: int,
        drug_name: str,
        dosage: Optional[str] = None,
        frequency: Optional[str] = None,
        history_id: Optional[int] = None,
        visit_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        prescribed_by: Optional[str] = None,
    ) -> int:
        """Records a prescribed medication."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO medications (patient_id, history_id, visit_id, drug_name, dosage, frequency, start_date, end_date, prescribed_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (patient_id, history_id, visit_id, drug_name, dosage, frequency, start_date, end_date, prescribed_by),
            )
            return cursor.lastrowid

    def add_treatment(
        self,
        patient_id: int,
        treatment_type: str,
        description: Optional[str] = None,
        history_id: Optional[int] = None,
        visit_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> int:
        """Records a treatment or therapeutic procedure."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO treatments (patient_id, history_id, visit_id, treatment_type, description, start_date, end_date, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (patient_id, history_id, visit_id, treatment_type, description, start_date, end_date, outcome),
            )
            return cursor.lastrowid

    def add_allergy(
        self,
        patient_id: int,
        substance: str,
        reaction: Optional[str] = None,
        severity: Optional[str] = None,
        noted_date: Optional[str] = None,
    ) -> int:
        """Records a known allergy."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO allergies (patient_id, substance, reaction, severity, noted_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (patient_id, substance, reaction, severity, noted_date),
            )
            return cursor.lastrowid

    # -------------------------------------------------------------------------
    # SUMMARY & QUERY METHODS
    # -------------------------------------------------------------------------

    def get_patient_summary(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a complete medical chart for a patient as a structured dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Patient header
            cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            p_row = cursor.fetchone()
            if not p_row:
                return None
            patient = dict(p_row)

            # Visits
            cursor.execute("SELECT * FROM visits WHERE patient_id = ? ORDER BY visit_date DESC", (patient_id,))
            patient["visits"] = [dict(r) for r in cursor.fetchall()]

            # Medical History
            cursor.execute("SELECT * FROM medical_history WHERE patient_id = ? ORDER BY diagnosed_date DESC", (patient_id,))
            patient["medical_history"] = [dict(r) for r in cursor.fetchall()]

            # Medications
            cursor.execute("SELECT * FROM medications WHERE patient_id = ? ORDER BY start_date DESC", (patient_id,))
            patient["medications"] = [dict(r) for r in cursor.fetchall()]

            # Treatments
            cursor.execute("SELECT * FROM treatments WHERE patient_id = ? ORDER BY start_date DESC", (patient_id,))
            patient["treatments"] = [dict(r) for r in cursor.fetchall()]

            # Allergies
            cursor.execute("SELECT * FROM allergies WHERE patient_id = ?", (patient_id,))
            patient["allergies"] = [dict(r) for r in cursor.fetchall()]

            return patient

    # -------------------------------------------------------------------------
    # AI DSL ACTION EXECUTOR
    # -------------------------------------------------------------------------

    def execute_dsl_action(self, action_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a structured action emitted by the AI model during conversation parsing.
        Example action_payload:
            {
                "action": "CREATE_PATIENT" | "RECORD_DIAGNOSIS" | "PRESCRIBE_MEDICATION" | "RECORD_ALLERGY",
                "params": { ... }
            }
        """
        action = action_payload.get("action", "").upper()
        params = action_payload.get("params", {})

        try:
            if action == "CREATE_PATIENT":
                pid = self.create_patient(
                    full_name=params["full_name"],
                    dob=params.get("dob"),
                    gender=params.get("gender"),
                    contact_number=params.get("contact_number"),
                    blood_group=params.get("blood_group"),
                )
                return {"status": "SUCCESS", "action": action, "patient_id": pid}

            elif action == "UPDATE_PATIENT":
                pid = params.get("patient_id")
                update_data = {k: v for k, v in params.items() if k != "patient_id"}
                updated = self.update_patient_details(pid, **update_data)
                return {"status": "SUCCESS" if updated else "NO_CHANGE", "action": action, "patient_id": pid}

            elif action == "LOG_VISIT":
                vid = self.add_visit(
                    patient_id=params["patient_id"],
                    visit_type=params.get("visit_type", "Consultation"),
                    chief_complaint=params.get("chief_complaint"),
                    doctor_notes=params.get("doctor_notes"),
                    doctor_name=params.get("doctor_name"),
                )
                return {"status": "SUCCESS", "action": action, "visit_id": vid}

            elif action == "RECORD_DIAGNOSIS":
                hid = self.add_medical_history(
                    patient_id=params["patient_id"],
                    condition_name=params["condition_name"],
                    visit_id=params.get("visit_id"),
                    diagnosed_date=params.get("diagnosed_date"),
                    status=params.get("status", "Active"),
                    notes=params.get("notes"),
                )
                return {"status": "SUCCESS", "action": action, "history_id": hid}

            elif action == "PRESCRIBE_MEDICATION":
                mid = self.add_medication(
                    patient_id=params["patient_id"],
                    drug_name=params["drug_name"],
                    dosage=params.get("dosage"),
                    frequency=params.get("frequency"),
                    history_id=params.get("history_id"),
                    visit_id=params.get("visit_id"),
                    start_date=params.get("start_date"),
                    end_date=params.get("end_date"),
                    prescribed_by=params.get("prescribed_by"),
                )
                return {"status": "SUCCESS", "action": action, "medication_id": mid}

            elif action == "RECORD_ALLERGY":
                aid = self.add_allergy(
                    patient_id=params["patient_id"],
                    substance=params["substance"],
                    reaction=params.get("reaction"),
                    severity=params.get("severity"),
                    noted_date=params.get("noted_date"),
                )
                return {"status": "SUCCESS", "action": action, "allergy_id": aid}

            else:
                return {"status": "ERROR", "message": f"Unknown DSL action: {action}"}

        except Exception as e:
            return {"status": "ERROR", "action": action, "error": str(e)}
