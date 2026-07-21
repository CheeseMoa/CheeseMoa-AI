# FastAPI 학습용 샘플 — 실제 프로젝트 코드 아님. 의존성은 requirements.txt에 없다(워커는 HTTP 미제공):
# 실행하려면 별도로 `pip install fastapi uvicorn` 후 `uvicorn healthcare_api:app`.
import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Healthcare API",
    description="A simple API for healthcare management",
    version="1.0.0",
)

# Mock databases
patients_db = {}
vitals_db = {}


class Patient(BaseModel):
    id: str
    name: str
    age: int
    gender: str
    blood_type: Optional[str] = None


class Vitals(BaseModel):
    heart_rate: int  # bpm
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    temperature: float  # Celsius
    timestamp: datetime.datetime = datetime.datetime.now()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Healthcare API", "status": "Healthy"}


@app.post("/patients/", response_model=Patient)
def create_patient(patient: Patient):
    if patient.id in patients_db:
        raise HTTPException(status_code=400, detail="Patient ID already exists")
    patients_db[patient.id] = patient
    vitals_db[patient.id] = []
    return patient


@app.get("/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: str):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patients_db[patient_id]


@app.post("/patients/{patient_id}/vitals")
def record_vitals(patient_id: str, vitals: Vitals):
    if patient_id not in patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    vitals_db[patient_id].append(vitals)
    return {"message": "Vitals recorded successfully", "vitals": vitals}


@app.get("/patients/{patient_id}/vitals", response_model=List[Vitals])
def get_vitals(patient_id: str):
    if patient_id not in vitals_db:
        raise HTTPException(status_code=404, detail="Patient not found")
    return vitals_db[patient_id]
