# tools.py
# LangChain tool definitions for the healthcare agent.
# Each function is decorated with @tool so the LLM can call them.

import json
import requests
import xmltodict
from langchain_core.tools import tool
from src.database import (
    find_patient, add_patient, update_patient_summary,
    find_doctors_by_specialty, get_available_slots, book_appointment,
    get_appointments, get_all_patients, get_all_doctors,
)
from src.rag_pipeline import retrieve_patient_info
from src.config import MEDLINE_BASE_URL


# --- patient management ---

@tool
def search_patient(query: str) -> str:
    """Search for a patient by name or phone number.
    Returns patient details if found."""
    patient = find_patient(query)
    if patient:
        return (
            f"Patient Found:\n"
            f"  Name: {patient['name']}\n"
            f"  Age: {patient['age']}\n"
            f"  Gender: {patient['gender']}\n"
            f"  Phone: {patient['phone']}\n"
            f"  Address: {patient['address']}\n"
            f"  Summary: {patient['summary'] or 'No summary available'}"
        )
    return f"No patient found matching '{query}'."


@tool
def register_patient(name: str, age: int, gender: str,
                     phone: str, address: str = "",
                     email: str = "") -> str:
    """Register a new patient or update an existing patient record.
    Requires name, age, gender, and phone number."""
    patient = add_patient(name, age, gender, phone, address, email)
    return (
        f"Patient registered successfully:\n"
        f"  Name: {patient['name']}\n"
        f"  ID: {patient['patient_id']}\n"
        f"  Phone: {patient['phone']}"
    )


@tool
def update_medical_record(patient_name: str, notes: str) -> str:
    """Update a patient's medical record with new notes.
    Requires the patient name and the clinical notes to add."""
    patient = find_patient(patient_name)
    if not patient:
        return f"Cannot update record: patient '{patient_name}' not found."
    updated = update_patient_summary(patient["patient_id"], notes)
    if updated:
        return (
            f"Medical record updated for {updated['name']}.\n"
            f"New entry added: {notes}"
        )
    return "Failed to update medical record."


@tool
def list_all_patients() -> str:
    """List all patients currently in the system."""
    patients = get_all_patients()
    if not patients:
        return "No patients registered in the system."
    lines = ["Current Patients:"]
    for p in patients:
        lines.append(
            f"  - {p['name']} | Age: {p['age']} | "
            f"Phone: {p['phone']} | Gender: {p['gender']}"
        )
    return "\n".join(lines)


# --- appointments & doctors ---

@tool
def find_doctor(specialty: str) -> str:
    """Find doctors by medical specialty (e.g. Nephrology, Cardiology,
    Endocrinology, General Medicine, Pulmonology)."""
    doctors = find_doctors_by_specialty(specialty)
    if not doctors:
        return f"No doctors found for specialty '{specialty}'."
    lines = [f"Doctors specializing in {specialty}:"]
    for doc in doctors:
        slots = get_available_slots(doc["doctor_id"])
        slot_str = ", ".join(
            f"{s['date']} at {s['time']}" for s in slots
        ) if slots else "No available slots"
        lines.append(
            f"  - {doc['name']} (ID: {doc['doctor_id']})\n"
            f"    Available: {slot_str}"
        )
    return "\n".join(lines)


@tool
def list_all_doctors_tool() -> str:
    """List all doctors and their specialties with available slots."""
    doctors = get_all_doctors()
    lines = ["All Doctors:"]
    for doc in doctors:
        slots = get_available_slots(doc["doctor_id"])
        slot_count = len(slots)
        lines.append(
            f"  - {doc['name']} | {doc['specialty']} | "
            f"{slot_count} slots available (ID: {doc['doctor_id']})"
        )
    return "\n".join(lines)


@tool
def book_appointment_tool(patient_name: str, doctor_id: str,
                          date: str, time: str) -> str:
    """Book a medical appointment for a patient.
    Requires patient name, doctor ID (e.g. D001), date (YYYY-MM-DD),
    and time (HH:MM)."""
    patient = find_patient(patient_name)
    if not patient:
        return f"Cannot book: patient '{patient_name}' not found."

    result = book_appointment(patient["patient_id"], doctor_id, date, time)
    if result["success"]:
        appt = result["appointment"]
        return (
            f"Appointment booked successfully!\n"
            f"  Appointment ID: {appt['appointment_id']}\n"
            f"  Patient: {patient['name']}\n"
            f"  Doctor: {appt['doctor_name']} ({appt['specialty']})\n"
            f"  Date/Time: {appt['date']} at {appt['time']}\n"
            f"  Status: {appt['status']}"
        )
    return f"Booking failed: {result['error']}"


@tool
def view_appointments(patient_name: str = "") -> str:
    """View appointments. If a patient name is provided, shows only their
    appointments; otherwise shows all appointments."""
    if patient_name:
        patient = find_patient(patient_name)
        if not patient:
            return f"Patient '{patient_name}' not found."
        appts = get_appointments(patient["patient_id"])
    else:
        appts = get_appointments()

    if not appts:
        return "No appointments found."

    lines = ["Appointments:"]
    for a in appts:
        lines.append(
            f"  [{a['appointment_id']}] {a['doctor_name']} ({a['specialty']})\n"
            f"    Date: {a['date']} at {a['time']} | Status: {a['status']}"
        )
    return "\n".join(lines)


# --- RAG (medical history from PDFs) ---

@tool
def retrieve_medical_history(query: str) -> str:
    """Retrieve relevant medical history from patient documents
    using semantic search (RAG). Provide a query about the patient
    or medical condition."""
    results = retrieve_patient_info(query, k=4)
    if not results:
        return "No relevant medical records found."

    lines = [f"Retrieved {len(results)} relevant records:"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n--- Record {i} (Source: {r['source']}, "
            f"Score: {r['relevance_score']}) ---\n"
            f"{r['content']}"
        )
    return "\n".join(lines)


# --- MedlinePlus search ---

@tool
def search_medical_info(query: str) -> str:
    """Search for medical information from MedlinePlus / NLM trusted
    medical sources. Use this for disease information, treatments,
    symptoms, and health conditions."""
    try:
        params = {
            "db": "healthTopics",
            "term": query,
            "retmax": 5,
        }
        response = requests.get(MEDLINE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = xmltodict.parse(response.text)
        feed = data.get("nlmSearchResult", {})
        results_list = feed.get("list", {}).get("document", [])

        if not results_list:
            return (
                f"No medical information found for '{query}'. "
                f"Try rephrasing your query."
            )

        if isinstance(results_list, dict):
            results_list = [results_list]

        lines = [f"Medical Information for '{query}':"]
        for item in results_list[:5]:
            title = item.get("@title", "No title")
            url = item.get("@url", "")

            # Extract snippet from content fields
            content_items = item.get("content", [])
            snippet = ""
            if isinstance(content_items, list):
                for c in content_items:
                    if isinstance(c, dict) and c.get("@name") == "FullSummary":
                        raw = c.get("#text", "")
                        # Strip HTML tags
                        import re
                        snippet = re.sub(r"<[^>]+>", "", raw)[:400]
                        break
            elif isinstance(content_items, dict):
                if content_items.get("@name") == "FullSummary":
                    raw = content_items.get("#text", "")
                    import re
                    snippet = re.sub(r"<[^>]+>", "", raw)[:400]

            lines.append(f"\n📌 {title}")
            if url:
                lines.append(f"   URL: {url}")
            if snippet:
                lines.append(f"   {snippet}...")

        return "\n".join(lines)

    except requests.RequestException as e:
        return f"Error searching medical information: {str(e)}"
    except Exception as e:
        return f"Error parsing medical results: {str(e)}"


# all tools in one list so the agent can bind them
ALL_TOOLS = [
    search_patient,
    register_patient,
    update_medical_record,
    list_all_patients,
    find_doctor,
    list_all_doctors_tool,
    book_appointment_tool,
    view_appointments,
    retrieve_medical_history,
    search_medical_info,
]
