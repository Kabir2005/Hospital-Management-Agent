import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("hospital.db")
cur = conn.cursor()

# Drop existing tables (if needed)
cur.execute("DROP TABLE IF EXISTS appointments")
cur.execute("DROP TABLE IF EXISTS doctors")

# Recreate doctors table with manual doctor_id
cur.execute("""
    CREATE TABLE doctors (
        doctor_id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT
    )
""")

# Manually assigned doctor IDs
doctors = [
    (1, "Dr. Mahesh Sharma", "Cardiology"),
    (2, "Dr. Arun Kumar", "Cardiology"),
    (3, "Dr. Gaurav Goel", "Neurology & Neurosurgery"),
    (4, "Dr. Ritu Yadav", "Neurology & Neurosurgery"),
    (5, "Dr. Dinesh Chandra", "Orthopedics"),
    (6, "Dr. Meenakshi Chauhan", "Orthopedics"),
    (7, "Dr. Richa Sharma", "Gynecology & Obstetrics"),
    (8, "Dr. Asha Mehta", "Gynecology & Obstetrics"),
    (9, "Dr. Shalini Verma", "Pediatrics"),
    (10, "Dr. Sanjay Raj", "Pediatrics"),
    (11, "Dr. Akhil Saxena", "ENT (Ear, Nose, Throat)"),
    (12, "Dr. Komal Gupta", "ENT (Ear, Nose, Throat)"),
    (13, "Dr. Priya Rana", "Dermatology"),
    (14, "Dr. Aman Mittal", "Dermatology"),
    (15, "Dr. Anuj Garg", "Gastroenterology"),
    (16, "Dr. Neha Jain", "Gastroenterology"),
    (17, "Dr. Karan Malhotra", "Urology"),
    (18, "Dr. Sameer Bhatt", "Urology")
]

cur.executemany("INSERT INTO doctors (doctor_id, name, department) VALUES (?, ?, ?)", doctors)

# Recreate appointments table
cur.execute("""
    CREATE TABLE appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        patient_name TEXT,
        doctor_id INTEGER,
        appointment_time TEXT,
        FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
    )
""")

# Insert sample appointments
appointments = [
    ("1234", "Kabir Gupta", 1, "2025-06-18 10:30"),
    ("1234", "Kabir Gupta", 2, "2025-06-20 09:00"),
    ("5678", "Neha Sharma", 3, "2025-06-22 14:00"),
]
cur.executemany("INSERT INTO appointments (patient_id, patient_name, doctor_id, appointment_time) VALUES (?, ?, ?, ?)", appointments)
# Create history table
cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        patient_name TEXT,
        visit_date TEXT,
        diagnosis TEXT
    )
""")

history_entries = [
    ("1234", "Kabir Gupta", "2023-09-14", "Hypertension"),
    ("1234", "Kabir Gupta", "2024-02-20", "Migraine"),
    ("5678", "Neha Sharma", "2024-01-05", "Fractured arm"),
]
cur.executemany("INSERT INTO history (patient_id, patient_name, visit_date, diagnosis) VALUES (?, ?, ?, ?)", history_entries)

# Commit and close
conn.commit()
cur.close()
conn.close()

print("Database setup with doctors, appointments, and history complete.")
