import os
import sys
import time
import pickle
import sqlite3
import queue
import threading
from datetime import datetime

import cv2
import numpy as np
import face_recognition
import customtkinter as ctk
from PIL import Image

DB_PATH = "attendance.db"
FACES_PATH = "known_faces.pkl"
TOLERANCE = 0.5  # más bajo = más estricto (0.45 si hay falsos positivos)


# --- BASE DE DATOS ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()


def mark_attendance(name):
    """Registra la asistencia si no existe hoy. Devuelve True si se registró."""
    conn = sqlite3.connect(DB_PATH)
    try:
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        cur = conn.execute("SELECT 1 FROM attendance WHERE name=? AND date=?", (name, date))
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                (name, date, time_str),
            )
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def names_registered_today():
    conn = sqlite3.connect(DB_PATH)
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        cur = conn.execute("SELECT name FROM attendance WHERE date=?", (today,))
        return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


# --- INTERFAZ ---
class AttendanceApp(ctk.CTk):
    def __init__(self, data):
        super().__init__()
        self.data = data

        self.title("Facial Recognition Attendance System")
        self.geometry("1100x620")
        self.configure(fg_color="#0A192F")

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.events = queue.Queue()  # comunicación hilo de video -> interfaz
        self.registered_today = names_registered_today()
        self.current_frame = None
        self.running = True

        self.setup_ui()

        # Cámara (CAP_DSHOW abre mucho más rápido en Windows)
        self.video_capture = cv2.VideoCapture(1, cv2.CAP_MSMF)
        if not self.video_capture.isOpened():
            self.status_label.configure(text="No se pudo abrir la cámara")
            self.status_frame.configure(fg_color="#C0392B")

        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        self.update_frame()

    def setup_ui(self):
        # Panel izquierdo (video)
        self.video_frame = ctk.CTkFrame(self, fg_color="black", corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Panel derecho
        self.right_panel = ctk.CTkFrame(self, fg_color="#112240", corner_radius=10)
        self.right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.status_frame = ctk.CTkFrame(self.right_panel, fg_color="#1F6AA5", corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=20)
        self.status_label = ctk.CTkLabel(
            self.status_frame, text="Esperando reconocimiento...", font=("Arial", 18, "bold")
        )
        self.status_label.pack(pady=15)

        self.info_label = ctk.CTkLabel(
            self.right_panel, text="Nombre: -\nFecha: -\nHora: -", font=("Arial", 16), justify="left"
        )
        self.info_label.pack(anchor="w", padx=20, pady=10)

        self.records_label = ctk.CTkLabel(
            self.right_panel, text="Registros Recientes", font=("Arial", 16, "bold")
        )
        self.records_label.pack(anchor="w", padx=20, pady=(20, 5))

        self.records_text = ctk.CTkTextbox(self.right_panel, height=200, font=("Courier", 12))
        self.records_text.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.update_records_table()

    def update_records_table(self):
        self.records_text.configure(state="normal")
        self.records_text.delete("1.0", "end")
        self.records_text.insert("end", f"{'ID':<5} | {'Nombre':<20} | {'Fecha':<12} | {'Hora':<10}\n")
        self.records_text.insert("end", "-" * 55 + "\n")

        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.execute(
                "SELECT id, name, date, time FROM attendance ORDER BY id DESC LIMIT 5"
            )
            for row in cur.fetchall():
                self.records_text.insert(
                    "end", f"{row[0]:<5} | {row[1]:<20} | {row[2]:<12} | {row[3]:<10}\n"
                )
        finally:
            conn.close()
        self.records_text.configure(state="disabled")

    # --- HILO DE VIDEO (no toca widgets de Tkinter) ---
    def video_loop(self):
        known_encodings = self.data["encodings"]
        known_names = self.data["names"]
        process_this_frame = True
        face_locations, face_names = [], []

        while self.running:
            if not self.video_capture.isOpened():
                time.sleep(0.2)
                continue

            ret, frame = self.video_capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Procesa un frame sí y otro no para ahorrar CPU
            if process_this_frame:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

                face_names = []
                for encoding in face_encodings:
                    name = "Unknown"
                    distances = face_recognition.face_distance(known_encodings, encoding)
                    best = int(np.argmin(distances))

                    if distances[best] < TOLERANCE:
                        name = known_names[best]

                        # Solo consulta la BD si aún no se registró hoy
                        if name not in self.registered_today:
                            if mark_attendance(name):
                                self.events.put(("marked", name))
                            self.registered_today.add(name)

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Dibujar resultados sobre el frame original
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1,
                )

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- HILO PRINCIPAL ---
    def update_frame(self):
        # Procesar eventos enviados por el hilo de video
        try:
            while True:
                kind, name = self.events.get_nowait()
                if kind == "marked":
                    self.update_ui_on_recognize(name)
                    self.update_records_table()
        except queue.Empty:
            pass

        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # mantener referencia

        if self.running:
            self.after(30, self.update_frame)

    def update_ui_on_recognize(self, name):
        now = datetime.now()
        self.status_frame.configure(fg_color="#00C851")
        self.status_label.configure(text=f"¡Asistencia Registrada!\n{name}")
        self.info_label.configure(
            text=f"Nombre: {name}\nFecha: {now.strftime('%Y-%m-%d')}\nHora: {now.strftime('%H:%M:%S')}"
        )

    def on_closing(self):
        self.running = False
        if self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.destroy()


def load_known_faces():
    if not os.path.exists(FACES_PATH):
        print(f"Error: no se encontró '{FACES_PATH}'. Ejecuta train.py primero.")
        sys.exit(1)

    with open(FACES_PATH, "rb") as f:
        data = pickle.load(f)

    if not data.get("encodings"):
        print("Error: 'known_faces.pkl' no contiene rostros. Ejecuta train.py con imágenes válidas.")
        sys.exit(1)

    return data


if __name__ == "__main__":
    data = load_known_faces()
    init_db()
    app = AttendanceApp(data)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()