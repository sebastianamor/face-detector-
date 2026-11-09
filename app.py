import cv2
import face_recognition
import pickle
import sqlite3
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import numpy as np

# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    return conn

def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    
    # Verificar si ya se registró hoy para evitar duplicados
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()
        conn.close()
        return True # Se registró correctamente
    conn.close()
    return False # Ya estaba registrado

# --- INTERFAZ GRÁFICA (CustomTkinter) ---
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Facial Recognition Attendance System")
        self.geometry("1000x650")
        self.configure(fg_color="#0A192F") # Color de fondo oscuro

        # Cargar rostros conocidos
        try:
            with open("known_faces.pkl", "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            print("Error: No se encontró 'known_faces.pkl'. Ejecuta train.py primero.")
            self.destroy()
            return

        self.setup_ui()
        
        # Variables del video
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.running = True
        
        # Iniciar hilo de video
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.update_frame()

    def setup_ui(self):
        # Panel izquierdo (Video)
        self.video_frame = ctk.CTkFrame(self, width=500, height=400, fg_color="black", corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, rowspan=2, sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Panel derecho (Info y Registros)
        self.right_panel = ctk.CTkFrame(self, width=400, fg_color="#112240", corner_radius=10)
        self.right_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Etiqueta de estado de asistencia
        self.status_frame = ctk.CTkFrame(self.right_panel, fg_color="#1F6AA5", corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=20)
        self.status_label = ctk.CTkLabel(self.status_frame, text="Esperando reconocimiento...", font=("Arial", 18, "bold"))
        self.status_label.pack(pady=15)

        # Información de la persona
        self.info_label = ctk.CTkLabel(self.right_panel, text="Nombre: -\nFecha: -\nHora: -", font=("Arial", 16), justify="left")
        self.info_label.pack(anchor="w", padx=20, pady=10)

        # Tabla de registros (Simulada con un Textbox para simplificar)
        self.records_label = ctk.CTkLabel(self.right_panel, text="Registros Recientes", font=("Arial", 16, "bold"))
        self.records_label.pack(anchor="w", padx=20, pady=(20, 5))
        
        self.records_text = ctk.CTkTextbox(self.right_panel, height=200, font=("Courier", 12))
        self.records_text.pack(fill="x", padx=20, pady=5)
        self.update_records_table()

    def update_records_table(self):
        self.records_text.configure(state="normal")
        self.records_text.delete("1.0", "end")
        self.records_text.insert("end", f"{'ID':<5} | {'Nombre':<20} | {'Fecha':<12} | {'Hora':<10}\n")
        self.records_text.insert("end", "-"*55 + "\n")
        
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, date, time FROM attendance ORDER BY id DESC LIMIT 5")
        for row in cursor.fetchall():
            self.records_text.insert("end", f"{row[0]:<5} | {row[1]:<20} | {row[2]:<12} | {row[3]:<10}\n")
        conn.close()
        self.records_text.configure(state="disabled")

    def video_loop(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                continue

            # Reducir el tamaño para que vaya más rápido
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.data["encodings"], face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.data["encodings"], face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.data["names"][best_match_index]
                    
                    # Si se reconoce, actualizar UI y guardar en BD
                    was_marked = mark_attendance(name)
                    if was_marked:
                        self.after(0, self.update_ui_on_recognize, name)
                        self.update_records_table()

                face_names.append(name)

            # Dibujar en el frame original
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            # Guardar frame para mostrarlo en Tkinter
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def update_frame(self):
        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
        
        if self.running:
            self.after(30, self.update_frame) # Actualizar a ~30 FPS

    def update_ui_on_recognize(self, name):
        now = datetime.now()
        self.status_frame.configure(fg_color="#00C851") # Verde
        self.status_label.configure(text=f"¡Asistencia Registrada!\n{name}")
        self.info_label.configure(text=f"Nombre: {name}\nFecha: {now.strftime('%Y-%m-%d')}\nHora: {now.strftime('%H:%M:%S')}")

    def on_closing(self):
        self.running = False
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.destroy()

if __name__ == "__main__":
    init_db()
    app = AttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()