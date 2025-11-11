import threading
import time
import random
from collections import deque
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

# ---------------------------- Configuración ----------------------------
NUM_MAQUINAS = 6
SIM_STEP = 0.2  # segundos por tick (5 Hz)
HISTORY_SECONDS = 120
MAX_HISTORY_LEN = int(HISTORY_SECONDS / SIM_STEP)
OUTPUT_DIR = "outputs_control"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_PARAMS = {
    "temperature_setpoint": 70.0,
    "temperature_limit": 95.0,
    "torque_setpoint": 300.0,
    "torque_limit": 480.0,
    "vibration_setpoint": 2.0,
    "vibration_limit": 6.0,
}

SPONTANEOUS_FAILURE_PROB_PER_SEC = 0.0008
CONTROL_KP = {"temperature": 0.6, "torque": 0.2, "vibration": 0.5}
IA_THRESHOLDS = {
    "temperature_high": BASE_PARAMS["temperature_limit"],
    "torque_high": BASE_PARAMS["torque_limit"],
    "vibration_high": BASE_PARAMS["vibration_limit"],
    "dtemp_dt": 5.0,
    "dtorque_dt": 50.0,
    "dvib_dt": 1.5
}

# ---------------------------- Clases ----------------------------
class Maquina:
    def __init__(self, id, sector="Sector A"):
        self.id = id
        self.name = f"Maq-{id+1}"
        self.sector = sector
        self.temp_set = BASE_PARAMS["temperature_setpoint"]
        self.torque_set = BASE_PARAMS["torque_setpoint"]
        self.vib_set = BASE_PARAMS["vibration_setpoint"]
        self.temperature = self.temp_set + random.uniform(-2,2)
        self.torque = self.torque_set + random.uniform(-10,10)
        self.vibration = self.vib_set + random.uniform(-0.3,0.3)
        self.control_signal = {"temperature":0.0,"torque":0.0,"vibration":0.0}
        self.time = 0.0
        self.injected_variation = None
        self.last_alert_time = None
        self.history = {
            "time": deque(maxlen=MAX_HISTORY_LEN),
            "temperature": deque(maxlen=MAX_HISTORY_LEN),
            "torque": deque(maxlen=MAX_HISTORY_LEN),
            "vibration": deque(maxlen=MAX_HISTORY_LEN),
            "alert": deque(maxlen=MAX_HISTORY_LEN),
            "severity": deque(maxlen=MAX_HISTORY_LEN)
        }

    def step(self, dt):
        self.time += dt
        noise_temp = random.gauss(0,0.15)
        noise_torque = random.gauss(0,1.5)
        noise_vib = random.gauss(0,0.03)

        err_temp = self.temp_set - self.temperature
        self.control_signal["temperature"] = CONTROL_KP["temperature"] * err_temp
        err_torque = self.torque_set - self.torque
        self.control_signal["torque"] = CONTROL_KP["torque"] * err_torque
        err_vib = self.vib_set - self.vibration
        self.control_signal["vibration"] = CONTROL_KP["vibration"] * err_vib

        dtemp = self.control_signal["temperature"]*0.1 + noise_temp
        dtorque = self.control_signal["torque"]*0.05 + noise_torque
        dvib = self.control_signal["vibration"]*0.05 + noise_vib

        if self.injected_variation:
            var = self.injected_variation
            self.temperature += np.clip(var.get("dtemp",0.0)*dt,-var.get("max_dtemp",100),var.get("max_dtemp",100))
            self.torque += np.clip(var.get("dtorque",0.0)*dt,-var.get("max_dtorque",1000),var.get("max_dtorque",1000))
            self.vibration += np.clip(var.get("dvib",0.0)*dt,-var.get("max_dvib",10),var.get("max_dvib",10))
            var["remaining"] -= dt
            if var["remaining"] <= 0:
                self.injected_variation = None

        self.temperature += dtemp*dt
        self.torque += dtorque*dt
        self.vibration += dvib*dt

        self.temperature = float(np.clip(self.temperature,-50,500))
        self.torque = float(np.clip(self.torque,0,2000))
        self.vibration = float(np.clip(self.vibration,0,100))

        self.history["time"].append(self.time)
        self.history["temperature"].append(self.temperature)
        self.history["torque"].append(self.torque)
        self.history["vibration"].append(self.vibration)

    def inject_variation(self,dtemp=0.0,dtorque=0.0,dvib=0.0,duration=1.0,
                         max_dtemp=100.0,max_dtorque=1000.0,max_dvib=10.0,reason="Manual"):
        self.injected_variation = {
            "dtemp":dtemp,"dtorque":dtorque,"dvib":dvib,
            "remaining":duration,
            "max_dtemp":max_dtemp,"max_dtorque":max_dtorque,"max_dvib":max_dvib,
            "reason":reason
        }

class Sistema:
    def __init__(self,n_machines=NUM_MAQUINAS):
        self.machines = [Maquina(i,sector=f"Sector {chr(65 + (i//3))}") for i in range(n_machines)]
        self.running = False
        self.lock = threading.Lock()
        self.time = 0.0
        self.log_columns = ["timestamp","maquina","sector","temperature","torque","vibration","alert","severity","alert_type","recommendation"]
        self.log = []

    def step_all(self, dt):
        with self.lock:
            for m in self.machines:
                m.step(dt)
            self.time += dt
            alerts = self.detectar_anomalias()
            for m in self.machines:
                last_alert = any([a for a in alerts if a["maquina"]==m.name])
                sev = m.history["severity"][-1] if len(m.history["severity"])>0 else 0
                alert_type = next((a["type"] for a in alerts if a["maquina"]==m.name), "")
                rec = next((a["recommendation"] for a in alerts if a["maquina"]==m.name), "")
                self.log.append({
                    "timestamp":self.time,"maquina":m.name,"sector":m.sector,
                    "temperature":m.temperature,"torque":m.torque,"vibration":m.vibration,
                    "alert":int(last_alert),"severity":sev,"alert_type":alert_type,"recommendation":rec
                })
        return alerts

    def detectar_anomalias(self):
        alerts = []
        for m in self.machines:
            atype = None
            rec = ""
            cause = ""
            sev = 0
            if m.temperature > IA_THRESHOLDS["temperature_high"]:
                atype = "TEMPERATURE_HIGH"
                cause = "Temperatura superior al límite"
                rec = "Reducir carga / revisar refrigeración"
                sev = 2
            if m.torque > IA_THRESHOLDS["torque_high"]:
                atype = "TORQUE_HIGH" if atype is None else atype+"+TORQUE_HIGH"
                cause += " Torque elevado."
                sev = max(sev,2)
            if m.vibration > IA_THRESHOLDS["vibration_high"]:
                atype = "VIBRATION_HIGH" if atype is None else atype+"+VIBRATION_HIGH"
                cause += " Vibración elevada."
                sev = max(sev,2)
            if len(m.history["time"])>=2:
                dt = m.history["time"][-1]-m.history["time"][-2]
                dtemp = (m.history["temperature"][-1]-m.history["temperature"][-2])/dt
                dtorque = (m.history["torque"][-1]-m.history["torque"][-2])/dt
                dvib = (m.history["vibration"][-1]-m.history["vibration"][-2])/dt
                if abs(dtemp)>IA_THRESHOLDS["dtemp_dt"] or abs(dtorque)>IA_THRESHOLDS["dtorque_dt"] or abs(dvib)>IA_THRESHOLDS["dvib_dt"]:
                    sev = max(sev,1)
            if random.random()<SPONTANEOUS_FAILURE_PROB_PER_SEC*SIM_STEP:
                m.inject_variation(
                    dtemp=random.uniform(10,25),
                    dtorque=random.uniform(50,200),
                    dvib=random.uniform(1.0,4.0),
                    duration=1.0,
                    max_dtemp=50,
                    max_dtorque=500,
                    max_dvib=5,
                    reason="Spontaneous"
                )
                atype = "SPONTANEOUS_VARIATION" if atype is None else atype+"+SPONT"
                cause += " Evento espontáneo"
                rec += " Revisar componente mecánico."
                sev = max(sev,2)
            m.history["alert"].append(1 if sev>0 else 0)
            m.history["severity"].append(sev)
            if atype:
                alerts.append({"maquina":m.name,"sector":m.sector,"type":atype,"cause":cause.strip(),"recommendation":rec.strip()})
                m.last_alert_time = self.time
        return alerts

# ---------------------------- GUI ----------------------------
class AppGUI:
    def __init__(self,root,sistema):
        self.root=root
        self.sistema=sistema
        self.root.title("Control de Máquinas Inteligente")
        self._build_frames()
        self._build_left_controls()
        self._build_right_plot()
        self._build_bottom_alerts()
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop,daemon=True)
        self.update_thread.start()

    def _build_frames(self):
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0,column=0,sticky="ns")
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0,column=1,sticky="nsew")
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.grid(row=1,column=0,columnspan=2,sticky="ew")
        self.root.columnconfigure(1,weight=1)
        self.root.rowconfigure(0,weight=1)

    def _build_left_controls(self):
        ttk.Label(self.left_frame,text="Controles",font=("Arial",12,"bold")).pack(pady=5)
        for m in self.sistema.machines:
            btn = ttk.Button(self.left_frame,text=f"Inyectar variación {m.name}",command=lambda ma=m: self._inject_random(ma))
            btn.pack(pady=2,fill="x")
        ttk.Button(self.left_frame,text="Exportar Log CSV",command=self._export_csv).pack(pady=5,fill="x")

    def _build_right_plot(self):
        self.fig, self.axs = plt.subplots(3,1,figsize=(10,6),sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig,self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)

    def _build_bottom_alerts(self):
        ttk.Label(self.bottom_frame,text="Panel IA").pack(anchor="w")
        self.alert_text = scrolledtext.ScrolledText(self.bottom_frame,height=8,state="disabled")
        self.alert_text.pack(fill="x")

    def _inject_random(self,machine):
        machine.inject_variation(
            dtemp=random.uniform(5,15),
            dtorque=random.uniform(20,150),
            dvib=random.uniform(0.5,3.0),
            duration=2.0,
            max_dtemp=40,
            max_dtorque=400,
            max_dvib=5,
            reason="Manual"
        )

    def _update_loop(self):
        while self.running:
            alerts = self.sistema.step_all(SIM_STEP)
            self._update_plot()
            self._update_alert_panel(alerts)
            time.sleep(SIM_STEP)

    def _update_plot(self):
        self.axs[0].clear();self.axs[1].clear();self.axs[2].clear()
        for m in self.sistema.machines:
            t = list(m.history["time"])
            temp = list(m.history["temperature"])
            torque = list(m.history["torque"])
            vib = list(m.history["vibration"])
            sev = list(m.history["severity"])
            colors = ["blue" if s==0 else "orange" if s==1 else "red" for s in sev]
            self.axs[0].scatter(t,temp,c=colors,s=10,label=m.name)
            self.axs[1].scatter(t,torque,c=colors,s=10,label=m.name)
            self.axs[2].scatter(t,vib,c=colors,s=10,label=m.name)
        self.axs[0].set_ylabel("Temp (°C)");self.axs[1].set_ylabel("Torque");self.axs[2].set_ylabel("Vib")
        self.axs[2].set_xlabel("Tiempo (s)")
        self.axs[0].legend()
        self.canvas.draw()

    def _update_alert_panel(self,alerts):
        if not alerts: return
        self.alert_text.config(state="normal")
        for a in alerts:
            self.alert_text.insert(tk.END,f"[{a['maquina']}] {a['type']}: {a['cause']}. Rec: {a['recommendation']}\n")
        self.alert_text.see(tk.END)
        self.alert_text.config(state="disabled")

    def _export_csv(self):
        df = pd.DataFrame(self.sistema.log)
        filename = os.path.join(OUTPUT_DIR,f"log_{int(time.time())}.csv")
        df.to_csv(filename,index=False)
        messagebox.showinfo("Exportar Log",f"Archivo exportado: {filename}")

# ---------------------------- MAIN ----------------------------
if __name__=="__main__":
    sistema = Sistema()
    root = tk.Tk()
    app = AppGUI(root,sistema)
    root.mainloop()
