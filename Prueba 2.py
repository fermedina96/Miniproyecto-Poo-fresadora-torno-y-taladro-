"""
sistema_control_profesional.py
Simulación profesional integrada (Tkinter + Matplotlib + NumPy + Pandas)
- Varias máquinas (fábrica maderera)
- Variables: Temperatura, Torque, Vibración
- Control automático (P-control), detección de irregularidades (IA basada en reglas)
- GUI: controles, gráfica en tiempo real, panel IA con diagnóstico y recomendaciones
- Exporta histórico a CSV
Autor: Dario Sebastian Severo Perez
"""

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
SIM_STEP = 0.2             # segundos por tick (5 Hz)
HISTORY_SECONDS = 120      # histórico en la gráfica (segundos)
MAX_HISTORY_LEN = int(HISTORY_SECONDS / SIM_STEP)
OUTPUT_DIR = "outputs_control"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Límites nominales
BASE_PARAMS = {
    "temperature_setpoint": 70.0,
    "temperature_limit": 95.0,
    "torque_setpoint": 300.0,
    "torque_limit": 480.0,
    "vibration_setpoint": 2.0,
    "vibration_limit": 6.0,
}

SPONTANEOUS_FAILURE_PROB_PER_SEC = 0.001  # baja probabilidad

CONTROL_KP = {
    "temperature": 0.6,
    "torque": 0.2,
    "vibration": 0.5
}

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
            "alert": deque(maxlen=MAX_HISTORY_LEN)
        }

    def step(self, dt):
        self.time += dt
        # Ruido natural
        noise_temp = random.gauss(0,0.15)
        noise_torque = random.gauss(0,1.5)
        noise_vib = random.gauss(0,0.03)

        # Control P
        err_temp = self.temp_set - self.temperature
        self.control_signal["temperature"] = CONTROL_KP["temperature"]*err_temp

        err_torque = self.torque_set - self.torque
        self.control_signal["torque"] = CONTROL_KP["torque"]*err_torque

        err_vib = self.vib_set - self.vibration
        self.control_signal["vibration"] = CONTROL_KP["vibration"]*err_vib

        # Dinámica
        dtemp = self.control_signal["temperature"]*0.1 + noise_temp
        dtorque = self.control_signal["torque"]*0.05 + noise_torque
        dvib = self.control_signal["vibration"]*0.05 + noise_vib

        # Variación inyectada
        if self.injected_variation:
            var = self.injected_variation
            self.temperature += np.clip(var.get("dtemp",0.0)*dt, -var.get("max_dtemp",100), var.get("max_dtemp",100))
            self.torque += np.clip(var.get("dtorque",0.0)*dt, -var.get("max_dtorque",1000), var.get("max_dtorque",1000))
            self.vibration += np.clip(var.get("dvib",0.0)*dt, -var.get("max_dvib",10), var.get("max_dvib",10))
            var["remaining"] -= dt
            if var["remaining"] <= 0:
                self.injected_variation = None

        # Actualización normal
        self.temperature += dtemp*dt
        self.torque += dtorque*dt
        self.vibration += dvib*dt

        # Soft bounds
        self.temperature = float(np.clip(self.temperature,-50,500))
        self.torque = float(np.clip(self.torque,0,2000))
        self.vibration = float(np.clip(self.vibration,0,100))

        # Historial
        self.history["time"].append(self.time)
        self.history["temperature"].append(self.temperature)
        self.history["torque"].append(self.torque)
        self.history["vibration"].append(self.vibration)

    def inject_variation(self,dtemp=0.0,dtorque=0.0,dvib=0.0,duration=0.2,
                         max_dtemp=100.0,max_dtorque=1000.0,max_dvib=10.0,reason="Manual"):
        self.injected_variation = {
            "dtemp":dtemp,
            "dtorque":dtorque,
            "dvib":dvib,
            "remaining":duration,
            "max_dtemp":max_dtemp,
            "max_dtorque":max_dtorque,
            "max_dvib":max_dvib,
            "reason":reason
        }

# ---------------------------- Sistema ----------------------------
class Sistema:
    def __init__(self,n_machines=NUM_MAQUINAS):
        self.machines = [Maquina(i, sector=f"Sector {chr(65+(i//3))}") for i in range(n_machines)]
        self.running = False
        self.lock = threading.Lock()
        self.time = 0.0
        self.log_columns = ["timestamp","maquina","sector","temperature","torque","vibration","alert","alert_type","recommendation"]
        self.log = []

    def step_all(self, dt):
        with self.lock:
            for m in self.machines:
                m.step(dt)
            self.time += dt
            alerts = self.detectar_anomalias()
            for m in self.machines:
                last_alert = any([a for a in alerts if a["maquina"]==m.name])
                alert_type = next((a["type"] for a in alerts if a["maquina"]==m.name), "")
                rec = next((a["recommendation"] for a in alerts if a["maquina"]==m.name), "")
                self.log.append({
                    "timestamp":self.time,
                    "maquina":m.name,
                    "sector":m.sector,
                    "temperature":m.temperature,
                    "torque":m.torque,
                    "vibration":m.vibration,
                    "alert":int(last_alert),
                    "alert_type":alert_type,
                    "recommendation":rec
                })
        return alerts

    def detectar_anomalias(self):
        alerts=[]
        for m in self.machines:
            atype=None
            rec=""
            cause=""
            # threshold
            if m.temperature > IA_THRESHOLDS["temperature_high"]:
                atype="TEMPERATURE_HIGH"
                cause="Temperatura alta"
                rec="Reducir carga / revisar refrigeración"
            if m.torque > IA_THRESHOLDS["torque_high"]:
                atype="TORQUE_HIGH" if atype is None else atype+"+TORQUE_HIGH"
                cause+=" Torque elevado."
            if m.vibration > IA_THRESHOLDS["vibration_high"]:
                atype="VIBRATION_HIGH" if atype is None else atype+"+VIBRATION_HIGH"
                cause+=" Vibración elevada."
            # derivadas
            if len(m.history["time"])>=2:
                dt = m.history["time"][-1]-m.history["time"][-2]
                dt = dt if dt>0 else SIM_STEP
                dtemp = (m.history["temperature"][-1]-m.history["temperature"][-2])/dt
                dtorque = (m.history["torque"][-1]-m.history["torque"][-2])/dt
                dvib = (m.history["vibration"][-1]-m.history["vibration"][-2])/dt
                if abs(dtemp)>IA_THRESHOLDS["dtemp_dt"]:
                    atype="D_TEMP_HIGH" if atype is None else atype+"+D_TEMP_HIGH"
                    cause+=f" ΔT/dt={dtemp:.1f}"
                    rec+=" Revisar calor/fricción."
                if abs(dtorque)>IA_THRESHOLDS["dtorque_dt"]:
                    atype="D_TORQUE_HIGH" if atype is None else atype+"+D_TORQUE_HIGH"
                    cause+=f" ΔTorque/dt={dtorque:.1f}"
                    rec+=" Revisar alimentación."
                if abs(dvib)>IA_THRESHOLDS["dvib_dt"]:
                    atype="D_VIB_HIGH" if atype is None else atype+"+D_VIB_HIGH"
                    cause+=f" ΔVib/dt={dvib:.2f}"
                    rec+=" Revisar rodamientos."

            # Variación espontánea detectable
            if random.random()<SPONTANEOUS_FAILURE_PROB_PER_SEC*SIM_STEP:
                dtemp_sp = max(IA_THRESHOLDS["temperature_high"]-m.temperature+1.0, random.uniform(1,5))
                dtorque_sp = max(IA_THRESHOLDS["torque_high"]-m.torque+5.0, random.uniform(5,50))
                dvib_sp = max(IA_THRESHOLDS["vibration_high"]-m.vibration+0.2, random.uniform(0.2,2.0))
                m.inject_variation(dtemp=dtemp_sp,dtorque=dtorque_sp,dvib=dvib_sp,
                                   duration=SIM_STEP,max_dtemp=dtemp_sp,max_dtorque=dtorque_sp,
                                   max_dvib=dvib_sp,reason="Spontaneous")
                atype="SPONTANEOUS_VARIATION" if atype is None else atype+"+SPONT"
                cause+=" Evento espontáneo"
                rec+=" Revisar componentes."

            if atype:
                alerts.append({"maquina":m.name,"sector":m.sector,"type":atype,
                               "cause":cause.strip(),"recommendation":rec.strip()})
                m.last_alert_time=self.time
        return alerts

    def inject_manual_variation(self,machine_index,**kwargs):
        if 0<=machine_index<len(self.machines):
            self.machines[machine_index].inject_variation(**kwargs)

    def export_log_csv(self,filename=None):
        if filename is None:
            filename=os.path.join(OUTPUT_DIR,f"log_{int(time.time())}.csv")
        df=pd.DataFrame(self.log)
        df.to_csv(filename,index=False)
        return filename

# ---------------------------- GUI ----------------------------
class AppGUI:
    def __init__(self,root,sistema:Sistema):
        self.root=root
        self.system=sistema
        self.running=False
        self.update_thread=None

        root.title("Sistema de Supervisión y Control - Fábrica Maderera")
        root.geometry("1200x700")

        self.left_frame=ttk.Frame(root,width=300)
        self.left_frame.pack(side=tk.LEFT,fill=tk.Y,padx=6,pady=6)

        self.center_frame=ttk.Frame(root)
        self.center_frame.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=6,pady=6)

        self.right_frame=ttk.Frame(root,width=320)
        self.right_frame.pack(side=tk.RIGHT,fill=tk.Y,padx=6,pady=6)

        self._build_left_controls()
        self._build_center_plot()
        self._build_right_panel()

        self.status_var=tk.StringVar(value="Estado: detenido")
        self.status_bar=ttk.Label(root,textvariable=self.status_var,relief=tk.SUNKEN,anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM,fill=tk.X)

    def _build_left_controls(self):
        ttk.Label(self.left_frame,text="Controles",font=("Arial",14,"bold")).pack(pady=6)
        ttk.Button(self.left_frame,text="Iniciar simulación",command=self.start).pack(fill=tk.X,pady=4)
        ttk.Button(self.left_frame,text="Detener simulación",command=self.stop).pack(fill=tk.X,pady=4)

        ttk.Label(self.left_frame,text="Máquina:",font=("Arial",10)).pack(pady=(10,2))
        self.machine_sel=tk.IntVar(value=0)
        for i,m in enumerate(self.system.machines):
            rb=ttk.Radiobutton(self.left_frame,text=m.name,variable=self.machine_sel,value=i)
            rb.pack(anchor=tk.W)

        ttk.Label(self.left_frame,text="Inyectar variación (manual):",font=("Arial",10)).pack(pady=(10,2))
        frm_var=ttk.Frame(self.left_frame)
        frm_var.pack(fill=tk.X,pady=2)
        ttk.Label(frm_var,text="ΔT/s:").grid(row=0,column=0)
        self.entry_dtemp=tk.DoubleVar(value=5.0)
        ttk.Entry(frm_var,textvariable=self.entry_dtemp,width=8).grid(row=0,column=1)
        ttk.Label(frm_var,text="ΔTorque/s:").grid(row=1,column=0)
        self.entry_dtorque=tk.DoubleVar(value=40.0)
        ttk.Entry(frm_var,textvariable=self.entry_dtorque,width=8).grid(row=1,column=1)
        ttk.Label(frm_var,text="ΔVib/s:").grid(row=2,column=0)
        self.entry_dvib=tk.DoubleVar(value=1.0)
        ttk.Entry(frm_var,textvariable=self.entry_dvib,width=8).grid(row=2,column=1)
        ttk.Label(frm_var,text="Duración (s):").grid(row=3,column=0)
        self.entry_dur=tk.DoubleVar(value=0.2)
        ttk.Entry(frm_var,textvariable=self.entry_dur,width=8).grid(row=3,column=1)

        ttk.Button(self.left_frame,text="Inyectar variación",command=self._inject_variation).pack(pady=6,fill=tk.X)
        ttk.Button(self.left_frame,text="Exportar log a CSV",command=self._export_log).pack(pady=4,fill=tk.X)

    def _inject_variation(self):
        idx=self.machine_sel.get()
        self.system.inject_manual_variation(idx,dtemp=self.entry_dtemp.get(),
                                            dtorque=self.entry_dtorque.get(),
                                            dvib=self.entry_dvib.get(),
                                            duration=self.entry_dur.get())
        messagebox.showinfo("Variación inyectada","Variación inyectada con éxito.")

    def _export_log(self):
        filename=self.system.export_log_csv()
        messagebox.showinfo("Exportar CSV",f"Log exportado en:\n{filename}")

    def _build_center_plot(self):
        self.fig, self.axs = plt.subplots(3,1,figsize=(6,6))
        self.fig.tight_layout(pad=3.0)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.center_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.lines=[]
        for ax in self.axs:
            line, = ax.plot([],[],label="")
            ax.legend()
            self.lines.append(line)
        self.axs[0].set_ylabel("Temp (°C)")
        self.axs[1].set_ylabel("Torque")
        self.axs[2].set_ylabel("Vibration")

    def _build_right_panel(self):
        ttk.Label(self.right_frame,text="Panel de alertas",font=("Arial",14,"bold")).pack(pady=6)
        self.text_alerts=scrolledtext.ScrolledText(self.right_frame,height=40)
        self.text_alerts.pack(fill=tk.BOTH,expand=True)

    def start(self):
        if not self.running:
            self.running=True
            self.system.running=True
            self.status_var.set("Estado: ejecutando")
            self.update_thread=threading.Thread(target=self._update_loop,daemon=True)
            self.update_thread.start()

    def stop(self):
        self.running=False
        self.system.running=False
        self.status_var.set("Estado: detenido")

    def _update_loop(self):
        while self.running:
            alerts=self.system.step_all(SIM_STEP)
            self._update_plot()
            self._update_alerts(alerts)
            time.sleep(SIM_STEP)

    def _update_plot(self):
        for idx,m in enumerate(self.system.machines):
            t = list(m.history["time"])
            self.lines[0].set_data(t,list(m.history["temperature"]))
            self.lines[1].set_data(t,list(m.history["torque"]))
            self.lines[2].set_data(t,list(m.history["vibration"]))
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()

    def _update_alerts(self,alerts):
        self.text_alerts.delete("1.0",tk.END)
        for a in alerts:
            self.text_alerts.insert(tk.END,f"[{a['maquina']} | {a['sector']}] {a['type']}: {a['cause']}\nRec: {a['recommendation']}\n\n")
        self.text_alerts.see(tk.END)

# ---------------------------- Main ----------------------------
def main():
    root=tk.Tk()
    sistema=Sistema()
    app=AppGUI(root,sistema)
    root.mainloop()
    sistema.export_log_csv()

if __name__=="__main__":
    main()
