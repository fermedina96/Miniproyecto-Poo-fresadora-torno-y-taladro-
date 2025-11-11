# sistema_control_final_reparar_y_lecturas.py
"""
Simulador final:
- 3 máquinas (Fresadora, Torno, Taladro)
- Variables: Temperatura, Torque, Vibración, Potencia, RPM
- Pestañas: cada máquina con 5 gráficas (una debajo de otra)
- Lecturas numéricas en tiempo real junto a cada conjunto de gráficas
- Detección de fallas: cuando ocurre, SE PARA TODO, se marca la gráfica y pestaña,
  se muestra popup y se pregunta si reparar ahora. Si aceptás, repara y reanuda.
  Si no, queda pausado hasta que pulses "Reparar falla (máquina seleccionada)".
- Botón "Reparar falla" también disponible.
- Botón "Simular Falla Manual" que fuerza una falla y la deja registrada.
- Exportar log CSV.
"""

import threading, time, random, os
from collections import deque
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ---------------- config ----------------
SIM_STEP = 1.0                # 1 segundo
HISTORY_SECONDS = 300
MAX_HISTORY_LEN = int(HISTORY_SECONDS / SIM_STEP)
OUTPUT_DIR = "outputs_control"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPONTANEOUS_FAILURE_PROB_PER_SEC = 0.0009
CONTROL_KP = {"temperature": 0.6, "torque": 0.2, "vibration": 0.5, "power": 0.3, "rpm": 0.1}

IA_THRESHOLDS_BASE = {
    "dtemp_dt": 5.0,
    "dtorque_dt": 50.0,
    "dvib_dt": 1.5,
    "dpower_dt": 10.0,
    "drpm_dt": 200.0
}
MACHINE_COLORS = ["tab:blue", "tab:green", "tab:orange"]
MACHINE_NAMES = ["Fresadora", "Torno", "Taladro"]

MACHINE_PROFILE = {
    "Fresadora": {"temp_range": (40,70), "torque_range": (40,70), "vib_range": (0.2,0.8),
                  "power_range": (70,95), "rpm_range": (1500,2500), "opt_temp":60.0},
    "Torno":    {"temp_range": (35,60), "torque_range": (30,60), "vib_range": (0.1,0.6),
                  "power_range": (60,90), "rpm_range": (1200,2000), "opt_temp":50.0},
    "Taladro":  {"temp_range": (30,50), "torque_range": (10,30), "vib_range": (0.05,0.4),
                  "power_range": (50,85), "rpm_range": (800,1500), "opt_temp":40.0},
}

# ---------------- Maquina ----------------
class Maquina:
    def __init__(self, idx, name):
        self.id = idx
        self.name = name
        self.sector = f"Sector {idx+1}"   # ✅ ahora existe sector

        profile = MACHINE_PROFILE[name]
        self.opt_temp = profile["opt_temp"]
        self.temp_set = self.opt_temp
        self.torque_set = (profile["torque_range"][0] + profile["torque_range"][1]) / 2
        self.vib_set = (profile["vib_range"][0] + profile["vib_range"][1]) / 2
        self.power_set = (profile["power_range"][0] + profile["power_range"][1]) / 2
        self.rpm_set = (profile["rpm_range"][0] + profile["rpm_range"][1]) / 2

        # initial readings
        self.temperature = self.temp_set + random.uniform(-1.5,1.5)
        self.torque = self.torque_set + random.uniform(-5,5)
        self.vibration = self.vib_set + random.uniform(-0.1,0.1)
        self.power = self.power_set + random.uniform(-2,2)
        self.rpm = self.rpm_set + random.uniform(-50,50)

        self._state = {"temperature":self.temperature, "torque":self.torque,
                       "vibration":self.vibration, "power":self.power, "rpm":self.rpm}

        self.control = {"temperature":0.0, "torque":0.0, "vibration":0.0, "power":0.0, "rpm":0.0}
        self.time = 0.0
        self.injected_variation = None

        self.tolerances = {"temperature_pct":10.0, "torque_pct":15.0, "vibration_pct":50.0,
                           "power_pct":20.0, "rpm_pct":15.0}

        self.history = {
            "time": deque(maxlen=MAX_HISTORY_LEN),
            "temperature": deque(maxlen=MAX_HISTORY_LEN),
            "torque": deque(maxlen=MAX_HISTORY_LEN),
            "vibration": deque(maxlen=MAX_HISTORY_LEN),
            "power": deque(maxlen=MAX_HISTORY_LEN),
            "rpm": deque(maxlen=MAX_HISTORY_LEN),
            "alert": deque(maxlen=MAX_HISTORY_LEN),
            "severity": deque(maxlen=MAX_HISTORY_LEN)
        }

        self.active = True       # True = operating, False = stopped by fault
        self.fault_info = None   # {"variable":..., "message":..., "time":...}

    def _ou_step(self, key, target, tau, sigma, dt):
        x = self._state[key]
        dx = - (x - target) * (dt / tau) + sigma * (dt**0.5) * random.gauss(0,1)
        x_new = x + dx
        self._state[key] = x_new
        return x_new

    def step(self, dt):
        # If inactive due to fault: just append current values (flat line)
        if not self.active:
            self.time += dt
            self.history["time"].append(self.time)
            self.history["temperature"].append(self.temperature)
            self.history["torque"].append(self.torque)
            self.history["vibration"].append(self.vibration)
            self.history["power"].append(self.power)
            self.history["rpm"].append(self.rpm)
            return

        self.time += dt
        # simple P-control to setpoints
        self.control["temperature"] = CONTROL_KP["temperature"] * (self.temp_set - self.temperature)
        self.control["torque"] = CONTROL_KP["torque"] * (self.torque_set - self.torque)
        self.control["vibration"] = CONTROL_KP["vibration"] * (self.vib_set - self.vibration)
        self.control["power"] = CONTROL_KP["power"] * (self.power_set - self.power)
        self.control["rpm"] = CONTROL_KP["rpm"] * (self.rpm_set - self.rpm)

        profile = MACHINE_PROFILE[self.name]
        # OU-like updates
        self._ou_step("temperature", self.temp_set + self.control["temperature"]*0.05, tau=8.0, sigma=0.2, dt=dt)
        self._ou_step("torque", self.torque_set + self.control["torque"]*0.02, tau=10.0, sigma=0.8, dt=dt)
        self._ou_step("vibration", self.vib_set + self.control["vibration"]*0.01, tau=6.0, sigma=0.02, dt=dt)
        self._ou_step("power", self.power_set + self.control["power"]*0.02, tau=8.0, sigma=0.6, dt=dt)
        self._ou_step("rpm", self.rpm_set + self.control["rpm"]*0.5, tau=15.0, sigma=30.0, dt=dt)

        # injected variation
        if self.injected_variation:
            var = self.injected_variation
            self._state["temperature"] += np.clip(var.get("dtemp",0.0)*dt, -var.get("max_dtemp",100), var.get("max_dtemp",100))
            self._state["torque"] += np.clip(var.get("dtorque",0.0)*dt, -var.get("max_dtorque",1000), var.get("max_dtorque",1000))
            self._state["vibration"] += np.clip(var.get("dvib",0.0)*dt, -var.get("max_dvib",10), var.get("max_dvib",10))
            self._state["power"] += np.clip(var.get("dpower",0.0)*dt, -var.get("max_dpower",100), var.get("max_dpower",100))
            self._state["rpm"] += np.clip(var.get("drpm",0.0)*dt, -var.get("max_drpm",5000), var.get("max_drpm",5000))
            var["remaining"] -= dt
            if var["remaining"] <= 0:
                self.injected_variation = None

        # update readings from state and clip reasonably
        self.temperature = float(np.clip(self._state["temperature"], profile["temp_range"][0]-10, profile["temp_range"][1]+30))
        self.torque = float(np.clip(self._state["torque"], profile["torque_range"][0]*0.5, profile["torque_range"][1]*1.5))
        self.vibration = float(np.clip(self._state["vibration"], 0.0, profile["vib_range"][1]*3.0))
        self.power = float(np.clip(self._state["power"], 0.0, 150.0))
        self.rpm = float(np.clip(self._state["rpm"], profile["rpm_range"][0]*0.5, profile["rpm_range"][1]*1.5))

        # append to history
        self.history["time"].append(self.time)
        self.history["temperature"].append(self.temperature)
        self.history["torque"].append(self.torque)
        self.history["vibration"].append(self.vibration)
        self.history["power"].append(self.power)
        self.history["rpm"].append(self.rpm)

    def inject_variation(self, dtemp=0.0, dtorque=0.0, dvib=0.0, dpower=0.0, drpm=0.0, duration=0.5,
                         max_dtemp=100.0, max_dtorque=1000.0, max_dvib=10.0, max_dpower=100.0, max_drpm=5000.0, reason="Manual"):
        self.injected_variation = {
            "dtemp":dtemp, "dtorque":dtorque, "dvib":dvib, "dpower":dpower, "drpm":drpm,
            "remaining":duration, "max_dtemp":max_dtemp, "max_dtorque":max_dtorque,
            "max_dvib":max_dvib, "max_dpower":max_dpower, "max_drpm":max_drpm,
            "reason":reason
        }

    def stop_for_fault(self, variable_name, message, timestamp):
        self.active = False
        self.fault_info = {"variable":variable_name, "message":message, "time":timestamp}

    def repair(self):
        # normalize state gently to setpoints and reactivate
        self._state["temperature"] = self.temp_set + random.uniform(-0.5,0.5)
        self._state["torque"] = self.torque_set + random.uniform(-1,1)
        self._state["vibration"] = self.vib_set + random.uniform(-0.02,0.02)
        self._state["power"] = self.power_set + random.uniform(-1,1)
        self._state["rpm"] = self.rpm_set + random.uniform(-20,20)
        # update immediate readings
        self.temperature = float(self._state["temperature"])
        self.torque = float(self._state["torque"])
        self.vibration = float(self._state["vibration"])
        self.power = float(self._state["power"])
        self.rpm = float(self._state["rpm"])
        self.injected_variation = None
        self.fault_info = None
        self.active = True

# ---------------- Sistema ----------------
class Sistema:
    def __init__(self):
        self.machines = [Maquina(i, MACHINE_NAMES[i]) for i in range(len(MACHINE_NAMES))]
        self.lock = threading.Lock()
        self.time = 0.0
        self.running = False
        self.log = []
        self.log_columns = ["timestamp","maquina","temperature","torque","vibration","power","rpm",
                            "alert","severity","alert_type","recommendation"]

    def step_all(self, dt):
        with self.lock:
            for m in self.machines:
                m.step(dt)
            self.time += dt
            alerts = self.detectar_anomalias()
            # append log
            for m in self.machines:
                last_alert = any(a for a in alerts if a["maquina"]==m.name)
                sev = m.history["severity"][-1] if len(m.history["severity"])>0 else 0
                atype = next((a["type"] for a in alerts if a["maquina"]==m.name), "")
                rec = next((a["recommendation"] for a in alerts if a["maquina"]==m.name), "")
                self.log.append({
                    "timestamp":self.time,
                    "maquina":m.name,
                    "temperature":m.temperature,
                    "torque":m.torque,
                    "vibration":m.vibration,
                    "power":m.power,
                    "rpm":m.rpm,
                    "alert":int(last_alert),
                    "severity":sev,
                    "alert_type":atype,
                    "recommendation":rec
                })
        return alerts

    def detectar_anomalias(self):
        alerts=[]
        for m in self.machines:
            # if machine already stopped by a fault, keep its alert history and skip new detection
            if not m.active:
                m.history["alert"].append(1 if m.fault_info else 0)
                m.history["severity"].append(2 if m.fault_info else 0)
                continue

            atype=None; rec=""; cause=""; sev=0
            temp_tol_abs = m.opt_temp * (m.tolerances["temperature_pct"]/100.0)
            temp_high = m.opt_temp + temp_tol_abs
            temp_low = m.opt_temp - temp_tol_abs
            torque_thresh = m.torque_set*(1.0 + m.tolerances["torque_pct"]/100.0)
            vib_thresh = m.vib_set*(1.0 + m.tolerances["vibration_pct"]/100.0)
            power_thresh = m.power_set*(1.0 + m.tolerances["power_pct"]/100.0)
            rpm_thresh = m.rpm_set*(1.0 + m.tolerances["rpm_pct"]/100.0)

            # checks
            if m.temperature > temp_high:
                atype="TEMPERATURE_HIGH"; cause += f"Temp {m.temperature:.1f}°C > opt+tol ({temp_high:.1f}°C)."; rec += "Reducir carga/revisar refrigeración."; sev=max(sev,2)
            if m.temperature < temp_low:
                atype = "TEMPERATURE_LOW" if atype is None else atype + "+TEMPERATURE_LOW"; cause += f"Temp {m.temperature:.1f}°C < opt-tol ({temp_low:.1f}°C)."; rec += "Revisar calefacción."; sev=max(sev,1)
            if m.torque > torque_thresh:
                atype = "TORQUE_HIGH" if atype is None else atype + "+TORQUE_HIGH"; cause += f" Torque {m.torque:.1f} > {torque_thresh:.1f}."; rec += " Revisar alimentación."; sev=max(sev,2)
            if m.vibration > vib_thresh:
                atype = "VIBRATION_HIGH" if atype is None else atype + "+VIBRATION_HIGH"; cause += f" Vib {m.vibration:.2f} > {vib_thresh:.2f}."; rec += " Revisar rodamientos."; sev=max(sev,2)
            if m.power > power_thresh:
                atype = "POWER_HIGH" if atype is None else atype + "+POWER_HIGH"; cause += f" Pot {m.power:.1f}% > {power_thresh:.1f}%."; rec += " Reducir demanda."; sev=max(sev,1)
            if m.rpm > rpm_thresh:
                atype = "RPM_HIGH" if atype is None else atype + "+RPM_HIGH"; cause += f" RPM {m.rpm:.0f} > {rpm_thresh:.0f}."; rec += " Ajustar velocidad."; sev=max(sev,1)

            # derivatives
            if len(m.history["time"])>=2:
                dt = m.history["time"][-1]-m.history["time"][-2]; dt = dt if dt>0 else SIM_STEP
                dtemp = (m.history["temperature"][-1]-m.history["temperature"][-2])/dt
                dtorque = (m.history["torque"][-1]-m.history["torque"][-2])/dt
                dvib = (m.history["vibration"][-1]-m.history["vibration"][-2])/dt
                dpow = (m.history["power"][-1]-m.history["power"][-2])/dt
                drpm = (m.history["rpm"][-1]-m.history["rpm"][-2])/dt
                if abs(dtemp) > IA_THRESHOLDS_BASE["dtemp_dt"]:
                    atype = "D_TEMP_HIGH" if atype is None else atype + "+D_TEMP_HIGH"; cause += f" ΔT/dt={dtemp:.1f}."; rec += " Revisar fricción."; sev=max(sev,1)
                if abs(dtorque) > IA_THRESHOLDS_BASE["dtorque_dt"]:
                    atype = "D_TORQUE_HIGH" if atype is None else atype + "+D_TORQUE_HIGH"; cause += f" ΔTorque/dt={dtorque:.1f}."; rec += " Revisar alimentación."; sev=max(sev,1)
                if abs(dvib) > IA_THRESHOLDS_BASE["dvib_dt"]:
                    atype = "D_VIB_HIGH" if atype is None else atype + "+D_VIB_HIGH"; cause += f" ΔVib/dt={dvib:.2f}."; rec += " Revisar rodamientos."; sev=max(sev,1)
                if abs(dpow) > IA_THRESHOLDS_BASE["dpower_dt"]:
                    atype = "D_POWER_HIGH" if atype is None else atype + "+D_POWER_HIGH"; cause += f" ΔPower/dt={dpow:.1f}."; rec += " Revisar demanda eléctrica."; sev=max(sev,1)
                if abs(drpm) > IA_THRESHOLDS_BASE["drpm_dt"]:
                    atype = "D_RPM_HIGH" if atype is None else atype + "+D_RPM_HIGH"; cause += f" ΔRPM/dt={drpm:.0f}."; rec += " Revisar transmisión."; sev=max(sev,1)

            # spontaneous
            if random.random() < SPONTANEOUS_FAILURE_PROB_PER_SEC * SIM_STEP:
                m.inject_variation(
                    dtemp=random.uniform(6,20),
                    dtorque=random.uniform(30,180),
                    dvib=random.uniform(0.6,3.5),
                    dpower=random.uniform(8,30),
                    drpm=random.uniform(100,600),
                    duration=0.6,
                    max_dtemp=50, max_dtorque=500, max_dvib=5, max_dpower=80, max_drpm=1000,
                    reason="Spontaneous"
                )
                atype = "SPONTANEOUS_VARIATION" if atype is None else atype + "+SPONT"
                cause += " Evento espontáneo."
                rec += " Revisar componente mecánico/eléctrico."
                sev = max(sev,2)

            m.history["alert"].append(1 if sev>0 else 0)
            m.history["severity"].append(sev)

            if atype:
                # determine primary var
                primary_var = ("temperature" if "TEMP" in atype else
                               "torque" if "TORQUE" in atype else
                               "vibration" if "VIB" in atype else
                               "power" if "POWER" in atype else
                               "rpm")
                m.stop_for_fault(primary_var, cause.strip(), self.time)
                alerts.append({
                    "maquina":m.name,
                    "sector":m.sector,
                    "type":atype,
                    "cause":cause.strip(),
                    "recommendation":rec.strip(),
                    "primary_var":primary_var
                })
                m.last_alert_time = self.time
        return alerts

    def inject_manual_variation(self, idx, **kwargs):
        if 0 <= idx < len(self.machines):
            self.machines[idx].inject_variation(**kwargs)

    def export_log_csv(self, filename=None):
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, f"log_{int(time.time())}.csv")
        df = pd.DataFrame(self.log, columns=self.log_columns)
        df.to_csv(filename, index=False)
        return filename

# ---------------- GUI ----------------
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.system = Sistema()
        self.running = False
        self.update_thread = None
        self.paused_by_fault = False
        self.last_alerts = []

        root.title("Simulador Control - Reparar y Lecturas")
        root.geometry("1400x820")

        self.left_frame = ttk.Frame(root, width=360); self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        self.center_frame = ttk.Frame(root); self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.right_frame = ttk.Frame(root, width=420); self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        self._build_left_controls()
        self._build_center_notebook()
        self._build_right_panel()

        self.status_var = tk.StringVar(value="Estado: detenido")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._refresh_ui()

    # ----------- PANEL IZQUIERDO -----------
    def _build_left_controls(self):
        ttk.Label(self.left_frame, text="Controles Generales", font=("Arial",12,"bold")).pack(pady=6)
        ttk.Button(self.left_frame, text="Iniciar simulación", command=self.start).pack(fill=tk.X, pady=4)
        ttk.Button(self.left_frame, text="Detener simulación", command=self.stop).pack(fill=tk.X, pady=4)
        ttk.Button(self.left_frame, text="Exportar log a CSV", command=self._export_log).pack(fill=tk.X, pady=6)
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(self.left_frame, text="Seleccionar máquina:", font=("Arial",10)).pack(pady=(6,2))
        self.machine_sel_cb = ttk.Combobox(self.left_frame, state="readonly",
                                           values=[m.name for m in self.system.machines])
        self.machine_sel_cb.current(0)
        self.machine_sel_cb.pack(fill=tk.X, padx=6)
        ttk.Button(self.left_frame, text="Ir a máquina", command=self._go_to_selected_machine).pack(pady=4, fill=tk.X)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(self.left_frame, text="Inyectar variación (máquina seleccionada)", font=("Arial",10,"bold")).pack(pady=4)
        frm = ttk.Frame(self.left_frame); frm.pack(fill=tk.X, padx=6)
        ttk.Label(frm, text="ΔT/s:").grid(row=0,column=0, sticky="w")
        self.entry_dtemp = tk.DoubleVar(value=6.0); ttk.Entry(frm, textvariable=self.entry_dtemp, width=8).grid(row=0,column=1)
        ttk.Label(frm, text="ΔTorque/s:").grid(row=1,column=0, sticky="w")
        self.entry_dtorque = tk.DoubleVar(value=40.0); ttk.Entry(frm, textvariable=self.entry_dtorque, width=8).grid(row=1,column=1)
        ttk.Label(frm, text="ΔVib/s:").grid(row=2,column=0, sticky="w")
        self.entry_dvib = tk.DoubleVar(value=1.0); ttk.Entry(frm, textvariable=self.entry_dvib, width=8).grid(row=2,column=1)
        ttk.Label(frm, text="ΔPower/s:").grid(row=3,column=0, sticky="w")
        self.entry_dpower = tk.DoubleVar(value=12.0); ttk.Entry(frm, textvariable=self.entry_dpower, width=8).grid(row=3,column=1)
        ttk.Label(frm, text="ΔRPM/s:").grid(row=4,column=0, sticky="w")
        self.entry_drpm = tk.DoubleVar(value=200.0); ttk.Entry(frm, textvariable=self.entry_drpm, width=8).grid(row=4,column=1)
        ttk.Label(frm, text="Duración(s):").grid(row=5,column=0, sticky="w")
        self.entry_dur = tk.DoubleVar(value=0.6); ttk.Entry(frm, textvariable=self.entry_dur, width=8).grid(row=5,column=1)
        ttk.Button(self.left_frame, text="Inyectar variación", command=self._inject_variation_selected).pack(pady=6, fill=tk.X, padx=6)

        ttk.Button(self.left_frame, text="Reiniciar máquina seleccionada", command=self._restart_selected_machine).pack(pady=6, fill=tk.X, padx=6)
        ttk.Button(self.left_frame, text="Reparar falla (máquina seleccionada)", command=self._repair_selected_machine).pack(pady=6, fill=tk.X, padx=6)
        ttk.Button(self.left_frame, text="⚠️ Simular Falla Manual", command=self._simulate_manual_fault).pack(pady=6, fill=tk.X, padx=6)

    # ----------- PANEL CENTRAL -----------
    def _build_center_notebook(self):
        self.notebook = ttk.Notebook(self.center_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        self.figures=[]; self.axes=[]; self.canvases=[]; self.lines=[]; self.reading_labels=[]

        for i,m in enumerate(self.system.machines):
            tab = ttk.Frame(self.notebook); self.notebook.add(tab, text=m.name)
            # figure with 5 plots (stacked vertically)
            fig, axs = plt.subplots(5,1,figsize=(7,8), sharex=True)
            fig.tight_layout(pad=3.0)
            l_temp, = axs[0].plot([], [], color=MACHINE_COLORS[i], label="Temperatura (°C)")
            l_torque, = axs[1].plot([], [], color=MACHINE_COLORS[i], label="Torque")
            l_vib, = axs[2].plot([], [], color=MACHINE_COLORS[i], label="Vibración")
            l_power, = axs[3].plot([], [], color=MACHINE_COLORS[i], label="Potencia (%)")
            l_rpm, = axs[4].plot([], [], color=MACHINE_COLORS[i], label="RPM")
            for ax in axs: ax.legend()
            axs[0].set_ylabel("Temp (°C)"); axs[1].set_ylabel("Torque"); axs[2].set_ylabel("Vib"); axs[3].set_ylabel("Power (%)"); axs[4].set_ylabel("RPM")
            axs[4].set_xlabel("Tiempo (s)")

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # right side: reading labels
            read_frame = ttk.Frame(tab, width=220)
            read_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
            lbl_title = ttk.Label(read_frame, text=f"Lecturas actuales - {m.name}", font=("Arial",10,"bold"))
            lbl_title.pack(pady=(4,8))
            # one label per variable
            l_temp_val = ttk.Label(read_frame, text="Temp: -- °C", font=("Arial",10)); l_temp_val.pack(anchor="w", pady=2)
            l_torque_val = ttk.Label(read_frame, text="Torque: -- N·m", font=("Arial",10)); l_torque_val.pack(anchor="w", pady=2)
            l_vib_val = ttk.Label(read_frame, text="Vibración: -- mm/s", font=("Arial",10)); l_vib_val.pack(anchor="w", pady=2)
            l_power_val = ttk.Label(read_frame, text="Potencia: -- %", font=("Arial",10)); l_power_val.pack(anchor="w", pady=2)
            l_rpm_val = ttk.Label(read_frame, text="RPM: --", font=("Arial",10)); l_rpm_val.pack(anchor="w", pady=2)

            # tolerances panel
            frm_tol = ttk.LabelFrame(read_frame, text="Tolerancias y Temp óptima")
            frm_tol.pack(fill=tk.X, pady=(10,4))
            ttk.Label(frm_tol, text="Temp opt (°C):").grid(row=0,column=0, sticky="w")
            temp_opt_var = tk.DoubleVar(value=m.opt_temp); ttk.Entry(frm_tol, textvariable=temp_opt_var, width=8).grid(row=0,column=1)
            ttk.Label(frm_tol, text="Temp % tol:").grid(row=1,column=0, sticky="w")
            temp_tol_var = tk.DoubleVar(value=m.tolerances["temperature_pct"]); ttk.Entry(frm_tol, textvariable=temp_tol_var, width=8).grid(row=1,column=1)
            ttk.Label(frm_tol, text="Torque % tol:").grid(row=2,column=0, sticky="w")
            torque_tol_var = tk.DoubleVar(value=m.tolerances["torque_pct"]); ttk.Entry(frm_tol, textvariable=torque_tol_var, width=8).grid(row=2,column=1)
            ttk.Label(frm_tol, text="Vib % tol:").grid(row=3,column=0, sticky="w")
            vib_tol_var = tk.DoubleVar(value=m.tolerances["vibration_pct"]); ttk.Entry(frm_tol, textvariable=vib_tol_var, width=8).grid(row=3,column=1)
            ttk.Label(frm_tol, text="Power % tol:").grid(row=4,column=0, sticky="w")
            power_tol_var = tk.DoubleVar(value=m.tolerances["power_pct"]); ttk.Entry(frm_tol, textvariable=power_tol_var, width=8).grid(row=4,column=1)
            ttk.Label(frm_tol, text="RPM % tol:").grid(row=5,column=0, sticky="w")
            rpm_tol_var = tk.DoubleVar(value=m.tolerances["rpm_pct"]); ttk.Entry(frm_tol, textvariable=rpm_tol_var, width=8).grid(row=5,column=1)

            def make_apply(machine=m, opt=temp_opt_var, tt=temp_tol_var, tq=torque_tol_var, vb=vib_tol_var, pw=power_tol_var, rp=rpm_tol_var):
                def apply_t():
                    machine.opt_temp = float(opt.get()); machine.temp_set = machine.opt_temp
                    machine.tolerances["temperature_pct"] = float(tt.get())
                    machine.tolerances["torque_pct"] = float(tq.get())
                    machine.tolerances["vibration_pct"] = float(vb.get())
                    machine.tolerances["power_pct"] = float(pw.get())
                    machine.tolerances["rpm_pct"] = float(rp.get())
                    messagebox.showinfo("Tolerancias","Tolerancias aplicadas.")
                return apply_t
            ttk.Button(frm_tol, text="Aplicar", command=make_apply()).grid(row=6,column=0, columnspan=2, pady=6)

            # alert box
            ttk.Label(read_frame, text="Alertas (últimas):").pack(anchor="w", pady=(6,0))
            alert_box = scrolledtext.ScrolledText(read_frame, height=6, state="disabled")
            alert_box.pack(fill=tk.X, pady=(0,6))

            # store refs
            self.figures.append(fig); self.axes.append(axs); self.canvases.append(canvas)
            self.lines.append({"temp":l_temp,"torque":l_torque,"vib":l_vib,"power":l_power,"rpm":l_rpm,"alert_box":alert_box})
            self.reading_labels.append({"temp":l_temp_val,"torque":l_torque_val,"vib":l_vib_val,"power":l_power_val,"rpm":l_rpm_val})

    # ----------- PANEL DERECHO -----------
    def _build_right_panel(self):
        ttk.Label(self.right_frame, text="Panel Global de Alertas", font=("Arial",12,"bold")).pack(pady=6)
        self.text_alerts = scrolledtext.ScrolledText(self.right_frame, height=22, state="disabled")
        self.text_alerts.pack(fill=tk.BOTH, expand=True, padx=6)
        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Button(self.right_frame, text="Exportar log CSV", command=self._export_log).pack(fill=tk.X, padx=6, pady=4)

    # ---------------- control actions ----------------
    def start(self):
        if not self.running and not self.paused_by_fault:
            self.running = True
            self.system.running = True
            self.status_var.set("Estado: ejecutando")
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop(self):
        self.running = False
        self.system.running = False
        self.status_var.set("Estado: detenido")

    def _go_to_selected_machine(self):
        idx = self.machine_sel_cb.current()
        if 0 <= idx < len(self.system.machines):
            self.notebook.select(idx)

    def _inject_variation_selected(self):
        idx = self.machine_sel_cb.current()
        if not (0 <= idx < len(self.system.machines)):
            messagebox.showwarning("Seleccionar","Seleccioná una máquina válida.")
            return
        dtemp = float(self.entry_dtemp.get())
        dtorque = float(self.entry_dtorque.get())
        dvib = float(self.entry_dvib.get())
        dpower = float(self.entry_dpower.get())
        drpm = float(self.entry_drpm.get())
        dur = float(self.entry_dur.get())
        self.system.inject_manual_variation(
            idx,
            dtemp=dtemp, dtorque=dtorque, dvib=dvib, dpower=dpower, drpm=drpm, duration=dur,
            max_dtemp=abs(dtemp)*2+1, max_dtorque=abs(dtorque)*2+5,
            max_dvib=max(1.0,abs(dvib)*2), max_dpower=abs(dpower)*2+5,
            max_drpm=abs(drpm)*2+50
        )
        messagebox.showinfo("Inyectar","Variación inyectada.")

    def _restart_selected_machine(self):
        idx = self.machine_sel_cb.current()
        if not (0 <= idx < len(self.system.machines)):
            messagebox.showwarning("Seleccionar","Seleccioná una máquina válida.")
            return
        m = self.system.machines[idx]
        if m.active:
            messagebox.showinfo("Reiniciar",f"{m.name} ya está operativa.")
            return
        m.repair()
        self._restore_visual(idx)
        messagebox.showinfo("Reiniciar", f"{m.name} reiniciada.")

    def _repair_selected_machine(self):
        idx = self.machine_sel_cb.current()
        if not (0 <= idx < len(self.system.machines)):
            messagebox.showwarning("Seleccionar","Seleccioná una máquina válida.")
            return
        m = self.system.machines[idx]
        if m.fault_info is None:
            messagebox.showinfo("Reparar","No hay falla activa.")
            return
        # repair and resume everything
        m.repair()
        self._restore_visual(idx)
        # resume global simulation
        self.paused_by_fault = False
        if not self.running:
            self.start()
        messagebox.showinfo("Reparado", f"Falla reparada en {m.name}. Simulación reanudada.")

    def _simulate_manual_fault(self):
        """Provoca una falla manual en la máquina seleccionada y la registra en CSV."""
        idx = self.machine_sel_cb.current()
        if not (0 <= idx < len(self.system.machines)):
            messagebox.showwarning("Seleccionar", "Seleccioná una máquina válida.")
            return

        m = self.system.machines[idx]

        # Forzamos una variación exagerada
        m.inject_variation(
            dtemp=random.uniform(10, 25),
            dtorque=random.uniform(80, 150),
            dvib=random.uniform(1.0, 3.0),
            dpower=random.uniform(10, 30),
            drpm=random.uniform(200, 600),
            duration=1.0,
            reason="Manual"
        )

        # Registramos en un CSV específico de fallas manuales
        fails_file = os.path.join(OUTPUT_DIR, "fallas_manual.csv")
        df = pd.DataFrame([{
            "timestamp": self.system.time,
            "maquina": m.name,
            "sector": m.sector,
            "descripcion": "Falla simulada manualmente",
            "detalle": "Sobrecarga de temperatura/torque/vibración/potencia/rpm"
        }])
        if not os.path.exists(fails_file):
            df.to_csv(fails_file, index=False)
        else:
            df.to_csv(fails_file, mode="a", header=False, index=False)

        messagebox.showinfo("Falla Simulada",
                            f"Se inyectó una falla manual en {m.name}.\n"
                            f"Se registró en fallas_manual.csv.\n"
                            f"La lógica normal de detección la capturará en el próximo ciclo.")

    def _export_log(self):
        # save csv
        filename = os.path.join(OUTPUT_DIR, f"log_{int(time.time())}.csv")
        df = pd.DataFrame(self.system.log)
        df.to_csv(filename, index=False)
        messagebox.showinfo("Exportar CSV", f"Log exportado:\n{filename}")

    # ---------------- update loop ----------------
    def _update_loop(self):
        while self.running:
            alerts = self.system.step_all(SIM_STEP)
            # update visuals/plots/readings
            self.root_after(self._refresh_all)
            if alerts:
                # pause global simulation and handle alerts on main thread
                self.running = False
                self.system.running = False
                self.paused_by_fault = True
                self.last_alerts = alerts
                self.root_after(lambda: self._handle_alerts_pausing(alerts))
                break
            time.sleep(SIM_STEP)

    def root_after(self, fn, delay=0):
        # helper to schedule in main thread
        try:
            self.root.after(delay, fn)
        except Exception:
            fn()

    def _refresh_all(self):
        # update plots and reading labels
        with self.system.lock:
            for i, m in enumerate(self.system.machines):
                t = list(m.history["time"])
                self.lines[i]["temp"].set_data(t, list(m.history["temperature"]))
                self.lines[i]["torque"].set_data(t, list(m.history["torque"]))
                self.lines[i]["vib"].set_data(t, list(m.history["vibration"]))
                self.lines[i]["power"].set_data(t, list(m.history["power"]))
                self.lines[i]["rpm"].set_data(t, list(m.history["rpm"]))
                # update reading labels (units)
                labels = self.reading_labels[i]
                labels["temp"].config(text=f"Temperatura: {m.temperature:.1f} °C")
                labels["torque"].config(text=f"Torque: {m.torque:.1f} N·m")
                labels["vib"].config(text=f"Vibración: {m.vibration:.2f} mm/s")
                labels["power"].config(text=f"Potencia: {m.power:.1f} %")
                labels["rpm"].config(text=f"RPM: {int(m.rpm)}")
                # update per-tab alerts box
                abox = self.lines[i]["alert_box"]
                abox.config(state="normal"); abox.delete("1.0", tk.END)
                recent = [entry for entry in reversed(self.system.log[-400:])
                          if entry["maquina"]==m.name and entry["alert"]]
                for r in recent[:10]:
                    abox.insert(tk.END, f"t={r['timestamp']:.1f}s | sev={r['severity']} | {r['alert_type']}\n")
                abox.config(state="disabled")
                # autoscale axes
                for ax in self.axes[i]:
                    ax.relim(); ax.autoscale_view()
                try:
                    self.canvases[i].draw_idle()
                except Exception:
                    pass

    def _handle_alerts_pausing(self, alerts):
        # Color de fondo para máquinas con falla
        affected = {a["maquina"] for a in alerts}
        for i, m in enumerate(self.system.machines):
            if m.name in affected:
                bg = "#ffcccc"  # rojo claro
            else:
                bg = "#dddddd"
            for ax in self.axes[i]:
                ax.set_facecolor(bg)
            try:
                self.canvases[i].draw_idle()
            except Exception:
                pass

        # append to global alerts panel
        self.text_alerts.config(state="normal")
        for a in alerts:
            msg = f"[{a['maquina']}] {a['type']}: {a['cause']}\nRec: {a['recommendation']}\n\n"
            self.text_alerts.insert(tk.END, msg)
        self.text_alerts.see(tk.END)
        self.text_alerts.config(state="disabled")

        # popup asking to repair now
        a0 = alerts[0]
        popup_msg = (f"Falla detectada en {a0['maquina']}:\n{a0['type']}\n{a0['cause']}\n"
                     f"Recomendación: {a0['recommendation']}\n\n¿Reparar ahora?")
        repair_now = messagebox.askyesno("FALLA DETECTADA", popup_msg)
        if repair_now:
            # repair affected machines and resume simulation
            for a in alerts:
                idx = next((i for i,m in enumerate(self.system.machines) if m.name==a["maquina"]), None)
                if idx is not None:
                    with self.system.lock:
                        self.system.machines[idx].repair()
                        # log repair event
                        self.system.log.append({
                            "timestamp": self.system.time,
                            "maquina": self.system.machines[idx].name,
                            "temperature": self.system.machines[idx].temperature,
                            "torque": self.system.machines[idx].torque,
                            "vibration": self.system.machines[idx].vibration,
                            "power": self.system.machines[idx].power,
                            "rpm": self.system.machines[idx].rpm,
                            "alert":0,
                            "severity":0,
                            "alert_type":"REPAIR",
                            "recommendation":"Reparado manualmente"
                        })
                    self._restore_visual(idx)
            # resume simulation
            self.paused_by_fault = False
            self.start()
            messagebox.showinfo("Reparado", "Fallas reparadas. Simulación reanudada.")
        else:
            # keep paused; user must press "Reparar falla (máquina seleccionada)" manually later
            messagebox.showinfo("Pausado", "Simulación pausada. Repará la falla cuando estés lista.")

    def _restore_visual(self, idx):
        try:
            base = MACHINE_COLORS[idx]
            self.lines[idx]["temp"].set_color(base); self.lines[idx]["temp"].set_linewidth(1.2)
            self.lines[idx]["torque"].set_color(base); self.lines[idx]["torque"].set_linewidth(1.0)
            self.lines[idx]["vib"].set_color(base); self.lines[idx]["vib"].set_linewidth(1.0)
            self.lines[idx]["power"].set_color(base); self.lines[idx]["power"].set_linewidth(1.0)
            self.lines[idx]["rpm"].set_color(base); self.lines[idx]["rpm"].set_linewidth(1.0)
            self.notebook.tab(idx, text=self.system.machines[idx].name)
        except Exception:
            pass

    def _refresh_ui(self):
        self.machine_sel_cb['values'] = [m.name for m in self.system.machines]
        if len(self.system.machines):
            self.machine_sel_cb.current(0)

# ---------------- main ----------------
def main():
    root = tk.Tk()
    app = AppGUI(root)

    def on_close():
        if messagebox.askyesno("Salir", "¿Exportar log a CSV antes de salir?"):
            try:
                fname = os.path.join(OUTPUT_DIR, f"log_{int(time.time())}.csv")
                pd.DataFrame(app.system.log).to_csv(fname, index=False)
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
