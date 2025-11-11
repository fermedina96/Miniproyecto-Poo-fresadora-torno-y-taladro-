"""
sistema_control_fusionado_final.py
Versión final con:
- 3 máquinas (Fresadora, Torno, Taladro)
- Variables: Temperatura (según temp óptima), Torque, Vibración, Potencia, RPM
- Tolerancias por variable y temp óptima por máquina (editable)
- Parada automática de la máquina al detectar falla
- Gráfica de la variable en falla se pone roja
- Título de pestaña marcado con ⚠ y color rojo
- Popup informativo cuando ocurre la falla
- Botón para reiniciar la máquina
- Simulación realista (OU-like), inyección manual y espontánea, export CSV
Autor: Integración solicitada
"""

import threading
import time
import random
from collections import deque
import math
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
NUM_MAQUINAS = 3
SIM_STEP = 0.2             # segundos por tick (5 Hz)
HISTORY_SECONDS = 120      # histórico visible (segundos)
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
    "Fresadora": {
        "temp_range": (40, 70),    # °C
        "torque_range": (40, 70),
        "vib_range": (0.2, 0.8),
        "power_range": (70, 95),
        "rpm_range": (1500, 2500),
        "opt_temp": 60.0
    },
    "Torno": {
        "temp_range": (35, 60),
        "torque_range": (30, 60),
        "vib_range": (0.1, 0.6),
        "power_range": (60, 90),
        "rpm_range": (1200, 2000),
        "opt_temp": 50.0
    },
    "Taladro": {
        "temp_range": (30, 50),
        "torque_range": (10, 30),
        "vib_range": (0.05, 0.4),
        "power_range": (50, 85),
        "rpm_range": (800, 1500),
        "opt_temp": 40.0
    }
}

# ---------------------------- Clases ----------------------------
class Maquina:
    def __init__(self, id, name, sector="Sector A"):
        self.id = id
        self.name = name
        self.sector = sector

        profile = MACHINE_PROFILE.get(name, MACHINE_PROFILE[MACHINE_NAMES[0]])
        self.opt_temp = profile["opt_temp"]
        self.temp_set = self.opt_temp
        self.torque_set = (profile["torque_range"][0] + profile["torque_range"][1]) / 2.0
        self.vib_set = (profile["vib_range"][0] + profile["vib_range"][1]) / 2.0
        self.power_set = (profile["power_range"][0] + profile["power_range"][1]) / 2.0
        self.rpm_set = (profile["rpm_range"][0] + profile["rpm_range"][1]) / 2.0

        self.temperature = self.temp_set + random.uniform(-1.5, 1.5)
        self.torque = self.torque_set + random.uniform(-5,5)
        self.vibration = self.vib_set + random.uniform(-0.1,0.1)
        self.power = self.power_set + random.uniform(-2,2)
        self.rpm = self.rpm_set + random.uniform(-50,50)

        self._state = {
            "temperature": self.temperature,
            "torque": self.torque,
            "vibration": self.vibration,
            "power": self.power,
            "rpm": self.rpm
        }

        self.control_signal = {"temperature":0.0,"torque":0.0,"vibration":0.0,"power":0.0,"rpm":0.0}
        self.time = 0.0
        self.injected_variation = None
        self.last_alert_time = None

        self.tolerances = {
            "temperature_pct": 10.0,
            "torque_pct": 15.0,
            "vibration_pct": 50.0,
            "power_pct": 20.0,
            "rpm_pct": 15.0
        }

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

        self.active = True   # si False -> máquina detenida por falla
        self.fault_info = None  # { "variable": "temperature", "message": "...", "time": t }

    def _ou_step(self, key, target, tau, sigma, dt):
        x = self._state[key]
        dx = - (x - target) * (dt / tau) + sigma * math.sqrt(dt) * random.gauss(0,1)
        x_new = x + dx
        self._state[key] = x_new
        return x_new

    def step(self, dt):
        # Si máquina está parada por falla, no evolucionan sus variables (se mantienen)
        if not self.active:
            self.time += dt
            # igualmente registramos el tiempo en historial para mostrar línea plana
            self.history["time"].append(self.time)
            self.history["temperature"].append(self.temperature)
            self.history["torque"].append(self.torque)
            self.history["vibration"].append(self.vibration)
            self.history["power"].append(self.power)
            self.history["rpm"].append(self.rpm)
            return

        self.time += dt

        # Control P hacia setpoints
        err_temp = self.temp_set - self.temperature
        self.control_signal["temperature"] = CONTROL_KP["temperature"] * err_temp
        err_torque = self.torque_set - self.torque
        self.control_signal["torque"] = CONTROL_KP["torque"] * err_torque
        err_vib = self.vib_set - self.vibration
        self.control_signal["vibration"] = CONTROL_KP["vibration"] * err_vib
        err_power = self.power_set - self.power
        self.control_signal["power"] = CONTROL_KP["power"] * err_power
        err_rpm = self.rpm_set - self.rpm
        self.control_signal["rpm"] = CONTROL_KP["rpm"] * err_rpm

        profile = MACHINE_PROFILE[self.name]
        t_temp = self._ou_step("temperature", self.temp_set + self.control_signal["temperature"]*0.05, tau=8.0, sigma=0.2, dt=dt)
        t_torque = self._ou_step("torque", self.torque_set + self.control_signal["torque"]*0.02, tau=10.0, sigma=0.8, dt=dt)
        t_vib = self._ou_step("vibration", self.vib_set + self.control_signal["vibration"]*0.01, tau=6.0, sigma=0.02, dt=dt)
        t_power = self._ou_step("power", self.power_set + self.control_signal["power"]*0.02, tau=8.0, sigma=0.6, dt=dt)
        t_rpm = self._ou_step("rpm", self.rpm_set + self.control_signal["rpm"]*0.5, tau=15.0, sigma=30.0, dt=dt)

        # aplicar inyecciones
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

        # actualizar lecturas desde estado suavizado
        self.temperature = float(np.clip(self._state["temperature"], profile["temp_range"][0]-10, profile["temp_range"][1]+30))
        self.torque = float(np.clip(self._state["torque"], profile["torque_range"][0]*0.5, profile["torque_range"][1]*1.5))
        self.vibration = float(np.clip(self._state["vibration"], 0.0, profile["vib_range"][1]*3.0))
        self.power = float(np.clip(self._state["power"], 0.0, 150.0))
        self.rpm = float(np.clip(self._state["rpm"], profile["rpm_range"][0]*0.5, profile["rpm_range"][1]*1.5))

        # registro en historial
        self.history["time"].append(self.time)
        self.history["temperature"].append(self.temperature)
        self.history["torque"].append(self.torque)
        self.history["vibration"].append(self.vibration)
        self.history["power"].append(self.power)
        self.history["rpm"].append(self.rpm)

    def inject_variation(self, dtemp=0.0, dtorque=0.0, dvib=0.0, dpower=0.0, drpm=0.0, duration=0.5,
                         max_dtemp=100.0, max_dtorque=1000.0, max_dvib=10.0, max_dpower=100.0, max_drpm=5000.0, reason="Manual"):
        self.injected_variation = {
            "dtemp": dtemp, "dtorque": dtorque, "dvib": dvib, "dpower": dpower, "drpm": drpm,
            "remaining": duration,
            "max_dtemp": max_dtemp, "max_dtorque": max_dtorque, "max_dvib": max_dvib, "max_dpower": max_dpower, "max_drpm": max_drpm,
            "reason": reason
        }

    def stop_for_fault(self, variable_name, message, timestamp):
        self.active = False
        self.fault_info = {"variable": variable_name, "message": message, "time": timestamp}

    def restart(self):
        self.active = True
        self.fault_info = None
        # opcionalmente limpiar injected_variation
        self.injected_variation = None

# ---------------------------- Sistema ----------------------------
class Sistema:
    def __init__(self, n_machines=NUM_MAQUINAS):
        self.machines = [Maquina(i, MACHINE_NAMES[i], sector=f"Sector {chr(65 + (i//3))}") for i in range(n_machines)]
        self.running = False
        self.lock = threading.Lock()
        self.time = 0.0
        self.log_columns = ["timestamp", "maquina", "sector", "temperature", "torque", "vibration", "power", "rpm", "alert", "severity", "alert_type", "recommendation"]
        self.log = []

    def step_all(self, dt):
        with self.lock:
            for m in self.machines:
                m.step(dt)
            self.time += dt
            alerts = self.detectar_anomalias()
            for m in self.machines:
                last_alert = any([a for a in alerts if a["maquina"] == m.name])
                sev = m.history["severity"][-1] if len(m.history["severity"]) > 0 else 0
                alert_type = next((a["type"] for a in alerts if a["maquina"] == m.name), "")
                rec = next((a["recommendation"] for a in alerts if a["maquina"] == m.name), "")
                self.log.append({
                    "timestamp": self.time,
                    "maquina": m.name,
                    "sector": m.sector,
                    "temperature": m.temperature,
                    "torque": m.torque,
                    "vibration": m.vibration,
                    "power": m.power,
                    "rpm": m.rpm,
                    "alert": int(last_alert),
                    "severity": sev,
                    "alert_type": alert_type,
                    "recommendation": rec
                })
        return alerts

    def detectar_anomalias(self):
        alerts = []
        for m in self.machines:
            # si la máquina ya está parada por falla, no re-evaluamos (podríamos mostrar la misma falla)
            atype = None
            rec = ""
            cause = ""
            sev = 0

            temp_tol_abs = m.opt_temp * (m.tolerances["temperature_pct"] / 100.0)
            temp_high = m.opt_temp + temp_tol_abs
            temp_low = m.opt_temp - temp_tol_abs

            torque_thresh = m.torque_set * (1.0 + m.tolerances["torque_pct"]/100.0)
            vib_thresh = m.vib_set * (1.0 + m.tolerances["vibration_pct"]/100.0)
            power_thresh = m.power_set * (1.0 + m.tolerances["power_pct"]/100.0)
            rpm_thresh = m.rpm_set * (1.0 + m.tolerances["rpm_pct"]/100.0)

            # temperatura relativa
            if m.temperature > temp_high:
                atype = "TEMPERATURE_HIGH"
                cause += f"Temp {m.temperature:.1f}°C > óptima+tol ({temp_high:.1f}°C)."
                rec += "Reducir carga / revisar refrigeración."
                sev = max(sev, 2)
            if m.temperature < temp_low:
                atype = "TEMPERATURE_LOW" if atype is None else atype + "+TEMPERATURE_LOW"
                cause += f" Temp {m.temperature:.1f}°C < óptima-tol ({temp_low:.1f}°C)."
                rec += "Revisar calefacción/condiciones."
                sev = max(sev, 1)

            # otras variables
            if m.torque > torque_thresh:
                atype = "TORQUE_HIGH" if atype is None else atype + "+TORQUE_HIGH"
                cause += f" Torque {m.torque:.1f} > {torque_thresh:.1f}."
                rec += " Revisar alimentación."
                sev = max(sev, 2)
            if m.vibration > vib_thresh:
                atype = "VIBRATION_HIGH" if atype is None else atype + "+VIBRATION_HIGH"
                cause += f" Vib {m.vibration:.2f} > {vib_thresh:.2f}."
                rec += " Revisar rodamientos."
                sev = max(sev, 2)
            if m.power > power_thresh:
                atype = "POWER_HIGH" if atype is None else atype + "+POWER_HIGH"
                cause += f" Pot {m.power:.1f}% > {power_thresh:.1f}%."
                rec += " Reducir demanda."
                sev = max(sev, 1)
            if m.rpm > rpm_thresh:
                atype = "RPM_HIGH" if atype is None else atype + "+RPM_HIGH"
                cause += f" RPM {m.rpm:.0f} > {rpm_thresh:.0f}."
                rec += " Ajustar velocidad."
                sev = max(sev, 1)

            # derivadas (tendencias)
            if len(m.history["time"]) >= 2:
                dt = m.history["time"][-1] - m.history["time"][-2]
                dt = dt if dt > 0 else SIM_STEP
                dtemp = (m.history["temperature"][-1] - m.history["temperature"][-2]) / dt
                dtorque = (m.history["torque"][-1] - m.history["torque"][-2]) / dt
                dvib = (m.history["vibration"][-1] - m.history["vibration"][-2]) / dt
                dpow = (m.history["power"][-1] - m.history["power"][-2]) / dt
                drpm = (m.history["rpm"][-1] - m.history["rpm"][-2]) / dt

                if abs(dtemp) > IA_THRESHOLDS_BASE["dtemp_dt"]:
                    atype = "D_TEMP_HIGH" if atype is None else atype + "+D_TEMP_HIGH"
                    cause += f" ΔT/dt={dtemp:.1f}."
                    rec += " Revisar fricción."
                    sev = max(sev, 1)
                if abs(dtorque) > IA_THRESHOLDS_BASE["dtorque_dt"]:
                    atype = "D_TORQUE_HIGH" if atype is None else atype + "+D_TORQUE_HIGH"
                    cause += f" ΔTorque/dt={dtorque:.1f}."
                    rec += " Revisar alimentación."
                    sev = max(sev, 1)
                if abs(dvib) > IA_THRESHOLDS_BASE["dvib_dt"]:
                    atype = "D_VIB_HIGH" if atype is None else atype + "+D_VIB_HIGH"
                    cause += f" ΔVib/dt={dvib:.2f}."
                    rec += " Revisar rodamientos."
                    sev = max(sev, 1)
                if abs(dpow) > IA_THRESHOLDS_BASE["dpower_dt"]:
                    atype = "D_POWER_HIGH" if atype is None else atype + "+D_POWER_HIGH"
                    cause += f" ΔPower/dt={dpow:.1f}."
                    rec += " Revisar demanda eléctrica."
                    sev = max(sev, 1)
                if abs(drpm) > IA_THRESHOLDS_BASE["drpm_dt"]:
                    atype = "D_RPM_HIGH" if atype is None else atype + "+D_RPM_HIGH"
                    cause += f" ΔRPM/dt={drpm:.0f}."
                    rec += " Revisar transmisión."
                    sev = max(sev, 1)

            # evento espontáneo
            if random.random() < SPONTANEOUS_FAILURE_PROB_PER_SEC * SIM_STEP:
                m.inject_variation(
                    dtemp=random.uniform(6,20),
                    dtorque=random.uniform(30,180),
                    dvib=random.uniform(0.6,3.5),
                    dpower=random.uniform(8,30),
                    drpm=random.uniform(100,600),
                    duration=0.6,
                    max_dtemp=50,
                    max_dtorque=500,
                    max_dvib=5,
                    max_dpower=80,
                    max_drpm=1000,
                    reason="Spontaneous"
                )
                atype = "SPONTANEOUS_VARIATION" if atype is None else atype + "+SPONT"
                cause += " Evento espontáneo."
                rec += " Revisar componente mecánico/eléctrico."
                sev = max(sev, 2)

            m.history["alert"].append(1 if sev > 0 else 0)
            m.history["severity"].append(sev)

            if atype and m.active:
                # Determinar variable principal causante (simple heurística)
                primary_var = None
                if "TEMP" in atype:
                    primary_var = "temperature"
                elif "TORQUE" in atype:
                    primary_var = "torque"
                elif "VIB" in atype:
                    primary_var = "vibration"
                elif "POWER" in atype:
                    primary_var = "power"
                elif "RPM" in atype:
                    primary_var = "rpm"
                else:
                    # fallback por severidad
                    primary_var = "temperature"

                # Parar la máquina y registrar falla
                m.stop_for_fault(primary_var, cause.strip(), self.time)

                alerts.append({
                    "maquina": m.name,
                    "sector": m.sector,
                    "type": atype,
                    "cause": cause.strip(),
                    "recommendation": rec.strip(),
                    "primary_var": primary_var
                })
                m.last_alert_time = self.time
        return alerts

    def inject_manual_variation(self, machine_index, **kwargs):
        if 0 <= machine_index < len(self.machines):
            self.machines[machine_index].inject_variation(**kwargs)

    def export_log_csv(self, filename=None):
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, f"log_{int(time.time())}.csv")
        df = pd.DataFrame(self.log, columns=self.log_columns)
        df.to_csv(filename, index=False)
        return filename

# ---------------------------- GUI ----------------------------
class AppGUI:
    def __init__(self, root, sistema: Sistema):
        self.root = root
        self.system = sistema
        self.running = False
        self.update_thread = None

        root.title("Sistema de Supervisión y Control - Final")
        root.geometry("1360x820")

        self.left_frame = ttk.Frame(root, width=360)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        self.center_frame = ttk.Frame(root)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.right_frame = ttk.Frame(root, width=420)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        self._build_left_controls()
        self._build_center_notebook()
        self._build_right_panel()

        self.status_var = tk.StringVar(value="Estado: detenido")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._refresh_ui_from_machines()

    def _build_left_controls(self):
        ttk.Label(self.left_frame, text="Controles Generales", font=("Arial", 12, "bold")).pack(pady=6)
        ttk.Button(self.left_frame, text="Iniciar simulación", command=self.start).pack(fill=tk.X, pady=4)
        ttk.Button(self.left_frame, text="Detener simulación", command=self.stop).pack(fill=tk.X, pady=4)
        ttk.Button(self.left_frame, text="Exportar log a CSV", command=self._export_log).pack(fill=tk.X, pady=6)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(self.left_frame, text="Seleccionar máquina:", font=("Arial", 10)).pack(pady=(6,2))
        self.machine_sel_cb = ttk.Combobox(self.left_frame, state="readonly", values=[m.name for m in self.system.machines])
        self.machine_sel_cb.current(0)
        self.machine_sel_cb.pack(fill=tk.X, padx=6)
        ttk.Button(self.left_frame, text="Ir a máquina", command=self._go_to_selected_machine).pack(pady=4, fill=tk.X)

        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(self.left_frame, text="Inyectar variación (máquina seleccionada)", font=("Arial", 10, "bold")).pack(pady=4)
        frm_var = ttk.Frame(self.left_frame); frm_var.pack(fill=tk.X, padx=6)
        ttk.Label(frm_var, text="ΔT/s:").grid(row=0, column=0, sticky="w"); self.entry_dtemp = tk.DoubleVar(value=6.0); ttk.Entry(frm_var, textvariable=self.entry_dtemp, width=8).grid(row=0, column=1)
        ttk.Label(frm_var, text="ΔTorque/s:").grid(row=1, column=0, sticky="w"); self.entry_dtorque = tk.DoubleVar(value=40.0); ttk.Entry(frm_var, textvariable=self.entry_dtorque, width=8).grid(row=1, column=1)
        ttk.Label(frm_var, text="ΔVib/s:").grid(row=2, column=0, sticky="w"); self.entry_dvib = tk.DoubleVar(value=1.0); ttk.Entry(frm_var, textvariable=self.entry_dvib, width=8).grid(row=2, column=1)
        ttk.Label(frm_var, text="ΔPower/s:").grid(row=3, column=0, sticky="w"); self.entry_dpower = tk.DoubleVar(value=12.0); ttk.Entry(frm_var, textvariable=self.entry_dpower, width=8).grid(row=3, column=1)
        ttk.Label(frm_var, text="ΔRPM/s:").grid(row=4, column=0, sticky="w"); self.entry_drpm = tk.DoubleVar(value=200.0); ttk.Entry(frm_var, textvariable=self.entry_drpm, width=8).grid(row=4, column=1)
        ttk.Label(frm_var, text="Duración(s):").grid(row=5, column=0, sticky="w"); self.entry_dur = tk.DoubleVar(value=0.6); ttk.Entry(frm_var, textvariable=self.entry_dur, width=8).grid(row=5, column=1)
        ttk.Button(self.left_frame, text="Inyectar variación", command=self._inject_variation_selected).pack(pady=6, fill=tk.X, padx=6)

        ttk.Button(self.left_frame, text="Reiniciar máquina seleccionada", command=self._restart_selected_machine).pack(pady=6, fill=tk.X, padx=6)

    def _build_center_notebook(self):
        self.notebook = ttk.Notebook(self.center_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_frames = []
        self.figures = []
        self.canvases = []
        self.axes = []
        self.lines = []

        for idx, m in enumerate(self.system.machines):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=m.name)
            self.tab_frames.append(tab)

            fig, axs = plt.subplots(5,1,figsize=(7,8), sharex=True)
            fig.tight_layout(pad=3.0)

            l_temp, = axs[0].plot([], [], color=MACHINE_COLORS[idx], label="Temperatura (°C)")
            l_torque, = axs[1].plot([], [], color=MACHINE_COLORS[idx], label="Torque")
            l_vib, = axs[2].plot([], [], color=MACHINE_COLORS[idx], label="Vibración")
            l_power, = axs[3].plot([], [], color=MACHINE_COLORS[idx], label="Potencia (%)")
            l_rpm, = axs[4].plot([], [], color=MACHINE_COLORS[idx], label="RPM")

            axs[0].set_ylabel("Temp (°C)"); axs[1].set_ylabel("Torque"); axs[2].set_ylabel("Vib"); axs[3].set_ylabel("Power (%)"); axs[4].set_ylabel("RPM")
            axs[4].set_xlabel("Tiempo (s)")
            for a in axs:
                a.legend()

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            frm_tol = ttk.LabelFrame(tab, text="Tolerancias y Temp óptima (editar y presionar 'Aplicar')")
            frm_tol.pack(fill=tk.X, padx=6, pady=6)

            ttk.Label(frm_tol, text="Temp opt (°C):").grid(row=0, column=0, sticky="w")
            temp_opt_var = tk.DoubleVar(value=m.opt_temp)
            ttk.Entry(frm_tol, textvariable=temp_opt_var, width=8).grid(row=0, column=1, padx=4)

            ttk.Label(frm_tol, text="Temp % tol:").grid(row=1, column=0, sticky="w")
            temp_tol_var = tk.DoubleVar(value=m.tolerances["temperature_pct"])
            ttk.Entry(frm_tol, textvariable=temp_tol_var, width=8).grid(row=1, column=1, padx=4)

            ttk.Label(frm_tol, text="Torque % tol:").grid(row=2, column=0, sticky="w")
            torque_tol_var = tk.DoubleVar(value=m.tolerances["torque_pct"])
            ttk.Entry(frm_tol, textvariable=torque_tol_var, width=8).grid(row=2, column=1, padx=4)

            ttk.Label(frm_tol, text="Vib % tol:").grid(row=3, column=0, sticky="w")
            vib_tol_var = tk.DoubleVar(value=m.tolerances["vibration_pct"])
            ttk.Entry(frm_tol, textvariable=vib_tol_var, width=8).grid(row=3, column=1, padx=4)

            ttk.Label(frm_tol, text="Power % tol:").grid(row=4, column=0, sticky="w")
            power_tol_var = tk.DoubleVar(value=m.tolerances["power_pct"])
            ttk.Entry(frm_tol, textvariable=power_tol_var, width=8).grid(row=4, column=1, padx=4)

            ttk.Label(frm_tol, text="RPM % tol:").grid(row=5, column=0, sticky="w")
            rpm_tol_var = tk.DoubleVar(value=m.tolerances["rpm_pct"])
            ttk.Entry(frm_tol, textvariable=rpm_tol_var, width=8).grid(row=5, column=1, padx=4)

            def make_apply(machine=m, optvar=temp_opt_var, tt=temp_tol_var, toq=torque_tol_var, vb=vib_tol_var, pw=power_tol_var, rp=rpm_tol_var):
                def apply_tols():
                    machine.opt_temp = float(optvar.get())
                    machine.temp_set = machine.opt_temp
                    machine.tolerances["temperature_pct"] = float(tt.get())
                    machine.tolerances["torque_pct"] = float(toq.get())
                    machine.tolerances["vibration_pct"] = float(vb.get())
                    machine.tolerances["power_pct"] = float(pw.get())
                    machine.tolerances["rpm_pct"] = float(rp.get())
                    messagebox.showinfo("Tolerancias", f"Tolerancias y temp óptima actualizadas para {machine.name}.")
                return apply_tols

            ttk.Button(frm_tol, text="Aplicar", command=make_apply()).grid(row=6, column=0, columnspan=2, pady=6)

            lbl = ttk.Label(tab, text="Alertas (últimas):")
            lbl.pack(anchor="w", padx=6)
            alert_box = scrolledtext.ScrolledText(tab, height=5, state="disabled")
            alert_box.pack(fill=tk.X, padx=6, pady=(0,6))

            self.figures.append(fig)
            self.axes.append(axs)
            self.canvases.append(canvas)
            self.lines.append({
                "temp": l_temp, "torque": l_torque, "vib": l_vib, "power": l_power, "rpm": l_rpm,
                "alert_box": alert_box
            })

    def _build_right_panel(self):
        ttk.Label(self.right_frame, text="Panel de alertas (global)", font=("Arial", 12, "bold")).pack(pady=6)
        self.text_alerts = scrolledtext.ScrolledText(self.right_frame, height=20, state="disabled")
        self.text_alerts.pack(fill=tk.BOTH, expand=True, padx=6)
        ttk.Separator(self.right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Button(self.right_frame, text="Exportar log CSV", command=self._export_log).pack(fill=tk.X, padx=6, pady=4)

    # ----------------- Acciones -----------------
    def start(self):
        if not self.running:
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
            messagebox.showwarning("Seleccionar", "Seleccioná una máquina válida.")
            return
        dtemp = float(self.entry_dtemp.get())
        dtorque = float(self.entry_dtorque.get())
        dvib = float(self.entry_dvib.get())
        dpower = float(self.entry_dpower.get())
        drpm = float(self.entry_drpm.get())
        dur = float(self.entry_dur.get())
        self.system.inject_manual_variation(idx,
                                            dtemp=dtemp, dtorque=dtorque, dvib=dvib, dpower=dpower, drpm=drpm,
                                            duration=dur,
                                            max_dtemp=abs(dtemp)*2+1, max_dtorque=abs(dtorque)*2+5,
                                            max_dvib=max(1.0,abs(dvib)*2), max_dpower=abs(dpower)*2+5, max_drpm=abs(drpm)*2+50,
                                            reason="Manual")
        messagebox.showinfo("Inyección", f"Variación inyectada en {self.system.machines[idx].name}.")

    def _restart_selected_machine(self):
        idx = self.machine_sel_cb.current()
        if not (0 <= idx < len(self.system.machines)):
            messagebox.showwarning("Seleccionar", "Seleccioná una máquina válida.")
            return
        m = self.system.machines[idx]
        if m.active:
            messagebox.showinfo("Reiniciar", f"{m.name} ya está ejecutando.")
            return
        m.restart()
        # restaurar colores de líneas y pestaña
        self._restore_tab_visual(idx)
        messagebox.showinfo("Reiniciar", f"{m.name} reiniciada.")

    def _export_log(self):
        filename = self.system.export_log_csv()
        messagebox.showinfo("Exportar CSV", f"Log exportado en:\n{filename}")

    # ----------------- Loop de actualización -----------------
    def _update_loop(self):
        while self.running:
            alerts = self.system.step_all(SIM_STEP)
            # manejar alertas: parar máquina ya se hace en detectar_anomalias()
            self._refresh_plots_all()
            self._handle_alerts_and_visuals(alerts)
            time.sleep(SIM_STEP)

    def _refresh_plots_all(self):
        with self.system.lock:
            for i, m in enumerate(self.system.machines):
                t = list(m.history["time"])
                temp = list(m.history["temperature"])
                torque = list(m.history["torque"])
                vib = list(m.history["vibration"])
                power = list(m.history["power"])
                rpm = list(m.history["rpm"])
                lines = self.lines[i]
                try:
                    lines["temp"].set_data(t, temp)
                    lines["torque"].set_data(t, torque)
                    lines["vib"].set_data(t, vib)
                    lines["power"].set_data(t, power)
                    lines["rpm"].set_data(t, rpm)
                except Exception:
                    continue
                axs = self.axes[i]
                for ax in axs:
                    ax.relim()
                    ax.autoscale_view()
                try:
                    self.canvases[i].draw_idle()
                except Exception:
                    pass

    def _handle_alerts_and_visuals(self, alerts):
        # global alerts panel (append)
        if alerts:
            self.text_alerts.config(state="normal")
            for a in alerts:
                msg = f"[{a['maquina']} | {a['sector']}] {a['type']}: {a['cause']}\nRec: {a['recommendation']}\n\n"
                self.text_alerts.insert(tk.END, msg)
                # popup for each alert (immediate notification)
                primary_var = a.get("primary_var", "")
                popup_msg = f"Falla detectada en {a['maquina']}:\n{a['type']}\n{a['cause']}\nRecomendación: {a['recommendation']}"
                # show popup (non-blocking)
                try:
                    messagebox.showerror("FALLA DETECTADA", popup_msg)
                except Exception:
                    pass
            self.text_alerts.see(tk.END)
            self.text_alerts.config(state="disabled")

        # per-tab update: color lines red for fault variable, mark tab label with ⚠ and red background (if possible)
        with self.system.lock:
            for i, m in enumerate(self.system.machines):
                box = self.lines[i]["alert_box"]
                recent = [entry for entry in reversed(self.system.log[-400:]) if entry["maquina"] == m.name and entry["alert"]]
                box.config(state="normal")
                box.delete("1.0", tk.END)
                for r in recent[:10]:
                    box.insert(tk.END, f"t={r['timestamp']:.1f}s | sev={r['severity']} | {r['alert_type']}\n")
                box.see(tk.END)
                box.config(state="disabled")

                # Visual: si máquina parada por falla -> marcar la línea correspondiente roja y tab con ⚠ y rojo
                if m.fault_info is not None:
                    var = m.fault_info.get("variable", "")
                    # cambiar color de la linea correspondiente a rojo; otras líneas a color tenue
                    for key, line in [("temperature", self.lines[i]["temp"]),
                                      ("torque", self.lines[i]["torque"]),
                                      ("vibration", self.lines[i]["vib"]),
                                      ("power", self.lines[i]["power"]),
                                      ("rpm", self.lines[i]["rpm"])]:
                        if key == var:
                            try:
                                line.set_color("red")
                                line.set_linewidth(2.2)
                            except Exception:
                                pass
                        else:
                            try:
                                # atenuar otras líneas
                                base_color = MACHINE_COLORS[i]
                                line.set_color(base_color)
                                line.set_linewidth(0.9)
                            except Exception:
                                pass
                    # marcar pestaña con ⚠ y texto rojo (tkinter tabs no permiten color del texto directamente cross-platform,
                    # así que usamos el prefijo y opcionalmente cambiamos el tab's background via style if available)
                    try:
                        self.notebook.tab(i, text="⚠ " + m.name)
                    except Exception:
                        pass
                else:
                    # restaurar apariencia normal
                    base_color = MACHINE_COLORS[i]
                    try:
                        self.lines[i]["temp"].set_color(base_color)
                        self.lines[i]["temp"].set_linewidth(1.2)
                        self.lines[i]["torque"].set_color(base_color)
                        self.lines[i]["torque"].set_linewidth(1.0)
                        self.lines[i]["vib"].set_color(base_color)
                        self.lines[i]["vib"].set_linewidth(1.0)
                        self.lines[i]["power"].set_color(base_color)
                        self.lines[i]["power"].set_linewidth(1.0)
                        self.lines[i]["rpm"].set_color(base_color)
                        self.lines[i]["rpm"].set_linewidth(1.0)
                        self.notebook.tab(i, text=m.name)
                    except Exception:
                        pass

    def _restore_tab_visual(self, idx):
        # helper para restaurar tras reinicio
        base_color = MACHINE_COLORS[idx]
        try:
            self.lines[idx]["temp"].set_color(base_color); self.lines[idx]["temp"].set_linewidth(1.2)
            self.lines[idx]["torque"].set_color(base_color); self.lines[idx]["torque"].set_linewidth(1.0)
            self.lines[idx]["vib"].set_color(base_color); self.lines[idx]["vib"].set_linewidth(1.0)
            self.lines[idx]["power"].set_color(base_color); self.lines[idx]["power"].set_linewidth(1.0)
            self.lines[idx]["rpm"].set_color(base_color); self.lines[idx]["rpm"].set_linewidth(1.0)
            self.notebook.tab(idx, text=self.system.machines[idx].name)
        except Exception:
            pass

    def _refresh_ui_from_machines(self):
        self.machine_sel_cb['values'] = [m.name for m in self.system.machines]
        if len(self.system.machines) > 0:
            self.machine_sel_cb.current(0)

# ---------------------------- Main ----------------------------
def main():
    root = tk.Tk()
    sistema = Sistema()
    app = AppGUI(root, sistema)

    def on_close():
        if messagebox.askyesno("Salir", "¿Desea salir y exportar el log actual a CSV?"):
            try:
                sistema.export_log_csv()
            except Exception:
                pass
            root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
