import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, PhotoImage, Label
from operations.neuron import open_csv

class LearningModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelo de Aprendizaje")
        self.root.geometry("1200x700")
        self.output_folder = "output_graphs"
        os.makedirs(self.output_folder, exist_ok=True)
        self.init_vars()
        self.create_widgets()

    def init_vars(self):
        self.csv_path_var = tk.StringVar()
        self.eta_var = tk.StringVar(value="0.01")
        self.epochs_var = tk.StringVar(value="1000")
        self.tolerance_var = tk.StringVar(value="0.001")

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.add_csv_input(frame)
        self.add_params_input(frame)
        self.add_start_button(frame)
        self.add_results_text(frame)
        self.add_graph_labels(frame)
        self.add_weight_table(frame) 

    def add_csv_input(self, frame):
        ttk.Label(frame, text="Ruta del archivo CSV:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.csv_path_var, width=50).grid(row=0, column=1)
        ttk.Button(frame, text="Cargar CSV", command=self.load_csv).grid(row=0, column=2)

    def add_params_input(self, frame):
        ttk.Label(frame, text="Tasa de aprendizaje (ETA):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.eta_var).grid(row=1, column=1)
        ttk.Label(frame, text="Cantidad de épocas:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.epochs_var).grid(row=2, column=1)
        ttk.Label(frame, text="Tolerancia:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.tolerance_var).grid(row=3, column=1)

    def add_start_button(self, frame):
        ttk.Button(frame, text="Iniciar", command=self.start_process).grid(row=4, column=0, columnspan=3, pady=10)

    def add_results_text(self, frame):
        ttk.Label(frame, text="Resultados:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.results_text = tk.Text(frame, height=10, width=70)
        self.results_text.grid(row=6, column=0, columnspan=3)
   
    def add_graph_labels(self, frame):
        self.img_label1 = Label(frame)
        self.img_label1.grid(row=7, column=0, padx=10, pady=10, sticky=tk.NSEW)
        self.img_label2 = Label(frame)
        self.img_label2.grid(row=7, column=1, padx=10, pady=10, sticky=tk.NSEW)

    def add_weight_table(self, frame):
        self.table_frame = ttk.Frame(frame)
        self.table_frame.grid(row=8, column=0, columnspan=3, pady=10)

        self.table = ttk.Treeview(self.table_frame, columns=("Iteración", "Peso 1"), show="headings")
        self.table.heading("Iteración", text="Iteración")
        self.table.heading("Peso 1", text="Peso 1")
      
        self.table.grid(row=0, column=0, sticky="nsew")

    def load_csv(self):
        filepath = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=[("Archivos CSV", "*.csv")])
        if filepath:
            self.csv_path_var.set(filepath)

    def plot_error_evolution(self, e):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(e, marker='o', linestyle='-', label='Error', color='green')
        ax.set_xlabel('Épocas'); ax.set_ylabel('|E|'); ax.set_title('Evolución de la norma del error'); ax.legend()
        filename = os.path.join(self.output_folder, 'norm_e_evolution.png')
        fig.savefig(filename); plt.close(fig)
        self.img_label1.image = PhotoImage(file=filename); self.img_label1.config(image=self.img_label1.image)

    def plot_weight_evolution(self, w):
        w = np.squeeze(np.array(w))
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.spines['left'].set_position('zero'); ax2.spines['left'].set_color('gray')
        ax2.spines['bottom'].set_position('zero'); ax2.spines['bottom'].set_color('gray')
        marks = ['s', 'o', '^']
        for i in range(w.shape[1]): ax2.plot(range(w.shape[0]), w[:, i], label=f'Peso {i}', marker=random.choice(marks), linestyle='-')
        ax2.set_xlim((0 - 1), (len(w) + 1)); ax2.set_xlabel('Épocas'); ax2.set_ylabel('Pesos (w)'); ax2.set_title('Evolución de los pesos'); ax2.legend()
        filename2 = os.path.join(self.output_folder, 'weight_evolution.png')
        fig2.savefig(filename2); plt.close(fig2)
        self.img_label2.image = PhotoImage(file=filename2); self.img_label2.config(image=self.img_label2.image)

    def start_process(self):
        csv_path = self.csv_path_var.get(); eta = float(self.eta_var.get()); epochs = int(self.epochs_var.get()); tolerance = float(self.tolerance_var.get())
        if not csv_path: messagebox.showwarning("Advertencia", "Por favor selecciona un archivo CSV."); return
        try:
            w_by_iterations, norm_e_by_iterations = open_csv(csv_path, tolerance, eta, epochs)
            if w_by_iterations:
                self.display_results(w_by_iterations[0], w_by_iterations[-1], eta, len(w_by_iterations), tolerance, w_by_iterations)
                self.plot_error_evolution(norm_e_by_iterations)
                self.plot_weight_evolution(w_by_iterations)
        except Exception as e: messagebox.showerror("Error", f"Error al iniciar el proceso: {e}")

    def display_results(self, initial_weights, final_weights, eta, epochs, tolerance, w_by_iterations):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Pesos iniciales: {initial_weights}\nPesos finales: {final_weights}\nETA: {eta}\nÉpocas: {epochs}\nTolerancia: {tolerance}\n------------------------------------------\n")
        for row in self.table.get_children():
            self.table.delete(row)
        for i, weights in enumerate(w_by_iterations):
            self.table.insert("", "end", values=(i, *weights))

def create_gui():
    root = tk.Tk()
    app = LearningModelGUI(root)
    root.mainloop()
