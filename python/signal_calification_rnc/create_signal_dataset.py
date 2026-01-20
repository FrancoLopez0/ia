import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Configuración de Parámetros ---
SAMPLE_RATE = 44100  # Hz
DURATION = 0.5       # Segundos por muestra
FREQ_MIN, FREQ_MAX = 100, 5000  # Rango de frecuencias fundamentales
NOISE_LEVEL = 0.15   # Amplitud del ruido blanco
N_SAMPLES_PER_CLASS = 500
OUTPUT_DIR = "dataset"

# Tipos de señales
CLASSES = ['sine', 'square', 'sawtooth', 'triangle']

def generate_signal(signal_type, freq, duration, sr, noise_level):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if signal_type == 'sine':
        x = np.sin(2 * np.pi * freq * t)
    elif signal_type == 'square':
        x = signal.square(2 * np.pi * freq * t)
    elif signal_type == 'sawtooth':
        x = signal.sawtooth(2 * np.pi * freq * t)
    elif signal_type == 'triangle':
        x = signal.sawtooth(2 * np.pi * freq * t, width=0.5)
    
    # Agregar ruido Gaussiano (Blanco)
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

# def save_spectrogram(sig, sr, folder, filename):
#     # Calcular el espectrograma (STFT)
#     frequencies, times, Sxx = signal.spectrogram(sig, sr, nperseg=512, noverlap=256)
    
#     plt.figure(figsize=(5, 5))
#     # Usar escala logarítmica (dB) para resaltar armónicos
#     plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
#     plt.axis('off')  # Quitar ejes para el dataset de entrenamiento
    
#     path = os.path.join(folder, filename)
#     plt.savefig(path, bbox_inches='tight', pad_inches=0)
#     plt.close()

def save_spectrogram(sig, sr, folder, filename):
    # Calcular el espectrograma (STFT)
    frequencies, times, Sxx = signal.spectrogram(sig, sr, nperseg=512, noverlap=256)
    
    # Configurar la figura
    # Usamos un tamaño fijo para asegurar consistencia en las entradas de la red
    fig = plt.figure(figsize=(4, 4), dpi=100) 
    ax = fig.add_subplot(111)
    
    # --- CAMBIO CLAVE AQUÍ ---
    # Usamos escala logarítmica (dB) para resaltar armónicos.
    # cmap='gray_r': Escala de grises invertida.
    # El fondo (baja energía) será blanco y las señales fuertes serán negras.
    # A menudo se prefiere 'gray_r' en publicaciones científicas, 
    # pero 'gray' (fondo negro) también funciona perfectamente.
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    ax.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='gray_r')
    
    # Eliminar ejes, etiquetas y marcos para que solo quede la data pura
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    path = os.path.join(folder, filename)
    # Guardar asegurando que no queden bordes blancos extra
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# --- Proceso Principal ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for cls in CLASSES:
        print(f"Generando clase: {cls}...")
        class_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(class_path, exist_ok=True)
        
        for i in range(N_SAMPLES_PER_CLASS):
            # Frecuencia aleatoria para variabilidad
            freq = np.random.uniform(FREQ_MIN, FREQ_MAX)
            sig = generate_signal(cls, freq, DURATION, SAMPLE_RATE, NOISE_LEVEL)
            
            save_spectrogram(sig, SAMPLE_RATE, class_path, f"{cls}_{i}.png")

    print(f"Dataset completado en la carpeta: '{OUTPUT_DIR}'")