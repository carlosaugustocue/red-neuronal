import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
import time

class GeneradorDigitos:
    """Genera dígitos simples para reconocimiento"""
    
    @staticmethod
    def generar_digito(numero, tamano=28, ruido=0.1):
        """Genera una imagen simple de un dígito"""
        img = np.zeros((tamano, tamano))
        centro_x, centro_y = tamano // 2, tamano // 2
        
        if numero == 0:
            # Círculo
            for i in range(tamano):
                for j in range(tamano):
                    dist = np.sqrt((i - centro_x)**2 + (j - centro_y)**2)
                    if 8 <= dist <= 12:
                        img[i, j] = 1
        
        elif numero == 1:
            # Línea vertical
            for i in range(5, tamano-5):
                img[i, centro_y] = 1
                img[i, centro_y-1] = 0.8
                img[i, centro_y+1] = 0.8
        
        elif numero == 2:
            # Forma de 2
            # Parte superior
            for j in range(8, 20):
                img[8, j] = 1
            # Diagonal
            for i in range(8, 20):
                img[i, 20-i+8] = 1
            # Parte inferior
            for j in range(8, 20):
                img[19, j] = 1
        
        elif numero == 3:
            # Forma de 3
            for j in range(8, 20):
                img[8, j] = 1   # Superior
                img[14, j] = 1  # Medio
                img[19, j] = 1  # Inferior
            for i in range(8, 20):
                img[i, 19] = 1  # Derecha
        
        elif numero == 4:
            # Forma de 4
            for i in range(6, 15):
                img[i, 10] = 1  # Vertical izquierda
            for j in range(10, 20):
                img[14, j] = 1  # Horizontal
            for i in range(6, 22):
                img[i, 18] = 1  # Vertical derecha
        
        elif numero == 5:
            # Forma de 5
            for j in range(8, 20):
                img[8, j] = 1   # Superior
                img[14, j] = 1  # Medio
                img[20, j] = 1  # Inferior
            for i in range(8, 14):
                img[i, 8] = 1   # Izquierda superior
            for i in range(14, 20):
                img[i, 19] = 1  # Derecha inferior
        
        elif numero == 6:
            # Forma de 6 (como en la imagen)
            for i in range(8, 20):
                img[i, 8] = 1   # Izquierda
            for j in range(8, 20):
                img[8, j] = 1   # Superior
                img[14, j] = 1  # Medio
                img[19, j] = 1  # Inferior
            for i in range(14, 20):
                img[i, 19] = 1  # Derecha inferior
        
        elif numero == 7:
            # Forma de 7
            for j in range(8, 20):
                img[8, j] = 1   # Superior
            for i in range(8, 20):
                img[i, 19] = 1  # Diagonal derecha
        
        elif numero == 8:
            # Forma de 8
            for i in range(8, 20):
                img[i, 8] = 1   # Izquierda
                img[i, 19] = 1  # Derecha
            for j in range(8, 20):
                img[8, j] = 1   # Superior
                img[14, j] = 1  # Medio
                img[19, j] = 1  # Inferior
        
        elif numero == 9:
            # Forma de 9
            for i in range(8, 15):
                img[i, 8] = 1   # Izquierda superior
            for i in range(8, 20):
                img[i, 19] = 1  # Derecha
            for j in range(8, 20):
                img[8, j] = 1   # Superior
                img[14, j] = 1  # Medio
        
        # Añadir ruido
        ruido_mask = np.random.random((tamano, tamano)) < ruido
        img[ruido_mask] += np.random.random(np.sum(ruido_mask)) * 0.3
        
        # Suavizar
        img = np.clip(img, 0, 1)
        
        return img

class RedReconocedorDigitos:
    def __init__(self, arquitectura=[784, 128, 64, 10]):
        """Red neuronal para reconocimiento de dígitos"""
        self.arquitectura = arquitectura
        self.capas = len(arquitectura)
        
        # Inicializar pesos y sesgos
        self.pesos = []
        self.sesgos = []
        
        for i in range(self.capas - 1):
            # Inicialización He para ReLU
            peso = np.random.randn(arquitectura[i + 1], arquitectura[i]) * np.sqrt(2.0 / arquitectura[i])
            sesgo = np.zeros((arquitectura[i + 1], 1))
            
            self.pesos.append(peso)
            self.sesgos.append(sesgo)
        
        # Para tracking
        self.activaciones = []
        self.z_valores = []
    
    def relu(self, z):
        """Función ReLU"""
        return np.maximum(0, z)
    
    def softmax(self, z):
        """Función Softmax para la salida"""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward_pass(self, X):
        """Forward pass con tracking de activaciones"""
        # Resetear tracking
        self.activaciones = []
        self.z_valores = []
        
        # Normalizar entrada
        activacion = X.flatten().reshape(-1, 1) / 255.0 if X.max() > 1 else X.flatten().reshape(-1, 1)
        self.activaciones.append(activacion.copy())
        
        # Forward pass por cada capa
        for i in range(self.capas - 1):
            z = np.dot(self.pesos[i], activacion) + self.sesgos[i]
            self.z_valores.append(z.copy())
            
            if i == self.capas - 2:  # Última capa
                activacion = self.softmax(z)
            else:
                activacion = self.relu(z)
            
            self.activaciones.append(activacion.copy())
        
        return self.activaciones[-1]
    
    def entrenar_basico(self, X_train, y_train, epocas=50, lr=0.001):
        """Entrenamiento básico (simplificado para demo)"""
        for epoca in range(epocas):
            for i in range(len(X_train)):
                x = X_train[i].flatten().reshape(-1, 1)
                y_true = np.zeros((10, 1))
                y_true[y_train[i]] = 1
                
                # Forward pass
                y_pred = self.forward_pass(x)
                
                # Backward pass simplificado
                error = y_pred - y_true
                
                # Actualizar solo la última capa (simplificado)
                self.pesos[-1] -= lr * np.dot(error, self.activaciones[-2].T)
                self.sesgos[-1] -= lr * error

class VisualizadorReconocedor:
    def __init__(self, root):
        self.root = root
        self.root.title("🔢 Reconocedor de Dígitos - Visualización de Activaciones")
        self.root.geometry("1600x900")
        self.root.configure(bg='black')
        
        # Crear red neuronal
        self.red = RedReconocedorDigitos([784, 128, 64, 10])
        
        # Generar datos de entrenamiento simples
        self.generar_datos_entrenamiento()
        
        # Entrenar la red básicamente
        self.red.entrenar_basico(self.X_train, self.y_train, epocas=20)
        
        # Variables
        self.imagen_actual = None
        self.digito_actual = 6  # Empezar con 6 como en la imagen
        self.animando = False
        
        # Crear interfaz
        self.crear_interfaz()
        
        # Mostrar dígito inicial
        self.mostrar_digito(self.digito_actual)
    
    def generar_datos_entrenamiento(self):
        """Genera datos de entrenamiento simples"""
        self.X_train = []
        self.y_train = []
        
        # Generar 5 ejemplos de cada dígito
        for digito in range(10):
            for _ in range(5):
                img = GeneradorDigitos.generar_digito(digito, ruido=0.2)
                self.X_train.append(img)
                self.y_train.append(digito)
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
    
    def crear_interfaz(self):
        """Crea la interfaz principal"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de control superior
        self.crear_panel_control(main_frame)
        
        # Área de visualización
        self.crear_area_visualizacion(main_frame)
    
    def crear_panel_control(self, parent):
        """Crea el panel de controles"""
        control_frame = tk.Frame(parent, bg='#1a1a1a', relief='raised', bd=2)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Título
        titulo = tk.Label(control_frame, text="🔢 RECONOCEDOR DE DÍGITOS", 
                         font=('Arial', 16, 'bold'), bg='#1a1a1a', fg='white')
        titulo.pack(pady=10)
        
        # Controles de dígitos
        botones_frame = tk.Frame(control_frame, bg='#1a1a1a')
        botones_frame.pack(pady=10)
        
        tk.Label(botones_frame, text="Seleccionar Dígito:", 
                font=('Arial', 12), bg='#1a1a1a', fg='white').pack(side=tk.LEFT, padx=10)
        
        # Botones para cada dígito
        for i in range(10):
            btn = tk.Button(botones_frame, text=str(i), width=4, height=2,
                           font=('Arial', 12, 'bold'), 
                           command=lambda d=i: self.mostrar_digito(d),
                           bg='#2d2d2d', fg='white', relief='raised')
            btn.pack(side=tk.LEFT, padx=2)
        
        # Controles adicionales
        controles_frame = tk.Frame(control_frame, bg='#1a1a1a')
        controles_frame.pack(pady=10)
        
        tk.Button(controles_frame, text="🎲 Dígito Aleatorio", 
                 command=self.digito_aleatorio, font=('Arial', 10),
                 bg='#0066cc', fg='white').pack(side=tk.LEFT, padx=10)
        
        tk.Button(controles_frame, text="🔄 Regenerar", 
                 command=self.regenerar_digito, font=('Arial', 10),
                 bg='#006600', fg='white').pack(side=tk.LEFT, padx=10)
        
        tk.Button(controles_frame, text="🎯 Re-entrenar Red", 
                 command=self.reentrenar_red, font=('Arial', 10),
                 bg='#cc6600', fg='white').pack(side=tk.LEFT, padx=10)
    
    def crear_area_visualizacion(self, parent):
        """Crea el área de visualización principal"""
        # Frame para matplotlib
        viz_frame = tk.Frame(parent, bg='black')
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Crear figura matplotlib
        self.fig = Figure(figsize=(16, 8), facecolor='black')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def mostrar_digito(self, digito):
        """Muestra un dígito específico"""
        self.digito_actual = digito
        self.imagen_actual = GeneradorDigitos.generar_digito(digito, ruido=0.1)
        
        # Procesar con la red neuronal
        predicciones = self.red.forward_pass(self.imagen_actual)
        
        # Actualizar visualización
        self.actualizar_visualizacion()
    
    def digito_aleatorio(self):
        """Muestra un dígito aleatorio"""
        digito = np.random.randint(0, 10)
        self.mostrar_digito(digito)
    
    def regenerar_digito(self):
        """Regenera el dígito actual"""
        self.mostrar_digito(self.digito_actual)
    
    def reentrenar_red(self):
        """Re-entrena la red neuronal"""
        messagebox.showinfo("Entrenando", "Re-entrenando la red neuronal...")
        self.generar_datos_entrenamiento()
        self.red.entrenar_basico(self.X_train, self.y_train, epocas=30)
        self.mostrar_digito(self.digito_actual)
        messagebox.showinfo("Completado", "¡Red re-entrenada!")
    
    def actualizar_visualizacion(self):
        """Actualiza toda la visualización"""
        self.fig.clear()
        
        # Configurar el layout
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 2, 1, 1])
        
        # 1. Imagen de entrada (arriba izquierda)
        ax_imagen = self.fig.add_subplot(gs[0, 0])
        self.dibujar_imagen_entrada(ax_imagen)
        
        # 2. Red neuronal (centro, ocupando 2 filas)
        ax_red = self.fig.add_subplot(gs[:, 1:3])
        self.dibujar_red_neuronal(ax_red)
        
        # 3. Predicciones (arriba derecha)
        ax_pred = self.fig.add_subplot(gs[0, 3])
        self.dibujar_predicciones(ax_pred)
        
        # 4. Información adicional (abajo derecha)
        ax_info = self.fig.add_subplot(gs[1, 3])
        self.dibujar_informacion(ax_info)
        
        # 5. Activaciones de entrada (abajo izquierda)
        ax_act = self.fig.add_subplot(gs[1, 0])
        self.dibujar_activaciones_entrada(ax_act)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def dibujar_imagen_entrada(self, ax):
        """Dibuja la imagen de entrada"""
        ax.clear()
        ax.set_title("ENTRADA (28×28)", color='white', fontsize=12, fontweight='bold')
        
        if self.imagen_actual is not None:
            im = ax.imshow(self.imagen_actual, cmap='gray', vmin=0, vmax=1)
            
            # Agregar un borde verde brillante
            for spine in ax.spines.values():
                spine.set_color('#00ff00')
                spine.set_linewidth(3)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
    
    def dibujar_red_neuronal(self, ax):
        """Dibuja la red neuronal con activaciones"""
        ax.clear()
        ax.set_title("RED NEURONAL - FLUJO DE ACTIVACIONES", 
                    color='white', fontsize=14, fontweight='bold')
        ax.set_facecolor('black')
        
        if not self.red.activaciones:
            return
        
        # Configuración de la red
        capas = self.red.arquitectura
        max_mostrar = [20, 15, 10, 10]  # Máximo de neuronas a mostrar por capa
        
        # Posiciones de las capas
        x_positions = [0, 3, 6, 9]
        layer_heights = []
        
        # Calcular alturas para centrar las capas
        for i, (capa_size, max_show) in enumerate(zip(capas, max_mostrar)):
            show_neurons = min(capa_size, max_show)
            layer_heights.append(show_neurons)
        
        max_height = max(layer_heights)
        
        # Dibujar neuronas y conexiones
        neuron_positions = {}
        
        for layer_idx, (capa_size, max_show) in enumerate(zip(capas, max_mostrar)):
            show_neurons = min(capa_size, max_show)
            x = x_positions[layer_idx]
            
            # Calcular posiciones Y para centrar
            start_y = (max_height - show_neurons) / 2
            
            # Obtener activaciones de esta capa
            if layer_idx < len(self.red.activaciones):
                activaciones_capa = self.red.activaciones[layer_idx]
                if len(activaciones_capa) > max_show:
                    # Tomar una muestra representativa
                    indices = np.linspace(0, len(activaciones_capa)-1, max_show, dtype=int)
                    activaciones_muestra = activaciones_capa[indices, 0]
                else:
                    activaciones_muestra = activaciones_capa[:, 0]
            else:
                activaciones_muestra = np.zeros(show_neurons)
            
            # Dibujar neuronas
            for neuron_idx in range(show_neurons):
                y = start_y + neuron_idx
                
                # Color basado en activación
                activacion = activaciones_muestra[neuron_idx] if neuron_idx < len(activaciones_muestra) else 0
                color_intensity = np.clip(activacion, 0, 1)
                
                if layer_idx == 0:  # Capa de entrada
                    color = plt.cm.gray(color_intensity)
                elif layer_idx == len(capas) - 1:  # Capa de salida
                    color = plt.cm.plasma(color_intensity)
                else:  # Capas ocultas
                    color = plt.cm.viridis(color_intensity)
                
                # Tamaño basado en activación
                size = 80 + color_intensity * 120
                
                circle = ax.scatter(x, y, s=size, c=[color], 
                                  edgecolors='white', linewidths=1, alpha=0.8)
                
                neuron_positions[(layer_idx, neuron_idx)] = (x, y)
                
                # Mostrar valor de activación en neuronas de salida
                if layer_idx == len(capas) - 1:
                    ax.text(x + 0.3, y, f'{activacion:.2f}', 
                           color='white', fontsize=8, va='center')
        
        # Dibujar conexiones (solo algunas para no saturar)
        for layer_idx in range(len(capas) - 1):
            current_neurons = min(capas[layer_idx], max_mostrar[layer_idx])
            next_neurons = min(capas[layer_idx + 1], max_mostrar[layer_idx + 1])
            
            # Dibujar solo algunas conexiones representativas
            for i in range(0, current_neurons, max(1, current_neurons // 5)):
                for j in range(0, next_neurons, max(1, next_neurons // 3)):
                    if (layer_idx, i) in neuron_positions and (layer_idx + 1, j) in neuron_positions:
                        x1, y1 = neuron_positions[(layer_idx, i)]
                        x2, y2 = neuron_positions[(layer_idx + 1, j)]
                        
                        # Intensidad basada en peso
                        if layer_idx < len(self.red.pesos):
                            peso_idx_i = min(i, self.red.pesos[layer_idx].shape[1] - 1)
                            peso_idx_j = min(j, self.red.pesos[layer_idx].shape[0] - 1)
                            peso = self.red.pesos[layer_idx][peso_idx_j, peso_idx_i]
                            alpha = min(abs(peso) * 2, 0.6)
                            color = '#ff4444' if peso < 0 else '#4444ff'
                        else:
                            alpha = 0.2
                            color = '#888888'
                        
                        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)
        
        # Etiquetas de capas
        layer_names = ['ENTRADA\n(784)', 'OCULTA 1\n(128)', 'OCULTA 2\n(64)', 'SALIDA\n(10)']
        for i, (x, name) in enumerate(zip(x_positions, layer_names)):
            ax.text(x, -1.5, name, ha='center', va='top', color='white', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-1, 10)
        ax.set_ylim(-2, max_height)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def dibujar_predicciones(self, ax):
        """Dibuja las predicciones de salida"""
        ax.clear()
        ax.set_title("PREDICCIONES", color='white', fontsize=12, fontweight='bold')
        ax.set_facecolor('black')
        
        if not self.red.activaciones:
            return
        
        predicciones = self.red.activaciones[-1][:, 0]
        digitos = list(range(10))
        
        # Crear barras
        bars = ax.barh(digitos, predicciones, color='skyblue', alpha=0.7)
        
        # Destacar la predicción más alta
        max_idx = np.argmax(predicciones)
        bars[max_idx].set_color('#ffff00')
        bars[max_idx].set_alpha(1.0)
        
        # Destacar el dígito correcto
        bars[self.digito_actual].set_edgecolor('#00ff00')
        bars[self.digito_actual].set_linewidth(3)
        
        # Añadir valores
        for i, (digito, pred) in enumerate(zip(digitos, predicciones)):
            ax.text(pred + 0.01, i, f'{pred:.3f}', 
                   va='center', color='white', fontsize=8)
        
        ax.set_xlabel('Probabilidad', color='white')
        ax.set_ylabel('Dígito', color='white')
        ax.set_xlim(0, 1)
        ax.tick_params(colors='white')
        
        # Añadir información de precisión
        if max_idx == self.digito_actual:
            ax.text(0.5, -1, "✅ CORRECTO", ha='center', transform=ax.transAxes,
                   color='#00ff00', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, -1, f"❌ Predijo: {max_idx}", ha='center', transform=ax.transAxes,
                   color='#ff0000', fontsize=10, fontweight='bold')
    
    def dibujar_informacion(self, ax):
        """Dibuja información adicional"""
        ax.clear()
        ax.set_title("INFORMACIÓN", color='white', fontsize=12, fontweight='bold')
        ax.set_facecolor('black')
        
        if not self.red.activaciones:
            return
        
        # Estadísticas
        predicciones = self.red.activaciones[-1][:, 0]
        max_pred = np.max(predicciones)
        confianza = max_pred
        prediccion = np.argmax(predicciones)
        
        info_text = f"""Dígito Real: {self.digito_actual}
Predicción: {prediccion}
Confianza: {confianza:.1%}

Arquitectura:
• Entrada: 784 píxeles
• Oculta 1: 128 neuronas
• Oculta 2: 64 neuronas  
• Salida: 10 clases

Activación: ReLU/Softmax
Estado: {"✅ Entrenada" if hasattr(self, 'entrenada') else "🔄 Lista"}"""
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
               color='white', fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="darkblue", alpha=0.3))
        
        ax.axis('off')
    
    def dibujar_activaciones_entrada(self, ax):
        """Dibuja un mapa de activaciones de la primera capa oculta"""
        ax.clear()
        ax.set_title("ACTIVACIONES L1", color='white', fontsize=10, fontweight='bold')
        ax.set_facecolor('black')
        
        if len(self.red.activaciones) > 1:
            # Tomar las primeras 64 activaciones de la primera capa oculta
            activaciones = self.red.activaciones[1][:64, 0]
            
            # Reorganizar en una grilla 8x8
            activaciones_grid = activaciones.reshape(8, 8)
            
            im = ax.imshow(activaciones_grid, cmap='hot', aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Colorbar pequeño
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.axis('off')

def main():
    """Función principal"""
    root = tk.Tk()
    
    # Mensaje de bienvenida
    messagebox.showinfo("🔢 Reconocedor de Dígitos", 
                       "¡Bienvenido al Reconocedor de Dígitos!\n\n"
                       "Esta aplicación simula el reconocimiento de dígitos escritos a mano.\n\n"
                       "🔍 Observa cómo fluyen las activaciones por la red\n"
                       "🎯 Ve las predicciones en tiempo real\n"
                       "🎲 Prueba diferentes dígitos\n\n"
                       "¡Selecciona un dígito para empezar!")
    
    app = VisualizadorReconocedor(root)
    root.mainloop()

if __name__ == "__main__":
    main()