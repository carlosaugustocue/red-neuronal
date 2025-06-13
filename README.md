# 🔢 Reconocedor de Dígitos con Visualización de Red Neuronal

Una aplicación educativa interactiva que permite visualizar en tiempo real cómo una red neuronal procesa y reconoce dígitos escritos a mano. Inspirada en las famosas visualizaciones de 3Blue1Brown sobre redes neuronales.


## 📋 Tabla de Contenidos

- [¿Qué es esta aplicación?](#-qué-es-esta-aplicación)
- [Características principales](#-características-principales)
- [Requisitos del sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Ejecución](#-ejecución)
- [Cómo usar la aplicación](#-cómo-usar-la-aplicación)
- [Descripción técnica](#-descripción-técnica)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Capturas de pantalla](#-capturas-de-pantalla)
- [Solución de problemas](#-solución-de-problemas)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## 🤖 ¿Qué es esta aplicación?

El **Reconocedor de Dígitos** es una herramienta educativa que simula el proceso de reconocimiento de dígitos manuscritos utilizando una red neuronal artificial. La aplicación permite a los usuarios:

- **Visualizar en tiempo real** cómo una red neuronal "ve" y procesa una imagen
- **Observar el flujo de activaciones** a través de cada capa de la red
- **Entender el proceso de clasificación** de imágenes de manera intuitiva
- **Experimentar** con diferentes dígitos y ver cómo responde la red

### 🎯 Propósito educativo

Esta aplicación está diseñada para:
- **Estudiantes** que quieren entender cómo funcionan las redes neuronales
- **Profesores** que necesitan una herramienta visual para explicar machine learning
- **Entusiastas de la IA** que desean explorar el reconocimiento de patrones
- **Desarrolladores** interesados en visualización de algoritmos de ML

## ✨ Características principales

### 🧠 Visualización de Red Neuronal
- **Arquitectura completa**: Muestra todas las capas de la red (784→128→64→10)
- **Activaciones en vivo**: Neuronas que cambian de tamaño y color según su activación
- **Conexiones ponderadas**: Líneas que muestran las conexiones entre neuronas
- **Flujo de datos**: Visualización del procesamiento paso a paso

### 🖼️ Procesamiento de Imágenes
- **Generación de dígitos**: Crea dígitos del 0 al 9 de forma procedural
- **Entrada realista**: Imágenes de 28×28 píxeles (estilo MNIST)
- **Normalización automática**: Preprocesamiento transparente de los datos

### 📊 Análisis de Resultados
- **Predicciones detalladas**: Probabilidades para cada clase (0-9)
- **Indicadores visuales**: Destacado de predicción correcta vs incorrecta
- **Métricas de confianza**: Nivel de certeza de la red
- **Mapas de activación**: Visualización de características detectadas

### 🎮 Controles Interactivos
- **Selección de dígitos**: Botones para cada número (0-9)
- **Generación aleatoria**: Crea dígitos aleatorios para probar
- **Regeneración**: Nuevas variaciones del mismo dígito
- **Re-entrenamiento**: Mejora la red neuronal en tiempo real

## 💻 Requisitos del sistema

### Sistema operativo
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+, otras distribuciones compatibles)

### Python
- **Versión mínima**: Python 3.7
- **Versión recomendada**: Python 3.8 o superior

### Memoria RAM
- **Mínimo**: 4 GB
- **Recomendado**: 8 GB o más

### Espacio en disco
- **Requerido**: ~100 MB (incluyendo dependencias)

## 🚀 Instalación

### 1. Verificar Python

Primero, verifica que tienes Python instalado:

```bash
python --version
# o
python3 --version
```

Si no tienes Python, descárgalo desde [python.org](https://www.python.org/downloads/)

### 2. Crear entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv venv_reconocedor

# Activar entorno virtual
# En Windows:
venv_reconocedor\Scripts\activate

# En macOS/Linux:
source venv_reconocedor/bin/activate
```

### 3. Instalar dependencias

```bash
pip install numpy matplotlib tkinter
```

### Instalación alternativa con requirements.txt

Si prefieres usar un archivo de requisitos, crea un archivo `requirements.txt`:

```txt
numpy>=1.19.0
matplotlib>=3.3.0
```

Luego instala:

```bash
pip install -r requirements.txt
```

### 4. Descargar el código

Descarga el archivo `reconocedor_digitos.py` y guárdalo en tu directorio de trabajo.

## ▶️ Ejecución

### Ejecutar la aplicación

```bash
python reconocedor_digitos.py
```

### Si usas Python 3 específicamente

```bash
python3 reconocedor_digitos.py
```

### Desde el entorno virtual

```bash
# Asegúrate de que el entorno virtual esté activado
source venv_reconocedor/bin/activate  # macOS/Linux
# o
venv_reconocedor\Scripts\activate     # Windows

# Ejecutar
python reconocedor_digitos.py
```

## 🎮 Cómo usar la aplicación

### Al iniciar

1. **Ventana de bienvenida**: Lee las instrucciones iniciales
2. **Interfaz principal**: Se abre con el dígito "6" por defecto
3. **Visualización automática**: La red procesa inmediatamente el dígito

### Controles disponibles

#### 🔢 Selección de dígitos
- **Botones 0-9**: Haz clic en cualquier número para procesarlo
- **🎲 Dígito Aleatorio**: Genera un dígito al azar
- **🔄 Regenerar**: Crea una nueva versión del dígito actual

#### 🛠️ Opciones avanzadas
- **🎯 Re-entrenar Red**: Mejora el rendimiento de la red neuronal
- **Interfaz responsiva**: Redimensiona la ventana según necesites

### Interpretando la visualización

#### 🖼️ Panel de entrada (superior izquierdo)
- **Imagen 28×28**: El dígito que la red está procesando
- **Borde verde**: Indica la imagen activa

#### 🧠 Red neuronal (centro)
- **Círculos**: Representan neuronas individuales
- **Tamaño**: Círculos más grandes = mayor activación
- **Color**: Intensidad de color = nivel de activación
- **Líneas**: Conexiones entre neuronas (azul=positivo, rojo=negativo)

#### 📊 Predicciones (superior derecho)
- **Barras horizontales**: Probabilidad para cada dígito (0-9)
- **Amarillo**: Predicción con mayor probabilidad
- **Borde verde**: Dígito correcto
- **✅/❌**: Indicador de acierto o error

#### 📈 Información adicional
- **Activaciones L1**: Mapa de calor de la primera capa oculta
- **Estadísticas**: Confianza, arquitectura, estado de la red

## 🔧 Descripción técnica

### Arquitectura de la red neuronal

```
Entrada: 784 neuronas (28×28 píxeles)
    ↓
Capa oculta 1: 128 neuronas (ReLU)
    ↓
Capa oculta 2: 64 neuronas (ReLU)
    ↓
Salida: 10 neuronas (Softmax)
```

### Algoritmos implementados

- **Inicialización**: He initialization para capas ReLU
- **Activación**: ReLU para capas ocultas, Softmax para salida
- **Entrenamiento**: Backpropagation simplificado
- **Optimización**: Gradient descent básico

### Características técnicas

- **Framework**: Numpy para cálculos, Matplotlib para visualización
- **Interfaz**: Tkinter (incluido con Python)
- **Rendimiento**: Optimizado para visualización en tiempo real
- **Datos**: Generación procedural de dígitos, no requiere datasets externos

## 📁 Estructura del proyecto

```
reconocedor-digitos/
│
├── reconocedor_digitos.py          # Archivo principal
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias (opcional)
└── venv_reconocedor/              # Entorno virtual (si se usa)
    ├── ...
```

### Clases principales

```python
class GeneradorDigitos:           # Genera dígitos sintéticos
class RedReconocedorDigitos:      # Implementa la red neuronal
class VisualizadorReconocedor:    # Maneja la interfaz y visualización
```

## 📸 Capturas de pantalla

### Interfaz principal
La aplicación muestra:
- **Panel superior**: Controles de selección de dígitos
- **Área central**: Visualización de la red neuronal completa
- **Paneles laterales**: Entrada, predicciones e información

### Características visuales
- **Tema oscuro**: Interfaz moderna estilo terminal
- **Colores codificados**: Diferentes colores para cada tipo de neurona
- **Animaciones**: Cambios suaves al procesar nuevos dígitos
- **Información en tiempo real**: Actualizaciones instantáneas

## 🔧 Solución de problemas

### Error: "No module named 'numpy'"
```bash
pip install numpy
```

### Error: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### Error: "No module named 'tkinter'"
En Ubuntu/Debian:
```bash
sudo apt-get install python3-tk
```

### La aplicación se ve borrosa (Windows)
- Clic derecho en `python.exe` → Propiedades → Compatibilidad
- Marcar "Anular comportamiento de escalado DPI alto"

### Rendimiento lento
- Cerrar otras aplicaciones pesadas
- Verificar que tienes suficiente RAM disponible
- Usar una versión más reciente de Python

### Pantalla muy pequeña
- La aplicación es redimensionable
- Arrastra las esquinas para ajustar el tamaño
- Tamaño mínimo recomendado: 1200×800 píxeles

## 🤝 Contribuir

### Formas de contribuir
- **Reportar bugs**: Usa los issues de GitHub
- **Sugerir características**: Propón nuevas funcionalidades
- **Mejorar código**: Submit pull requests
- **Documentación**: Ayuda a mejorar este README

### Desarrollo local

1. **Fork** el repositorio
2. **Clona** tu fork localmente
3. **Crea** una rama para tu característica: `git checkout -b nueva-caracteristica`
4. **Desarrolla** y **prueba** tus cambios
5. **Commit**: `git commit -m "Descripción de cambios"`
6. **Push**: `git push origin nueva-caracteristica`
7. **Pull Request**: Envía tu propuesta

### Estándares de código
- **PEP 8**: Seguir las convenciones de Python
- **Documentación**: Comentar funciones complejas
- **Testing**: Probar cambios antes de enviar

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

```
MIT License

Copyright (c) 2024 [Tu Nombre]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Texto completo de la licencia MIT...]
```

## 🙏 Reconocimientos

- **3Blue1Brown**: Inspiración para las visualizaciones de redes neuronales
- **MNIST Dataset**: Concepto de reconocimiento de dígitos manuscritos
- **Python Community**: Por las excelentes librerías de ciencia de datos

## 📞 Contacto y soporte

### ¿Necesitas ayuda?
- **Issues**: Para reportar problemas técnicos
- **Discussions**: Para preguntas generales
- **Email**: [caaranzazu_230@cue.edu.co]

### Enlaces útiles
- **Documentación de NumPy**: https://numpy.org/doc/
- **Documentación de Matplotlib**: https://matplotlib.org/
- **Tutorial de Redes Neuronales**: https://www.3blue1brown.com/neural-networks

---

## 🚀 ¡Empieza ahora!

```bash
# Instalación rápida
pip install numpy matplotlib
python reconocedor_digitos.py
```

**¡Disfruta explorando cómo las máquinas aprenden a reconocer patrones!** 🤖✨

---
