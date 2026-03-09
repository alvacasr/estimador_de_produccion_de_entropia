# Estimador de Producción de Entropía

Modelo de aprendizaje automático para estimar la producción de entropía en sistemas estocásticos utilizando redes neuronales.

## Descripción

Este proyecto implementa un modelo de red neuronal diseñado para estimar la producción de entropía a partir de datos de trayectorias generadas por sistemas estocásticos. El objetivo es aproximar cantidades termodinámicas que, en muchos casos, son difíciles de calcular analíticamente.

El estimador se entrena utilizando datos simulados y está implementado utilizando PyTorch.

Este proyecto fue desarrollado en el contexto de investigación en física estadística fuera del equilibrio y aprendizaje automático.

## Motivación

La producción de entropía es una magnitud fundamental en la termodinámica fuera del equilibrio. Sin embargo, estimarla a partir de trayectorias experimentales o simuladas puede ser complicado debido a que normalmente no se conoce completamente la dinámica subyacente del sistema.

El aprendizaje automático permite desarrollar métodos basados en datos para aproximar directamente la producción de entropía a partir de trayectorias observadas.

## Características

- Modelo de red neuronal para estimar producción de entropía
- Entrenamiento utilizando trayectorias simuladas
- Implementación con PyTorch
- Estructura modular para experimentación

## Tecnologías utilizadas

- Python
- PyTorch
- NumPy
- Matplotlib

## Estructura del proyecto

estimador_de_produccion_de_entropia  
│  
├── data/                # Datos de simulación y entrenamiento  
├── models/              # Modelos de redes neuronales  
├── notebooks/           # Experimentos y análisis exploratorio  
├── train.py             # Entrenamiento del modelo  
├── evaluate.py          # Evaluación del modelo  
└── README.md  

## Metodología

El flujo de trabajo del proyecto sigue los siguientes pasos:

1. Generación de datos de trayectorias estocásticas.
2. Entrenamiento de una red neuronal utilizando el conjunto de datos generado.
3. Estimación de la producción de entropía a partir del modelo entrenado.
4. Evaluación del desempeño del modelo comparando con valores teóricos.

## Instalación

Clonar el repositorio:

git clone https://github.com/alvacasr/estimador_de_produccion_de_entropia.git  
cd estimador_de_produccion_de_entropia

Instalar dependencias:

pip install -r requirements.txt

## Uso

Entrenar el modelo:

python train.py

Evaluar el modelo:

python evaluate.py

## Resultados

El modelo aprende a aproximar la producción de entropía a partir de datos de trayectorias generadas por procesos estocásticos.

Los resultados incluyen:

- valores estimados de producción de entropía
- comparación con valores teóricos
- visualización del desempeño del modelo

## Posibles mejoras

Algunas extensiones posibles del proyecto incluyen:

- probar diferentes arquitecturas de redes neuronales
- mejorar la estabilidad del entrenamiento
- aplicar el modelo a datos experimentales
- explorar otros observables termodinámicos

## Autor

Ramón Alvarado
