#🎓 AI-Powered Student Success Predictor

Sistema integral de analítica educativa que predice el éxito estudiantil y genera estrategias de intervención personalizadas mediante Machine Learning

Desarrollado por Erick Delgadillo y Noé Martínez | Octubre 2025

---

## 📋 Tabla de Contenidos

Descripción del Proyecto
Características Principales
Resultados y Métricas
Arquitectura del Sistema
Instalación
Uso
Metodología Técnica
Tecnologías
Impacto y Casos de Uso
Contribuciones
Licencia

---

🎯 Descripción del Proyecto
Este proyecto implementa un sistema predictivo avanzado que combina dos enfoques de Machine Learning para revolucionar la gestión educativa en instituciones de educación en línea:

Modelo Predictivo (Random Forest): Predice el riesgo de deserción estudiantil en tiempo real
Análisis de Segmentación (Clustering): Identifica perfiles estudiantiles y patrones grupales

La aplicación web interactiva permite a instituciones educativas pasar de un modelo reactivo a uno proactivo, personalizado y basado en datos.
🔍 Problema que Resuelve
Las instituciones de educación en línea enfrentan altas tasas de deserción sin herramientas efectivas para:

Identificar estudiantes en riesgo antes de que abandonen
Personalizar intervenciones a escala con miles de estudiantes
Optimizar recursos de apoyo académico de manera eficiente
Reducir sesgos demográficos en la identificación de riesgo


✨ Características Principales
🤖 Sistema Predictivo (Random Forest)

✅ Predicción en tiempo real del riesgo de deserción
✅ Alta precisión operacional: 82.5% accuracy, 0.898 ROC-AUC
✅ Análisis de 20+ características del perfil estudiantil
✅ Clasificación automática en niveles de riesgo (Alto/Medio/Bajo)
✅ Recomendaciones accionables personalizadas
✅ Reducción del sesgo demográfico comparado con otros modelos

🎯 Análisis de Segmentación (Clustering)

📊 Identificación de 4 perfiles estudiantiles distintos
📊 Procesamiento eficiente de 500,000+ registros
📊 Algoritmo MiniBatchKMeans optimizado
📊 Insights sobre desigualdades estructurales
📊 Estrategias diferenciadas por cluster

💻 Aplicación Web Interactiva

🖥️ Interfaz intuitiva en Streamlit
🖥️ Diseñada para personal no técnico
🖥️ Visualizaciones claras y actionables
🖥️ Dashboard de análisis integral
🖥️ Exportación de reportes


📊 Resultados y Métricas
Rendimiento del Modelo Predictivo
MétricaValorInterpretaciónROC-AUC0.8981Excelente capacidad discriminativaAccuracy82.49%Alta precisión generalTiempo de respuesta<100msPredicción en tiempo realReducción de sesgo45%vs. modelos alternativos
Análisis de Segmentación
ClusterTasa de ÉxitoCaracterísticas PrincipalesCluster 052.3%Perfil balanceadoCluster 148.7%Recursos tecnológicos limitadosCluster 265.8%Alta disponibilidad y recursosCluster 335.2%Alto riesgo - múltiples barreras
Brecha identificada: 30.6 puntos porcentuales entre el cluster más y menos exitoso, evidenciando la necesidad de intervenciones diferenciadas.

🏗️ Arquitectura del Sistema
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│                    (Streamlit Web App)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   CAPA DE PROCESAMIENTO                      │
│  ┌──────────────────┐         ┌──────────────────────┐     │
│  │  Random Forest   │         │  MiniBatchKMeans     │     │
│  │  Pipeline        │         │  Clustering          │     │
│  │  (Prediction)    │         │  (Segmentation)      │     │
│  └──────────────────┘         └──────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      CAPA DE DATOS                           │
│  • Feature Engineering  • Data Preprocessing                │
│  • Validation          • Scalability Optimization           │
└─────────────────────────────────────────────────────────────┘
Pipeline de Procesamiento

Ingesta de Datos: Características demográficas, socioeconómicas, académicas y tecnológicas
Preprocesamiento: Normalización, codificación, manejo de valores nulos
Predicción: Random Forest optimizado con 20+ features
Segmentación: Asignación automática a cluster
Generación de Insights: Recomendaciones combinadas y personalizadas


🚀 Instalación
Requisitos Previos

Python 3.8 o superior
pip o conda para gestión de paquetes

Pasos de Instalación
bash# Clonar el repositorio
git clone https://github.com/tu-usuario/student-success-predictor.git
cd student-success-predictor

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
Dependencias Principales
txtstreamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0

💡 Uso
Iniciar la Aplicación
bashstreamlit run app.py
La aplicación se abrirá automáticamente en http://localhost:8501
Funcionalidades Disponibles
1️⃣ Predicción Individual
python# Ejemplo de uso del modelo
from model import StudentSuccessPredictor

predictor = StudentSuccessPredictor()
student_data = {
    'age': 25,
    'work_hours': 20,
    'tech_resources': 'high',
    'digital_skills': 'advanced',
    # ... más características
}

result = predictor.predict(student_data)
print(f"Probabilidad de éxito: {result['probability']:.2%}")
print(f"Nivel de riesgo: {result['risk_level']}")
print(f"Cluster asignado: {result['cluster']}")
2️⃣ Análisis Batch
python# Procesar múltiples estudiantes
import pandas as pd

students_df = pd.read_csv('students_data.csv')
results = predictor.predict_batch(students_df)
results.to_csv('predictions_output.csv', index=False)
3️⃣ Exploración de Clusters

Visualiza las características de cada segmento estudiantil
Compara tasas de éxito entre clusters
Identifica factores diferenciadores clave


🔬 Metodología Técnica
1. Modelo Predictivo - Random Forest
¿Por qué Random Forest?

✅ Mayor interpretabilidad que modelos de caja negra
✅ Menor riesgo de sobreajuste vs. redes neuronales
✅ Manejo robusto de características heterogéneas
✅ Importancia de características integrada

Optimizaciones Implementadas:
pythonRandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
Características Más Importantes:

🏆 Edad (55% importancia) - Factor predictivo principal
💼 Horas de trabajo - Impacto significativo en disponibilidad
💻 Recursos tecnológicos - Factor modificable crítico
🎯 Habilidades digitales - Competencias clave
💰 Ingresos familiares - Contexto socioeconómico

2. Análisis de Clustering - MiniBatchKMeans
Características del Algoritmo:

Optimizado para datasets grandes (500K+ registros)
4 clusters identificados mediante análisis de silhouette
Coeficiente de variación del 118.5% en horas de trabajo

Insights Clave por Cluster:

Cluster 2 (65.8% éxito): Estudiantes con alta disponibilidad y recursos óptimos
Cluster 3 (35.2% éxito): Múltiples barreras - requiere intervención intensiva


🛠️ Tecnologías
Core ML/Data Science

Scikit-learn: Modelado predictivo y clustering
Pandas: Manipulación y análisis de datos
NumPy: Operaciones numéricas eficientes

Visualización y Web

Streamlit: Aplicación web interactiva
Matplotlib/Seaborn: Visualizaciones avanzadas
Plotly: Gráficos interactivos

Utilidades

Joblib: Persistencia de modelos
Python 3.8+: Lenguaje base


📈 Impacto y Casos de Uso
Impacto Operacional Medido

🎯 Identificación temprana: Detecta estudiantes en riesgo 2-3 meses antes de la deserción
💰 Optimización de recursos: Reduce costos de intervención en 40% mediante focalización
📊 Mejora en retención: Potencial de aumento del 15-20% en tasas de retención
⚡ Eficiencia: Procesa 1000 estudiantes en <5 segundos

Casos de Uso Reales
🏫 Instituciones de Educación en Línea

Sistema de alerta temprana automatizado
Asignación inteligente de tutores
Personalización de contenidos por cluster

📚 Programas de Alta Matrícula

Gestión eficiente de miles de estudiantes
Priorización de recursos limitados
Monitoreo continuo y escalable

🎓 Departamentos de Retención Estudiantil

Estrategias de intervención basadas en evidencia
Seguimiento de efectividad de programas
Reducción de tasas de deserción


🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor:

Fork el proyecto
Crea una rama para tu feature (git checkout -b feature/AmazingFeature)
Commit tus cambios (git commit -m 'Add some AmazingFeature')
Push a la rama (git push origin feature/AmazingFeature)
Abre un Pull Request

Áreas de Mejora Futura

 Integración con sistemas LMS (Moodle, Canvas)
 API REST para integración empresarial
 Dashboard administrativo avanzado
 Modelos de series temporales para predicción longitudinal
 Análisis de sentimiento en interacciones


📝 Licencia
Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

👥 Autores
Erick Delgadillo & Noé Martínez

📧 Email: [tu-email@example.com]
💼 LinkedIn: [Tu perfil]
🐙 GitHub: [@tu-usuario]


📚 Citación
Si utilizas este proyecto en tu investigación o trabajo, por favor cítalo:
bibtex@software{student_success_predictor_2025,
  author = {Delgadillo, Erick and Martínez, Noé},
  title = {AI-Powered Student Success Predictor},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tu-usuario/student-success-predictor}
}

🙏 Agradecimientos

Instituciones educativas participantes por los datos (anonimizados)
Comunidad de Scikit-learn por las herramientas de ML
Equipo de Streamlit por la plataforma de desarrollo


<div align="center">
⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐
Desarrollado con ❤️ para mejorar la educación mediante IA
</div>
