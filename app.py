import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import sklearn.compose._column_transformer

# Configuración de la página
st.set_page_config(
    page_title="PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LINEA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ===== VARIABLES DE COLOR (TEMA CLARO COMPLETO) ===== */
    :root {
        --primary-color: #2563eb;       /* Azul profesional */
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --success-color: #059669;       /* Verde éxito */
        --warning-color: #d97706;       /* Amarillo advertencia */
        --danger-color: #dc2626;        /* Rojo peligro */
        --text-primary: #000000;        /* Texto NEGRO para máximo contraste */
        --text-secondary: #374151;      /* Texto secundario */
        --text-light: #6b7280;          /* Texto claro */
        --bg-light: #ffffff;            /* Fondo BLANCO */
        --bg-card: #ffffff;             /* Fondo tarjetas BLANCO */
        --border-color: #d1d5db;        /* Bordes gris claro */
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* ===== ESTILOS BASE - FONDO BLANCO EN TODA LA APP ===== */
    .stApp {
        background-color: var(--bg-light) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ===== CORRECCIÓN DEFINITIVA PARA FORMULARIO ===== */
    /* Fondo del sidebar BLANCO */
    section[data-testid="stSidebar"] {
        background-color: white !important;
    }

    /* Todos los textos en sidebar en NEGRO */
    .css-1d391kg, 
    .css-1y4p8pa,
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
        background-color: white !important;
    }

    /* Labels de TODOS los controles - NEGRO Y NEGRITA */
    .stSelectbox label, 
    .stSlider label, 
    .stRadio label, 
    .stNumberInput label, 
    .stTextInput label,
    .stMultiSelect label,
    .stTextArea label,
    div[data-testid="stForm"] label,
    .stMarkdown label {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* Texto dentro de los controles - NEGRO */
    .stSelectbox div,
    .stSlider div,
    .stRadio div,
    .stNumberInput div,
    .stTextInput input,
    .stTextInput div,
    div[data-baseweb="select"] div,
    div[role="listbox"] div,
    div[role="option"] {
        color: #000000 !important;
        background-color: white !important;
    }

    /* Títulos del formulario - AZUL PARA DESTACAR */
    .css-1d391kg h3, 
    .css-1y4p8pa h3,
    .stMarkdown h3 {
        color: var(--primary-color) !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
    }

    /* Subtítulos de secciones - AZUL */
    .subsection-header {
        font-size: 1.1rem !important;
        color: var(--primary-color) !important;
        font-weight: 700 !important;
        margin: 1rem 0 0.5rem 0 !important;
        border-left: 4px solid var(--primary-light) !important;
        padding-left: 0.75rem !important;
        background: rgba(37, 99, 235, 0.05) !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }

    /* Texto descriptivo general - GRIS OSCURO */
    .css-1d391kg p, 
    .css-1y4p8pa p, 
    .css-1d391kg .stMarkdown, 
    .css-1y4p8pa .stMarkdown {
        color: #374151 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* Botón del formulario - AZUL CON CONTRASTE */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5) !important;
    }

    /* ===== GARANTIZAR QUE TODO EL TEXTO SEA VISIBLE ===== */
    /* Fuerza texto negro en toda la app */
    .stMarkdown, 
    .stText, 
    .stWrite, 
    .stAlert,
    p, li, span, div:not(button):not(svg):not(path) {
        color: var(--text-primary) !important;
    }

    /* Contenedores principales BLANCOS */
    .main .block-container,
    .stApp > div {
        background-color: white !important;
    }

</style>
""", unsafe_allow_html=True)



# Título principal con badges
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-header">PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LÍNEA</h1>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
        <span class="rf-badge">Random Forest Optimizado</span>
        <span class="cluster-badge">Análisis de Clusters</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Crear pestañas
tab1, tab2 = st.tabs(["🎯 PREDICCIÓN Y ANÁLISIS", "📊 ANÁLISIS DE CLUSTERS"])

with tab1:
    st.markdown("""
    **Sistema inteligente mejorado** que predice la probabilidad de éxito en educación en línea con **Random Forest optimizado** y **análisis de segmentación por clusters**.
    Precisión del **89.8%** | ROC-AUC de **0.898** | **4 clusters identificados**
    """)

# Mapeos para las variables
MAPEOS = {
    'si_no': {'Sí': 1, 'No': 0},
    'sexo': {'Hombre': 0, 'Mujer': 1, 'Otro': 2},
    'genero': {'Femenino': 0, 'Masculino': 1, 'Transgénero': 2, 'No binario': 3, 'Otro': 4},
    'situacion_conyugal': {'Soltero(a)': 0, 'Unión libre': 1, 'Casado(a)': 2, 'Divorciado(a)': 3, 'Separado(a)': 4, 'Viudo(a)': 5},
    'calificacion': {'Excelente': 0, 'Bueno': 1, 'Regular': 2, 'Malo': 3},
    'regimen_secundaria': {'Pública': 0, 'Privada': 1},
    'tipo_secundaria': {'General': 0, 'Técnica': 1, 'Telesecundaria': 2, 'Abierta': 3, 'Para adultos': 4},
    'edad_categoria': {'14-18': 0, '19-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
}

@st.cache_resource
def cargar_modelos():
    """
    Función para cargar todos los modelos con parche de compatibilidad
    """
    try:
        # PARCHE: Crear el atributo faltante si no existe
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            # Crear una clase dummy que simule el comportamiento esperado
            class _RemainderColsList(list):
                """Clase de compatibilidad para versiones antiguas de sklearn"""
                pass
            
            # Agregar el atributo al módulo
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
            setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
        
        # Cargar modelo Random Forest
        pipeline_rf = None
        metadata = None
        modelo_clusters = None
        
        # Intentar cargar el modelo RF
        try:
            if os.path.exists('modelo_rf_streamlit_compatible.joblib'):
                pipeline_rf = joblib.load('modelo_rf_streamlit_compatible.joblib')
                metadata = joblib.load('metadatos_compatible.joblib')
                st.sidebar.success("✅ Modelo RF cargado con workaround")
            else:
                st.warning("⚠️ Modelo RF no encontrado, usando modelo alternativo")
        except Exception as e:
            st.warning(f"⚠️ Error cargando RF: {e}")
        
        # Intentar cargar el modelo de clustering
        try:
            modelo_clusters = joblib.load('modelo_clusterizacion.pkl')
            st.sidebar.success("✅ Modelo de clustering cargado")
        except Exception as e:
            st.warning(f"⚠️ Modelo de clustering no disponible: {e}")
            modelo_clusters = None
        
        return pipeline_rf, metadata, modelo_clusters
        
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {str(e)}")
        return None, None, None

def generar_calificacion_final(row):
    # Factores que influyen positivamente
    factores_positivos = 0
    
    if row.get('estudios_previos_bachillerato') == 1:  # 'Sí' mapeado a 1
        factores_positivos += 0.5
    if row.get('cursos_linea_3anos') == 1:  # 'Sí' mapeado a 1
        factores_positivos += 0.3
    if row.get('trabaja') == 0:  # 'No' mapeado a 0
        factores_positivos += 0.2
    
    # Factores que influyen negativamente
    factores_negativos = 0
    
    if row.get('horas_trabajo_numeric', 0) >= 32:
        factores_negativos += 0.3
    
    # Calificación base + factores ajustados
    calificacion_base = np.random.normal(7.5, 1.5)
    calificacion_ajustada = calificacion_base + factores_positivos - factores_negativos
    
    # Ajustar a escala 0-10
    calificacion_final = np.clip(calificacion_ajustada, 5.0, 10.0)
    
    return round(calificacion_final, 1)

def crear_formulario():
    """Crear formulario interactivo completo - VERSIÓN MEJORADA"""
    with st.sidebar:
        st.markdown("### 📝 FORMULARIO DEL ESTUDIANTE")
        st.markdown("Complete la información para obtener su análisis personalizado")
        
        with st.form("formulario_estudiante"):
            # ===== SECCIÓN 1: DATOS DEMOGRÁFICOS =====
            st.markdown('<div class="subsection-header">👤 Datos demográficos</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                edad = st.slider("Edad", 14, 70, 25, help="Edad actual del estudiante")
            with col2:
                sexo = st.selectbox("Sexo", options=list(MAPEOS['sexo'].keys()))
            
            genero = st.selectbox("Género", options=list(MAPEOS['genero'].keys()))
            situacion_conyugal = st.selectbox("Situación conyugal", options=list(MAPEOS['situacion_conyugal'].keys()))
            
            # ===== SECCIÓN 2: SALUD Y ORIGEN =====
            st.markdown('<div class="subsection-header">🏥 Salud y origen</div>', unsafe_allow_html=True)
            
            discapacidad = st.selectbox("¿Tiene discapacidad?", options=list(MAPEOS['si_no'].keys()))
            indigena = st.selectbox("¿Se considera indígena?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 3: SITUACIÓN ECONÓMICA =====
            st.markdown('<div class="subsection-header">💰 Situación económica</div>', unsafe_allow_html=True)
            
            trabaja = st.selectbox("¿Trabaja actualmente?", options=list(MAPEOS['si_no'].keys()))
            
            if trabaja == 'Sí':
                horas_trabajo = st.slider("Horas de trabajo semanales", 0, 60, 40, 
                                        help="Número de horas que trabaja por semana")
            else:
                horas_trabajo = 0
            
            ingresos_hogar = st.select_slider(
                "Ingresos mensuales del hogar (MXN)",
                options=[3000, 7500, 12500, 17500, 22500, 30000],
                value=12500,
                format_func=lambda x: f"${x:,.0f} MXN"
            )
            
            beca = st.selectbox("¿Recibe alguna beca?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 4: TRAYECTORIA ACADÉMICA =====
            st.markdown('<div class="subsection-header">📚 Trayectoria académica</div>', unsafe_allow_html=True)
            
            col5, col6 = st.columns(2)
            with col5:
                regimen_secundaria = st.selectbox("Régimen de secundaria", options=list(MAPEOS['regimen_secundaria'].keys()))
            with col6:
                tipo_secundaria = st.selectbox("Tipo de secundaria", options=list(MAPEOS['tipo_secundaria'].keys()))
            
            estudios_previos = st.selectbox("¿Tiene estudios previos de bachillerato?", options=list(MAPEOS['si_no'].keys()))
            cursos_linea = st.selectbox("¿Ha tomado cursos en línea antes?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 5: HABILIDADES Y RECURSOS =====
            st.markdown('<div class="subsection-header">💻 Habilidades y recursos</div>', unsafe_allow_html=True)
            
            col7, col8 = st.columns(2)
            with col7:
                recursos_tec = st.slider("Recursos tecnológicos", 1, 5, 3, 
                                       help="Nivel de acceso a tecnología (1=bajo, 5=alto)")
            with col8:
                responsabilidades = st.slider("Responsabilidades", 1, 7, 3,
                                            help="Número de responsabilidades adicionales")
            
            comunicacion = st.select_slider("Habilidad de comunicación", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            evaluacion = st.select_slider("Habilidad evaluación información", 
                                        options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            organizacion = st.select_slider("Habilidad de organización", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            
            # Calcular categoría de edad automáticamente
            if edad <= 18:
                edad_categoria = '14-18'
            elif edad <= 25:
                edad_categoria = '19-25'
            elif edad <= 35:
                edad_categoria = '26-35'
            elif edad <= 45:
                edad_categoria = '36-45'
            else:
                edad_categoria = '45+'
            
            # Botón de enviar
            submitted = st.form_submit_button("🎓 Obtener Análisis Completo", use_container_width=True)
            
            datos = {
                'edad': edad, 'sexo': sexo, 'genero': genero, 'situacion_conyugal': situacion_conyugal,
                'discapacidad': discapacidad, 'indigena': indigena, 'trabaja': trabaja,
                'horas_trabajo_numeric': horas_trabajo, 'ingresos_hogar_numeric': ingresos_hogar,
                'beca': beca, 'regimen_secundaria': regimen_secundaria, 'tipo_secundaria': tipo_secundaria,
                'estudios_previos_bachillerato': estudios_previos, 'cursos_linea_3anos': cursos_linea,
                'score_recursos_tecnologicos': recursos_tec, 'score_responsabilidades': responsabilidades,
                'comunicacion_escrita': comunicacion, 'evaluacion_informacion': evaluacion,
                'organizacion_plataforma': organizacion, 'edad_categoria': edad_categoria
            }
            
            return submitted, datos

def preprocesar_datos(datos):
    """Preprocesar datos para el modelo"""
    datos_procesados = {}
    
    for key, value in datos.items():
        if key in ['edad', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 
                  'score_recursos_tecnologicos', 'score_responsabilidades']:
            datos_procesados[key] = float(value)
        else:
            # Buscar en los mapeos correspondientes
            for mapeo_key, mapeo in MAPEOS.items():
                if value in mapeo:
                    datos_procesados[key] = mapeo[value]
                    break
            else:
                datos_procesados[key] = value
    
    return datos_procesados

def crear_dataframe_modelo(datos_procesados):
    """Crear DataFrame con la estructura que el modelo espera"""
    columnas_esperadas = [
        'edad', 'sexo', 'genero', 'situacion_conyugal', 'discapacidad', 'indigena',
        'trabaja', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 'beca',
        'regimen_secundaria', 'tipo_secundaria', 'estudios_previos_bachillerato',
        'cursos_linea_3anos', 'score_recursos_tecnologicos', 'score_responsabilidades',
        'comunicacion_escrita', 'evaluacion_informacion', 'organizacion_plataforma',
        'edad_categoria'
    ]
    
    df = pd.DataFrame(columns=columnas_esperadas)
    
    for columna in columnas_esperadas:
        if columna in datos_procesados:
            df[columna] = [datos_procesados[columna]]
        else:
            df[columna] = [0]  # Valor por defecto
    
    return df

def predecir_cluster(datos_procesados, modelo_clusters):
    """Predecir el cluster del estudiante"""
    try:
        # Calcular calificación final EXACTA como en el código original
        calificacion_final = generar_calificacion_final(datos_procesados)
        datos_procesados['calificacion_final'] = calificacion_final * 10  # Convertir a 0-100
        
        features_clustering = modelo_clusters['features_clustering']
        df_input = pd.DataFrame([datos_procesados])
        
        features_faltantes = [f for f in features_clustering if f not in df_input.columns]
        if features_faltantes:
            st.warning(f"⚠️ Algunas features faltan para clustering: {features_faltantes}")
            return None
        
        X_scaled = modelo_clusters['scaler'].transform(df_input[features_clustering])
        cluster = modelo_clusters['kmeans_model'].predict(X_scaled)[0]
        distancias = modelo_clusters['kmeans_model'].transform(X_scaled)[0]
        
        confianza = 1 / (1 + distancias.min())
        
        return {
            'cluster': cluster,
            'distancias': distancias,
            'confianza': confianza,
            'features_utilizadas': features_clustering,
            'calificacion_final_calculada': calificacion_final
        }
    except Exception as e:
        st.error(f"❌ Error en predicción de cluster: {str(e)}")
        return None

def crear_gauge_chart(probabilidad):
    """Crear gráfico tipo gauge para la probabilidad"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probabilidad * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Éxito (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 40], 'color': "#FFCDD2"},
                {'range': [40, 60], 'color': "#FFECB3"},
                {'range': [60, 80], 'color': "#C8E6C9"},
                {'range': [80, 100], 'color': "#A5D6A7"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    
    fig.update_layout(height=300)
    return fig

def mostrar_feature_importance_personalizada(datos_usuario):
    """Mostrar feature importance personalizada"""
    # Feature importance del Random Forest (basada en tus resultados)
    features_rf = {
        'edad': 0.548,
        'edad_categoria': 0.102,
        'estudios_previos_bachillerato': 0.051,
        'horas_trabajo_numeric': 0.050,
        'ingresos_hogar_numeric': 0.034,
        'score_recursos_tecnologicos': 0.033,
        'cursos_linea_3anos': 0.032,
        'tipo_secundaria': 0.027,
        'score_responsabilidades': 0.023
    }
    
    st.markdown('<div class="section-header">📊 Importancia de factores para el caso particular</div>', unsafe_allow_html=True)
    
    # Crear gráfico de barras
    fig_importance = px.bar(
        x=list(features_rf.values())[:6],
        y=list(features_rf.keys())[:6],
        orientation='h',
        title="Top 6 factores más importantes (Random Forest)",
        color=list(features_rf.values())[:6],
        color_continuous_scale='Greens'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Análisis personalizado
    st.markdown("**🔍 Análisis de tu perfil:**")
    
    if datos_usuario['edad'] > 30:
        st.markdown(f'<div class="feature-importance-box">📊 **Edad ({datos_usuario["edad"]} años)**: Factor dominante pero menos que en otros modelos (55% vs 80% en GB)</div>', unsafe_allow_html=True)
    
    if datos_usuario['ingresos_hogar_numeric'] < 15000:
        st.markdown('<div class="feature-importance-box">💰 **Ingresos**: Nivel económico bajo puede ser un factor de riesgo importante</div>', unsafe_allow_html=True)
    
    if datos_usuario['horas_trabajo_numeric'] > 30:
        st.markdown('<div class="feature-importance-box">⏰ **Carga laboral**: Muchas horas de trabajo pueden afectar el rendimiento académico</div>', unsafe_allow_html=True)

def mostrar_resultados_cluster(resultado_cluster):
    """Mostrar resultados del análisis de cluster"""
    if not resultado_cluster:
        return
    
    cluster = resultado_cluster['cluster']
    confianza = resultado_cluster['confianza']
    calificacion = resultado_cluster.get('calificacion_final_calculada', 'N/A')
    
    st.markdown('<div class="section-header">🎯 ANÁLISIS DE SEGMENTACIÓN (CLUSTERS)</div>', unsafe_allow_html=True)
    
    # Descripciones de clusters
    descripciones_clusters = {
        0: {
            "nombre": "🎓 Estudiantes con Ventaja Socioeconómica",
            "descripcion": "Jóvenes con buenos recursos tecnológicos y económicos, menor carga de responsabilidades.",
            "tamaño": "21.5% de estudiantes",
            "exito_promedio": "59.1%"
        },
        1: {
            "nombre": "💼 Estudiantes Trabajadores",
            "descripcion": "Adultos jóvenes que combinan estudio con trabajo extensivo, alta carga de responsabilidades.",
            "tamaño": "27.6% de estudiantes",
            "exito_promedio": "45.1%"
        },
        2: {
            "nombre": "🌟 Estudiantes Maduros Resilientes", 
            "descripcion": "Adultos con mayor edad y responsabilidades, pero alto rendimiento académico.",
            "tamaño": "21.7% de estudiantes",
            "exito_promedio": "65.8%"
        },
        3: {
            "nombre": "📚 Estudiantes Jóvenes en Desventaja",
            "descripcion": "Adolescentes con recursos limitados, requieren apoyo prioritario.",
            "tamaño": "29.2% de estudiantes",
            "exito_promedio": "35.2%"
        }
    }
    
    info_cluster = descripciones_clusters.get(cluster, {})
    
    # Métricas del cluster
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Cluster Asignado", f"Cluster {cluster}")
    
    with col2:
        st.metric("Confianza Cluster", f"{confianza:.1%}")
    
    with col3:
        st.metric("Calificación Calculada", f"{calificacion}/10")
    
    with col4:
        st.metric("Tamaño del Grupo", info_cluster.get("tamaño", "N/A"))
    
    with col5:
        st.metric("Éxito Promedio", info_cluster.get("exito_promedio", "N/A"))
    
    # Información detallada del cluster
    st.markdown(f'<div class="cluster-info-box">'
                f'<h3>📋 {info_cluster.get("nombre", "Cluster Desconocido")}</h3>'
                f'<p><strong>Descripción:</strong> {info_cluster.get("descripcion", "Información no disponible")}</p>'
                f'<p><strong>Características principales:</strong></p>'
                f'<ul>'
                f'<li><strong>Tamaño:</strong> {info_cluster.get("tamaño", "N/A")}</li>'
                f'<li><strong>Éxito académico promedio:</strong> {info_cluster.get("exito_promedio", "N/A")}</li>'
                f'<li><strong>Calificación estimada:</strong> {calificacion}/10</li>'
                f'<li><strong>Confianza de asignación:</strong> {confianza:.1%}</li>'
                f'</ul>'
                f'</div>', unsafe_allow_html=True)
    
    # Gráfico de distancias a clusters
    fig_distancias = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in range(4)],
            y=resultado_cluster['distancias'],
            marker_color=['#2196F3' if i == cluster else '#64B5F6' for i in range(4)],
            text=[f'{d:.2f}' for d in resultado_cluster['distancias']],
            textposition='auto',
        )
    ])
    
    fig_distancias.update_layout(
        title="Distancias a los Centroides de Cada Cluster",
        xaxis_title="Cluster",
        yaxis_title="Distancia",
        height=400
    )
    
    st.plotly_chart(fig_distancias, use_container_width=True)
    
    # Recomendaciones específicas por cluster
    recomendaciones_clusters = {
        0: [
            "**Potencial de liderazgo**: Participar como mentor de otros estudiantes",
            "**Programas de excelencia**: Explorar oportunidades académicas avanzadas",
            "**Proyectos especiales**: Involucrarse en iniciativas institucionales"
        ],
        1: [
            "**Gestión del tiempo**: Solicitar flexibilidad en horarios de entrega",
            "**Comunidad de apoyo**: Unirse a grupos de estudiantes trabajadores",
            "**Asesoría académica**: Recibir orientación específica para balance trabajo-estudio"
        ],
        2: [
            "**Redes de apoyo**: Participar en comunidades de estudiantes maduros", 
            "**Recursos tecnológicos**: Acceder a programas de actualización digital",
            "**Experiencia compartida**: Contribuir como referente para otros estudiantes"
        ],
        3: [
            "**Apoyo prioritario**: Solicitar asistencia económica y tecnológica urgente",
            "**Mentorías personalizadas**: Participar en programas de acompañamiento",
            "**Seguimiento estrecho**: Mantener contacto frecuente con asesores académicos"
        ]
    }
    
    st.markdown(f'<div class="section-header">💡 RECOMENDACIONES ESPECÍFICAS PARA CLUSTER {cluster}</div>', unsafe_allow_html=True)
    
    for i, rec in enumerate(recomendaciones_clusters.get(cluster, []), 1):
        st.markdown(f'<div class="recommendation-box">**{i}.** {rec}</div>', unsafe_allow_html=True)

def mostrar_resultados_prediccion(probabilidad, prediccion, datos_originales, metadata):
    """Mostrar resultados de la predicción de éxito académico"""
    # Determinar nivel de riesgo con RF
    if probabilidad < 0.35:
        nivel_riesgo = "MUY ALTO"
        color_clase = "risk-high"
        emoji = "🔴"
        confianza = "Baja"
    elif probabilidad < 0.5:
        nivel_riesgo = "ALTO"  
        color_clase = "risk-high"
        emoji = "🟠"
        confianza = "Media"
    elif probabilidad < 0.7:
        nivel_riesgo = "MEDIO"
        color_clase = "risk-medium"
        emoji = "🟡"
        confianza = "Buena"
    elif probabilidad < 0.85:
        nivel_riesgo = "BAJO"
        color_clase = "risk-low"
        emoji = "🟢"
        confianza = "Alta"
    else:
        nivel_riesgo = "MUY BAJO"
        color_clase = "risk-low"
        emoji = "✅"
        confianza = "Muy alta"
    
    # Mostrar métricas principales
    st.markdown('<div class="section-header">🌲 PREDICCIÓN DE ÉXITO ACADÉMICO</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilidad de éxito", f"{probabilidad:.1%}")
    
    with col2:
        resultado = "✅ ÉXITO PROBABLE" if prediccion == 1 else "⚠️ RIESGO ALTO"
        st.metric("Predicción", resultado)
    
    with col3:
        st.metric("Nivel de riesgo", f"{emoji} {nivel_riesgo}")
    
    with col4:
        st.metric("Confianza", confianza)
    
    # Gráfico gauge interactivo
    st.plotly_chart(crear_gauge_chart(probabilidad), use_container_width=True)
    
    # Mostrar feature importance personalizada
    mostrar_feature_importance_personalizada(datos_originales)

def generar_recomendaciones_combinadas(probabilidad, datos, cluster_resultado):
    """Generar recomendaciones combinando predicción y clusterización"""
    st.markdown('<div class="section-header">🎯 RECOMENDACIONES INTEGRALES</div>', unsafe_allow_html=True)
    
    recomendaciones = []
    
    # Recomendaciones basadas en la predicción
    if probabilidad < 0.4:
        recomendaciones.append("**Intervención integral inmediata**: El modelo RF indica múltiples factores de riesgo")
        recomendaciones.append("**Contacto urgente con asesor académico** para plan personalizado")
    elif probabilidad < 0.6:
        recomendaciones.append("**Plan de mejora multifactor**: RF identifica áreas específicas de mejora")
        recomendaciones.append("**Programa de acompañamiento** personalizado recomendado")
    else:
        recomendaciones.append("**Perfil favorable**: RF predice alta probabilidad de éxito")
        recomendaciones.append("**Mantener estrategia actual** con seguimiento regular")
    
    # Recomendaciones específicas basadas en features importantes en RF
    if datos['edad'] > 35:
        recomendaciones.append("**Programa para adultos mayores**: Estrategias específicas para estudiantes maduros")
    
    if datos['ingresos_hogar_numeric'] < 12000:
        recomendaciones.append("**Apoyo económico crítico**: RF identifica ingresos como factor clave")
        recomendaciones.append("**Solicitar beca o apoyo financiero** urgentemente")
    
    if datos['horas_trabajo_numeric'] > 35:
        recomendaciones.append("**Gestión tiempo-trabajo crítica**: RF muestra que es factor importante")
        recomendaciones.append("**Negociar flexibilidad laboral** si es posible")
    
    # Recomendaciones basadas en el cluster
    if cluster_resultado:
        cluster = cluster_resultado['cluster']
        if cluster == 0:
            recomendaciones.append("**Aprovechar ventajas socioeconómicas**: Potencial de liderazgo académico")
        elif cluster == 1:
            recomendaciones.append("**Balance trabajo-estudio**: Estrategias específicas para estudiantes trabajadores")
        elif cluster == 2:
            recomendaciones.append("**Capitalizar resiliencia**: Compartir experiencias como estudiante maduro")
        elif cluster == 3:
            recomendaciones.append("**Apoyo prioritario integral**: Enfoque multidimensional para estudiantes jóvenes")
    
    # Mostrar recomendaciones
    for i, rec in enumerate(recomendaciones, 1):
        st.markdown(f'<div class="recommendation-box">**{i}.** {rec}</div>', unsafe_allow_html=True)

def main():
    """Función principal"""
    pipeline, metadata, modelo_clusters = cargar_modelos()
    
    # Crear formulario en sidebar
    submitted, datos_usuario = crear_formulario()
    
    # Área principal para resultados (solo en la pestaña 1)
    with tab1:
        if submitted:
            try:
                with st.spinner('🌲 Random Forest analizando... Modelo más balanceado procesando datos...'):
                    # Preprocesar datos
                    datos_procesados = preprocesar_datos(datos_usuario)
                    
                    # Crear DataFrame
                    X_nuevo = crear_dataframe_modelo(datos_procesados)
                    
                    # Hacer predicción de éxito académico
                    probabilidad = None
                    prediccion = None
                    
                    if pipeline is not None:
                        probabilidad = pipeline.predict_proba(X_nuevo)[0, 1]
                        prediccion = pipeline.predict(X_nuevo)[0]
                    else:
                        # Simular predicción si el modelo no está disponible
                        st.warning("⚠️ Usando predicción simulada - modelo RF no disponible")
                        probabilidad = 0.65
                        prediccion = 1
                    
                    # Hacer predicción de cluster
                    resultado_cluster = None
                    if modelo_clusters is not None:
                        with st.spinner('🎯 Analizando segmentación por clusters...'):
                            resultado_cluster = predecir_cluster(datos_procesados, modelo_clusters)
                
                # Mostrar resultados
                st.success("🌲 ¡Análisis completado exitosamente! (Predicción + Clusterización)")
                
                # Mostrar resultados de predicción
                if probabilidad is not None:
                    mostrar_resultados_prediccion(probabilidad, prediccion, datos_usuario, metadata)
                
                # Mostrar resultados de clusterización
                if resultado_cluster:
                    mostrar_resultados_cluster(resultado_cluster)
                else:
                    st.warning("⚠️ No se pudo realizar el análisis de clusters")
                
                # Mostrar recomendaciones combinadas
                if probabilidad is not None:
                    generar_recomendaciones_combinadas(probabilidad, datos_usuario, resultado_cluster)
                
                # Información técnica
                with st.expander("🔧 DETALLES TÉCNICOS COMPLETOS"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🌲 Random Forest:**")
                        if probabilidad is not None:
                            st.write(f"🎯 Probabilidad exacta: {probabilidad:.6f}")
                        st.write(f"⚖️ Umbral de clasificación: 0.5")
                        if metadata:
                            st.write(f"📈 ROC-AUC: {metadata.get('roc_auc', 'N/A')}")
                            st.write(f"🎯 Accuracy: {metadata.get('accuracy', 'N/A')}")
                    
                    with col2:
                        st.write("**🎯 Clusterización:**")
                        if resultado_cluster:
                            st.write(f"🔢 Cluster asignado: {resultado_cluster['cluster']}")
                            st.write(f"🎯 Confianza cluster: {resultado_cluster['confianza']:.3f}")
                            if modelo_clusters:
                                st.write(f"📊 Número de clusters: {modelo_clusters['kmeans_model'].n_clusters}")
                                st.write(f"🔍 Features utilizadas: {len(resultado_cluster['features_utilizadas'])}")
                    
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("ℹ️ Verifica que todos los campos estén completos correctamente.")
        
        # Información cuando no hay predicción
        else:
            st.info("""
            👈 **Complete el formulario** para obtener un análisis completo con Random Forest optimizado y segmentación por clusters.
            
            **🔍 Análisis que recibirás:**
            -  **🌲 Predicción de éxito académico** con Random Forest optimizado
            -  **🎯 Segmentación por cluster** para identificar tu perfil estudiantil
            -  **💡 Recomendaciones personalizadas** basadas en ambos análisis
            -  **📊 Visualizaciones interactivas** de resultados
            
            ** Ventajas del sistema integrado:**
            -  **Predicción precisa** (89.8% ROC-AUC)
            -  **Segmentación inteligente** (4 clusters identificados)
            -  **Recomendaciones contextualizadas** por perfil
            -  **Enfoque integral** para el éxito académico
            """)
            
            # Información sobre clusters
            st.markdown('<div class="section-header">🎯 CLUSTERS IDENTIFICADOS</div>', unsafe_allow_html=True)
            
            clusters_info = {
                "Cluster 0": "🎓 Estudiantes con Ventaja Socioeconómica (21.5%) - Éxito: 59.1%",
                "Cluster 1": "💼 Estudiantes Trabajadores (27.6%) - Éxito: 45.1%", 
                "Cluster 2": "🌟 Estudiantes Maduros Resilientes (21.7%) - Éxito: 65.8%",
                "Cluster 3": "📚 Estudiantes Jóvenes en Desventaja (29.2%) - Éxito: 35.2%"
            }
            
            for cluster, desc in clusters_info.items():
                st.markdown(f"• **{cluster}**: {desc}")

    # Pestaña 2: Análisis de Clusters
    with tab2:
        st.markdown('<div class="section-header">📊 ANÁLISIS DETALLADO DE CLUSTERS</div>', unsafe_allow_html=True)
        
        # Mostrar las imágenes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Visualización 2D de Clusters (PCA)")
            try:
                st.image("images/clusters_PCA_2D.png", use_column_width=True, 
                        caption="Distribución de estudiantes en los 4 clusters identificados")
            except FileNotFoundError:
                st.warning("⚠️ No se encontró la imagen 'clusters_PCA_2D.png'. Asegúrate de que esté en el mismo directorio.")
        
        with col2:
            st.markdown("### Análisis de Componentes Principales")
            try:
                st.image("images/radar_cluster_estandarizado.png", use_column_width=True, 
                        caption="Radar con las características de los estudiantes")
            except FileNotFoundError:
                st.warning("⚠️ No se encontró la imagen 'radar_cluster_estandarizado.png'. Asegúrate de que esté en el mismo directorio.")
        
        # Texto descriptivo de los clusters
        st.markdown('<div class="section-header">🎯 CARACTERIZACIÓN DETALLADA DE PERFILES</div>', unsafe_allow_html=True)
        
        # Cluster 2
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>🌟 Cluster 2: "Estudiantes maduros con alta responsabilidad"</h3>
            <p><strong>Perfil:</strong> Adultos mayores (45.6 años promedio) con alta carga de responsabilidades</p>
            <p><strong>Fortalezas:</strong> Mayor tasa de éxito académico (65.8%) a pesar de responsabilidades</p>
            <p><strong>Desafíos:</strong> Recursos tecnológicos limitados (2.98/4.0)</p>
            <p><strong>Oportunidad:</strong> Estudiantes resilientes que podrían beneficiarse de mejor acceso tecnológico</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 0
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>🎓 Cluster 0: "Estudiantes privilegiados jóvenes"</h3>
            <p><strong>Perfil:</strong> Jóvenes (20 años) con alto nivel socioeconómico y recursos tecnológicos</p>
            <p><strong>Fortalezas:</strong> Alto éxito académico (59.1%) apoyado por buenos recursos</p>
            <p><strong>Ventaja:</strong> Menos responsabilidades y mayor acceso a recursos educativos</p>
            <p><strong>Potencial:</strong> Podrían servir como mentores o grupos de referencia</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 1
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>💼 Cluster 1: "Estudiantes trabajadores"</h3>
            <p><strong>Perfil:</strong> Adultos jóvenes (21.9 años) que combinan estudio con trabajo extensivo (27h/semana)</p>
            <p><strong>Desafío:</strong> Bajo éxito académico (45.1%) posiblemente por carga laboral</p>
            <p><strong>Necesidad:</strong> Flexibilidad horaria y apoyo específico para estudiantes trabajadores</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 3
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>📚 Cluster 3: "Estudiantes jóvenes con desventaja socioeconómica"</h3>
            <p><strong>Perfil:</strong> Adolescentes (17.6 años) con bajos ingresos familiares y recursos limitados</p>
            <p><strong>Alerta crítica:</strong> Menor tasa de éxito (35.2%) requiere intervención prioritaria</p>
            <p><strong>Factores de riesgo:</strong> Bajos ingresos, recursos tecnológicos insuficientes</p>
            <p><strong>Estrategia:</strong> Apoyo económico y tecnológico urgente</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Resumen estadístico
        st.markdown('<div class="section-header">📈 RESUMEN ESTADÍSTICO DE CLUSTERS</div>', unsafe_allow_html=True)
        
        # Crear una tabla resumen
        resumen_data = {
            'Cluster': ['2 - Maduros', '0 - Privilegiados', '1 - Trabajadores', '3 - En desventaja'],
            'Edad Promedio': ['45.6 años', '20 años', '21.9 años', '17.6 años'],
            'Éxito Académico': ['65.8%', '59.1%', '45.1%', '35.2%'],
            'Tamaño': ['21.7%', '21.5%', '27.6%', '29.2%'],
            'Prioridad': ['Media', 'Baja', 'Media', 'Alta']
        }
        
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        # Recomendaciones generales
        st.markdown('<div class="section-header">💡 RECOMENDACIONES ESTRATÉGICAS</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
        <strong>Estrategias diferenciadas por cluster:</strong>
        <ul>
        <li><strong>Cluster 2:</strong> Programas de actualización tecnológica y horarios flexibles</li>
        <li><strong>Cluster 0:</strong> Programas de liderazgo y mentoría estudiantil</li>
        <li><strong>Cluster 1:</strong> Flexibilidad en entregas y asesoría para balance trabajo-estudio</li>
        <li><strong>Cluster 3:</strong> Apoyo económico urgente, becas tecnológicas y acompañamiento intensivo</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()








