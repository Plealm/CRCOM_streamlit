import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import glob
import json
import os

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==========================================
st.set_page_config(
    page_title="Monitor CRC - Análisis 360",
    page_icon="🇨🇴",
    layout="wide"
)

# CSS MEJORADO - PESTAÑAS MÁS VISIBLES
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    
    /* Estilo de Pestañas Mejorado */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 70px;
        background-color: #ffffff;
        border-radius: 10px;
        color: #262730;
        font-size: 16px;
        font-weight: 700;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 0 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #0066cc;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white !important;
        border: 2px solid #0066cc;
        box-shadow: 0 4px 12px rgba(0,102,204,0.4);
    }
    
    /* Estilo Métricas */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Headers de secciones */
    .stMarkdown h3 {
        color: #0066cc;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Footer personalizado */
    .footer {
        position: relative;
        margin-top: 50px;
        padding: 20px;
        background-color: #f1f3f6;
        border-radius: 10px;
        text-align: center;
        font-size: 14px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE DATOS (POLARS)
# ==========================================

@st.cache_data(ttl=3600)
def cargar_geojson():
    url = 'https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json'
    with urlopen(url) as response:
        return json.load(response)

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def cargar_datos_polars(patron_archivos):
    archivos = glob.glob(patron_archivos)
    
    if not archivos:
        return None, None

    try:
        # 1. Leemos los archivos
        df = pl.read_parquet(archivos)

        # 2. Seleccionamos SOLO lo necesario y optimizamos tipos (AQUÍ ESTÁ LA MAGIA)
        # Convertir Strings a Categorical reduce el uso de RAM hasta en un 80%
        df = df.select([
            pl.col('ANNO').cast(pl.Int16), # Año cabe en Int16
            pl.col('TRIMESTRE').cast(pl.Int8),
            pl.col('ID_DEPARTAMENTO'), # Lo necesitamos para el mapa
            pl.col('DEPARTAMENTO').cast(pl.Categorical),
            pl.col('MUNICIPIO').cast(pl.Categorical),
            pl.col('EMPRESA').cast(pl.Categorical),
            pl.col('SEGMENTO').cast(pl.Categorical),
            pl.col('SERVICIO_PAQUETE').cast(pl.Categorical),
            pl.col('TECNOLOGIA').cast(pl.Categorical),
            pl.col('VELOCIDAD_EFECTIVA_DOWNSTREAM'),
            pl.col('VELOCIDAD_EFECTIVA_UPSTREAM'),
            pl.col('CANTIDAD_LINEAS_ACCESOS'),
            pl.col('VALOR_FACTURADO_O_COBRADO'), 
            pl.col('OTROS_VALORES_FACTURADOS')
        ])

        # 3. Transformaciones
        df = df.with_columns([
            pl.col("VALOR_FACTURADO_O_COBRADO").fill_null(0),
            pl.col("OTROS_VALORES_FACTURADOS").fill_null(0),
            pl.col("CANTIDAD_LINEAS_ACCESOS").fill_null(0),
            pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").fill_null(0),
            pl.format("{}-T{}", pl.col("ANNO"), pl.col("TRIMESTRE")).alias("PERIODO"),
            # Convertimos a string solo al final y para la columna específica del mapa
            pl.col("ID_DEPARTAMENTO").cast(pl.String).str.zfill(2).alias("ID_DEPTO_MAPA")
        ])

        # Calculamos Valor Total
        df = df.with_columns(
            (pl.col("VALOR_FACTURADO_O_COBRADO") + pl.col("OTROS_VALORES_FACTURADOS")).alias("VALOR_TOTAL")
        )

        # 4. Extraemos las opciones (Usamos casting a string temporal para obtener listas limpias)
        opciones = {
            'anos': df["ANNO"].unique().sort().to_list(),
            'deptos': df["DEPARTAMENTO"].unique().cast(pl.String).sort().to_list(),
            'empresas': df["EMPRESA"].unique().cast(pl.String).sort().to_list(),
            'paquetes': df["SERVICIO_PAQUETE"].unique().cast(pl.String).sort().to_list(),
            'tecnologias': df["TECNOLOGIA"].unique().cast(pl.String).sort().to_list(),
            'max_val_facturado': df["VALOR_FACTURADO_O_COBRADO"].max(),
            'max_otros': df["OTROS_VALORES_FACTURADOS"].max()
        }

        return df, opciones

    except Exception as e:
        st.error(f"Error Polars: {e}")
        return None, None

# ==========================================
# 3. INICIALIZACIÓN
# ==========================================

PATRON_ARCHIVOS = "./data_part_*.parquet" 

with st.spinner('Cargando motor de datos...'):
    df, opciones = cargar_datos_polars(PATRON_ARCHIVOS)
    geojson_colombia = cargar_geojson()

if df is None:
    st.error(f"❌ No se encontraron archivos con el patrón: {PATRON_ARCHIVOS}")
    st.warning("Asegúrate de haber subido los archivos data_part_0.parquet, data_part_1.parquet, etc.")
    st.stop()

# ==========================================
# 4. SIDEBAR - FILTROS
# ==========================================

st.sidebar.header("🔍 Filtros de Análisis")

# A. Año
sel_ano = st.sidebar.multiselect("📅 Año", opciones['anos'], default=opciones['anos'])

# B. Departamento
sel_depto = st.sidebar.multiselect("📍 Departamento", opciones['deptos'])

# C. Municipio (Filtrado dinámico)
munis_disponibles = []
if len(sel_depto) > 0:
    subset_munis = df.filter(pl.col("DEPARTAMENTO").is_in(sel_depto))
    munis_disponibles = subset_munis["MUNICIPIO"].unique().sort().to_list()

sel_muni = st.sidebar.multiselect(
    "🏙️ Municipio", 
    munis_disponibles if len(munis_disponibles) > 0 else [],
    disabled=len(sel_depto) == 0,
    help="Seleccione primero un Departamento"
)

# D. Otros Filtros
sel_empresa = st.sidebar.multiselect("🏢 Empresa", opciones['empresas'])
sel_paquete = st.sidebar.multiselect("📦 Paquete", opciones['paquetes'])
sel_tecno = st.sidebar.multiselect("📡 Tecnología", opciones['tecnologias'])

# E. Sliders
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Filtros Financieros")

val_facturado_range = st.sidebar.slider(
    "Valor Facturado (COP)",
    min_value=0.0,
    max_value=float(opciones['max_val_facturado']),
    value=(0.0, float(opciones['max_val_facturado'])),
    format="$%.0f"
)

otros_valores_range = st.sidebar.slider(
    "Otros Valores (COP)",
    min_value=0.0,
    max_value=float(opciones['max_otros']),
    value=(0.0, float(opciones['max_otros'])),
    format="$%.0f"
)

# NOTAS EN SIDEBAR
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ **Nota de Datos:**\nLos datos usados son los datos que no presentaron inconvenientes de consistencia.")
st.sidebar.success("👨‍💻 **Créditos:**\nDesarrollado por **Pedro Jose Leal Mesa**")

# ==========================================
# 5. APLICACIÓN DE FILTROS
# ==========================================

df_filtrado = df.filter(
    (pl.col("VALOR_FACTURADO_O_COBRADO") >= val_facturado_range[0]) & 
    (pl.col("VALOR_FACTURADO_O_COBRADO") <= val_facturado_range[1]) &
    (pl.col("OTROS_VALORES_FACTURADOS") >= otros_valores_range[0]) & 
    (pl.col("OTROS_VALORES_FACTURADOS") <= otros_valores_range[1])
)

if sel_ano: df_filtrado = df_filtrado.filter(pl.col("ANNO").is_in(sel_ano))
if sel_depto: df_filtrado = df_filtrado.filter(pl.col("DEPARTAMENTO").is_in(sel_depto))
if sel_muni: df_filtrado = df_filtrado.filter(pl.col("MUNICIPIO").is_in(sel_muni))
if sel_empresa: df_filtrado = df_filtrado.filter(pl.col("EMPRESA").is_in(sel_empresa))
if sel_paquete: df_filtrado = df_filtrado.filter(pl.col("SERVICIO_PAQUETE").is_in(sel_paquete))
if sel_tecno: df_filtrado = df_filtrado.filter(pl.col("TECNOLOGIA").is_in(sel_tecno))

# ==========================================
# 6. PESTAÑAS Y GRÁFICOS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 GENERAL", 
    "💰 FINANCIERO", 
    "📈 TENDENCIAS", 
    "📶 CONECTIVIDAD", 
    "🏆 COMPETENCIA",
    "🎯 SEGMENTACIÓN",
    "📍 GEOGRÁFICO"
])

# --------------------------------------------------------
# PESTAÑA 1: ANÁLISIS GENERAL
# --------------------------------------------------------
with tab1:
    st.markdown("### 🗺️ Panorama General de Registros")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Registros", f"{df_filtrado.height:,}")
    k2.metric("Departamentos", df_filtrado["DEPARTAMENTO"].n_unique())
    k3.metric("Municipios", df_filtrado["MUNICIPIO"].n_unique())
    k4.metric("Vel. Bajada Prom.", f"{df_filtrado['VELOCIDAD_EFECTIVA_DOWNSTREAM'].mean():.1f} Mbps")
    k5.metric("Empresas", df_filtrado["EMPRESA"].n_unique())

    row1_c1, row1_c2 = st.columns([3, 2])
    
    with row1_c1:
        st.info("🗺️ Distribución Geográfica de Registros")
        map_data = df_filtrado.group_by(["ID_DEPTO_MAPA", "DEPARTAMENTO"]).len().to_pandas()
        if not map_data.empty:
            fig_map = px.choropleth(
                map_data, geojson=geojson_colombia, locations='ID_DEPTO_MAPA',
                featureidkey='properties.DPTO', color='len', 
                color_continuous_scale="Viridis",
                hover_name="DEPARTAMENTO",
                labels={'len': 'Registros'}
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=450)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Sin datos para el mapa")

    with row1_c2:
        st.info("🏅 Top 10 Municipios")
        muni_data = df_filtrado.group_by("MUNICIPIO").len().sort("len", descending=True).head(10).to_pandas().sort_values("len", ascending=True)
        fig_muni = px.bar(muni_data, x='len', y='MUNICIPIO', orientation='h', 
                          color='len', color_continuous_scale='Teal',
                          labels={'len': 'Registros'})
        fig_muni.update_layout(showlegend=False)
        st.plotly_chart(fig_muni, use_container_width=True)

    row2_c1, row2_c2 = st.columns([1, 2])

    with row2_c1:
        st.info("📦 Mix de Servicios")
        serv_data = df_filtrado.group_by("SERVICIO_PAQUETE").len().to_pandas()
        fig_donut = px.pie(serv_data, values='len', names='SERVICIO_PAQUETE', hole=0.5)
        fig_donut.update_layout(
            height=400, 
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with row2_c2:
        st.info("👥 Registros por Segmento")
        seg_data = df_filtrado.group_by("SEGMENTO").len().sort("len").to_pandas()
        fig_seg = px.bar(seg_data, x='len', y='SEGMENTO', orientation='h', 
                        text_auto='.2s', color='len', color_continuous_scale='Blues')
        fig_seg.update_layout(showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 2: ANÁLISIS FINANCIERO
# --------------------------------------------------------
with tab2:
    st.markdown("### 💰 Comportamiento de Facturación")
    
    total_facturado = df_filtrado["VALOR_FACTURADO_O_COBRADO"].sum()
    total_otros = df_filtrado["OTROS_VALORES_FACTURADOS"].sum()
    total_general = df_filtrado["VALOR_TOTAL"].sum()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("💵 Valor Facturado", f"${total_facturado/1e9:,.2f}B")
    k2.metric("💳 Otros Valores", f"${total_otros/1e9:,.2f}B")
    k3.metric("💰 Total General", f"${total_general/1e9:,.2f}B")

    c1, c2 = st.columns(2)

    with c1:
        st.info("📊 Composición de Ingresos")
        comp_data = pl.DataFrame({
            "Tipo": ["Valor Facturado", "Otros Valores"],
            "Monto": [total_facturado, total_otros]
        }).to_pandas()
        fig_comp = px.pie(comp_data, values='Monto', names='Tipo', hole=0.4,
                          color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        st.plotly_chart(fig_comp, use_container_width=True)

    with c2:
        st.info("🎯 Valor Total por Paquete")
        val_paq = df_filtrado.group_by("SERVICIO_PAQUETE").agg(pl.col("VALOR_TOTAL").sum()).to_pandas()
        fig_tree = px.treemap(
            val_paq, 
            path=['SERVICIO_PAQUETE'], 
            values='VALOR_TOTAL',
            color='VALOR_TOTAL',
            color_continuous_scale='Greens'
        )
        fig_tree.update_traces(textinfo="label+value")
        st.plotly_chart(fig_tree, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.info("🏢 Top 10 Operadores - Total")
        val_op = df_filtrado.group_by("EMPRESA").agg(pl.col("VALOR_TOTAL").sum()).sort("VALOR_TOTAL", descending=True).head(10).to_pandas()
        fig_op_val = px.bar(
            val_op, 
            x='EMPRESA', 
            y='VALOR_TOTAL', 
            color='VALOR_TOTAL', 
            color_continuous_scale='YlGnBu',
            text_auto='.2s'
        )
        fig_op_val.update_layout(yaxis_title="Total (COP)", xaxis_title=None, showlegend=False)
        st.plotly_chart(fig_op_val, use_container_width=True)

    with c4:
        st.info("📡 Ingresos por Tecnología")
        val_tec = df_filtrado.group_by("TECNOLOGIA").agg(pl.col("VALOR_TOTAL").sum()).sort("VALOR_TOTAL", descending=True).to_pandas()
        fig_tec = px.bar(val_tec, x='TECNOLOGIA', y='VALOR_TOTAL', 
                        color='VALOR_TOTAL', color_continuous_scale='Reds',
                        text_auto='.2s')
        fig_tec.update_layout(showlegend=False)
        st.plotly_chart(fig_tec, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 3: TENDENCIAS
# --------------------------------------------------------
with tab3:
    st.markdown("### 📈 Evolución Temporal del Mercado")

    df_temp = df_filtrado.group_by("PERIODO").agg([
        pl.len().alias("REGISTROS"),
        pl.col("VALOR_TOTAL").sum(),
        pl.col("CANTIDAD_LINEAS_ACCESOS").sum()
    ]).sort("PERIODO").to_pandas()

    fig_main_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_main_trend.add_trace(
        go.Scatter(x=df_temp['PERIODO'], y=df_temp['VALOR_TOTAL'], 
                  name="Facturación ($)", line=dict(color='green', width=3)), 
        secondary_y=False
    )
    fig_main_trend.add_trace(
        go.Bar(x=df_temp['PERIODO'], y=df_temp['REGISTROS'], 
              name="Registros", opacity=0.3, marker_color='lightblue'), 
        secondary_y=True
    )
    fig_main_trend.update_layout(title="Evolución: Facturación vs Volumen", height=450)
    st.plotly_chart(fig_main_trend, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.caption("📡 Evolución Tecnologías (% Market Share)")
        tec_trend = df_filtrado.group_by(["PERIODO", "TECNOLOGIA"]).len().sort("PERIODO").to_pandas()
        fig_area_tec = px.area(tec_trend, x="PERIODO", y="len", color="TECNOLOGIA", groupnorm='percent')
        st.plotly_chart(fig_area_tec, use_container_width=True)

    with c2:
        st.caption("📦 Popularidad de Paquetes")
        paq_trend = df_filtrado.group_by(["PERIODO", "SERVICIO_PAQUETE"]).len().sort("PERIODO").to_pandas()
        fig_line_paq = px.line(paq_trend, x="PERIODO", y="len", color="SERVICIO_PAQUETE", markers=True)
        st.plotly_chart(fig_line_paq, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.caption("⚡ Velocidad Bajada Promedio")
        vel_trend = df_filtrado.group_by("PERIODO").agg(pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").mean()).sort("PERIODO").to_pandas()
        fig_vel = px.line(vel_trend, x="PERIODO", y="VELOCIDAD_EFECTIVA_DOWNSTREAM", 
                         markers=True, line_shape='spline')
        st.plotly_chart(fig_vel, use_container_width=True)

    with c4:
        st.caption("🔥 Intensidad Top 5 Operadores")
        top5_ops = df_filtrado.group_by("EMPRESA").len().sort("len", descending=True).head(5)["EMPRESA"].to_list()
        heat_data = df_filtrado.filter(pl.col("EMPRESA").is_in(top5_ops)).group_by(["PERIODO", "EMPRESA"]).len().sort("PERIODO").to_pandas()
        fig_heat = px.density_heatmap(heat_data, x="PERIODO", y="EMPRESA", z="len", 
                                      color_continuous_scale="YlOrRd")
        st.plotly_chart(fig_heat, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 4: CONECTIVIDAD
# --------------------------------------------------------
with tab4:
    st.markdown("### 📶 Detalles de Conectividad")
    total_lineas = df_filtrado["CANTIDAD_LINEAS_ACCESOS"].sum()
    vel_down_prom = df_filtrado["VELOCIDAD_EFECTIVA_DOWNSTREAM"].mean()
    vel_up_prom = df_filtrado["VELOCIDAD_EFECTIVA_UPSTREAM"].mean()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("📱 Total Líneas/Accesos", f"{total_lineas:,.0f}")
    k2.metric("⬇️ Vel. Bajada Prom.", f"{vel_down_prom:.1f} Mbps")
    k3.metric("⬆️ Vel. Subida Prom.", f"{vel_up_prom:.1f} Mbps")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("📡 Líneas por Tecnología")
        lin_tec = df_filtrado.group_by("TECNOLOGIA").agg(pl.col("CANTIDAD_LINEAS_ACCESOS").sum()).to_pandas()
        fig_lin_tec = px.pie(lin_tec, values="CANTIDAD_LINEAS_ACCESOS", names="TECNOLOGIA", hole=0.4)
        st.plotly_chart(fig_lin_tec, use_container_width=True)

    with c2:
        st.caption("👥 Líneas por Segmento")
        lin_seg = df_filtrado.group_by("SEGMENTO").agg(pl.col("CANTIDAD_LINEAS_ACCESOS").sum()).sort("CANTIDAD_LINEAS_ACCESOS", descending=True).to_pandas()
        fig_lin_seg = px.bar(lin_seg, x="SEGMENTO", y="CANTIDAD_LINEAS_ACCESOS", 
                            color="CANTIDAD_LINEAS_ACCESOS", text_auto='.2s',
                            color_continuous_scale='Purples')
        fig_lin_seg.update_layout(showlegend=False)
        st.plotly_chart(fig_lin_seg, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("🏆 Top 10 Deptos - Mejor Velocidad")
        vel_depto = df_filtrado.group_by("DEPARTAMENTO").agg(pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").mean()).sort("VELOCIDAD_EFECTIVA_DOWNSTREAM", descending=True).head(10).to_pandas()
        fig_vel_dep = px.bar(vel_depto, x="VELOCIDAD_EFECTIVA_DOWNSTREAM", y="DEPARTAMENTO", 
                            orientation='h', color='VELOCIDAD_EFECTIVA_DOWNSTREAM',
                            color_continuous_scale='Teal')
        fig_vel_dep.update_layout(showlegend=False)
        st.plotly_chart(fig_vel_dep, use_container_width=True)

    with c4:
        st.caption("⚖️ Simetría Down vs Up")
        df_sample_vel = df_filtrado.sample(n=min(1000, df_filtrado.height), seed=1).to_pandas()
        fig_scat_vel = px.scatter(df_sample_vel, x="VELOCIDAD_EFECTIVA_DOWNSTREAM", 
                                  y="VELOCIDAD_EFECTIVA_UPSTREAM", color="TECNOLOGIA",
                                  opacity=0.6)
        st.plotly_chart(fig_scat_vel, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 5: COMPETENCIA
# --------------------------------------------------------
with tab5:
    st.markdown("### 🏆 Competencia y Mercado")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("💰 Market Share (Ingresos)")
        share_val = df_filtrado.group_by("EMPRESA").agg(pl.col("VALOR_TOTAL").sum()).sort("VALOR_TOTAL", descending=True).head(8).to_pandas()
        fig_share1 = px.pie(share_val, values="VALOR_TOTAL", names="EMPRESA", hole=0.5)
        st.plotly_chart(fig_share1, use_container_width=True)

    with c2:
        st.caption("📊 Market Share (Volumen)")
        share_vol = df_filtrado.group_by("EMPRESA").len().sort("len", descending=True).head(8).to_pandas()
        fig_share2 = px.pie(share_vol, values="len", names="EMPRESA", hole=0.5)
        st.plotly_chart(fig_share2, use_container_width=True)

    st.caption("👑 Operador Líder por Departamento")
    dom_op = (
        df_filtrado
        .group_by(["DEPARTAMENTO", "EMPRESA"])
        .len()
        .sort("len", descending=True)
        .group_by("DEPARTAMENTO")
        .first()
        .to_pandas()
    )
    fig_dom = px.bar(dom_op, x="DEPARTAMENTO", y="len", color="EMPRESA",
                    text='EMPRESA')
    fig_dom.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_dom, use_container_width=True)

    st.caption("🔧 Mix Tecnológico - Top Jugadores")
    top10_ops_list = share_vol["EMPRESA"].to_list()
    div_op = df_filtrado.filter(pl.col("EMPRESA").is_in(top10_ops_list)).group_by(["EMPRESA", "TECNOLOGIA"]).len().to_pandas()
    fig_div = px.bar(div_op, x="EMPRESA", y="len", color="TECNOLOGIA", text_auto=True)
    fig_div.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_div, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 6: SEGMENTACIÓN
# --------------------------------------------------------
with tab6:
    st.markdown("### 🎯 Análisis por Segmento y Estrato")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info("📊 Distribución de Registros por Segmento")
        seg_dist = df_filtrado.group_by("SEGMENTO").len().sort("len", descending=True).to_pandas()
        fig_seg_dist = px.bar(seg_dist, x='SEGMENTO', y='len', 
                             color='len', color_continuous_scale='Blues',
                             text_auto='.2s')
        fig_seg_dist.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_seg_dist, use_container_width=True)
    
    with c2:
        st.info("💰 Ingresos por Segmento")
        seg_val = df_filtrado.group_by("SEGMENTO").agg(pl.col("VALOR_TOTAL").sum()).sort("VALOR_TOTAL", descending=True).to_pandas()
        fig_seg_val = px.pie(seg_val, values='VALOR_TOTAL', names='SEGMENTO')
        st.plotly_chart(fig_seg_val, use_container_width=True)
    
    c3, c4 = st.columns(2)
    
    with c3:
        st.info("📡 Tecnología Preferida por Segmento")
        seg_tec = df_filtrado.group_by(["SEGMENTO", "TECNOLOGIA"]).len().to_pandas()
        fig_seg_tec = px.sunburst(
            seg_tec, 
            path=['SEGMENTO', 'TECNOLOGIA'], 
            values='len',
            color='len',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_seg_tec, use_container_width=True)

    with c4:
        st.info("⚡ Velocidad Promedio (Mbps) por Segmento")
        seg_vel = df_filtrado.group_by("SEGMENTO").agg(
            pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").mean()
        ).sort("VELOCIDAD_EFECTIVA_DOWNSTREAM", descending=True).to_pandas()
        
        fig_seg_vel = px.bar(
            seg_vel, 
            x='SEGMENTO', 
            y='VELOCIDAD_EFECTIVA_DOWNSTREAM',
            color='VELOCIDAD_EFECTIVA_DOWNSTREAM',
            color_continuous_scale='Viridis',
            text_auto='.1f'
        )
        fig_seg_vel.update_layout(yaxis_title="Mbps", showlegend=False)
        st.plotly_chart(fig_seg_vel, use_container_width=True)

# --------------------------------------------------------
# PESTAÑA 7: GEOGRÁFICO
# --------------------------------------------------------
with tab7:
    st.markdown("### 📍 Análisis Geográfico Detallado")

    col_geo1, col_geo2 = st.columns([2, 1])

    with col_geo1:
        st.markdown("#### 🗺️ Mapa de Calor: Ingresos por Departamento")
        map_rev_data = df_filtrado.group_by(["ID_DEPTO_MAPA", "DEPARTAMENTO"]).agg(
            pl.col("VALOR_TOTAL").sum()
        ).to_pandas()

        if not map_rev_data.empty:
            fig_map_rev = px.choropleth(
                map_rev_data,
                geojson=geojson_colombia,
                locations='ID_DEPTO_MAPA',
                featureidkey='properties.DPTO',
                color='VALOR_TOTAL',
                color_continuous_scale="Inferno",
                hover_name="DEPARTAMENTO",
                title="Ingresos Totales (COP)",
                labels={'VALOR_TOTAL': 'Ingresos'}
            )
            fig_map_rev.update_geos(fitbounds="locations", visible=False)
            fig_map_rev.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0}, 
                height=500,
                coloraxis_colorbar=dict(title="COP")
            )
            st.plotly_chart(fig_map_rev, use_container_width=True)
        else:
            st.warning("No hay datos suficientes para generar el mapa de ingresos.")

    with col_geo2:
        st.markdown("#### 🏆 Top 10 Municipios por Ingresos")
        top_munis = df_filtrado.group_by("MUNICIPIO").agg(
            pl.col("VALOR_TOTAL").sum()
        ).sort("VALOR_TOTAL", descending=True).head(10).to_pandas()

        st.dataframe(
            top_munis.style.format({"VALOR_TOTAL": "${:,.0f}"}).background_gradient(cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 📉 Municipios con Menor Conectividad")
        low_speed_munis = df_filtrado.group_by("MUNICIPIO").agg(
            pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").mean()
        ).sort("VELOCIDAD_EFECTIVA_DOWNSTREAM").head(10).to_pandas()

        st.dataframe(
            low_speed_munis.style.format({"VELOCIDAD_EFECTIVA_DOWNSTREAM": "{:.2f} Mbps"}),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### 📋 Tabla de Datos Agregada")
    
    with st.expander("Ver Tabla Detallada por Departamento y Municipio"):
        tabla_resumen = df_filtrado.group_by(["DEPARTAMENTO", "MUNICIPIO"]).agg([
            pl.len().alias("TOTAL_REGISTROS"),
            pl.col("VALOR_TOTAL").sum().alias("TOTAL_INGRESOS"),
            pl.col("VELOCIDAD_EFECTIVA_DOWNSTREAM").mean().alias("VEL_BAJADA_PROM"),
            pl.col("CANTIDAD_LINEAS_ACCESOS").sum().alias("TOTAL_ACCESOS")
        ]).sort("TOTAL_INGRESOS", descending=True).to_pandas()

        st.dataframe(
            tabla_resumen,
            use_container_width=True,
            column_config={
                "TOTAL_INGRESOS": st.column_config.NumberColumn(format="$%.0f"),
                "VEL_BAJADA_PROM": st.column_config.NumberColumn(format="%.2f Mbps"),
                "TOTAL_ACCESOS": st.column_config.NumberColumn(format="%.0f")
            }
        )

# ==========================================
# FOOTER / NOTAS FINALES
# ==========================================
st.markdown("""
<div class="footer">
    <p>📊 <b>Monitor CRC - Análisis 360</b></p>
    <p>👨‍💻 Desarrollado por <b>Pedro Jose Leal Mesa</b></p>
    <p>ℹ️ <i>Nota: Los datos usados son los datos que no presentaron inconvenientes de consistencia.</i></p>
</div>

""", unsafe_allow_html=True)
