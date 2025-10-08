import pandas as pd
from unidecode import unidecode
import plotly.graph_objects as go
import os, base64, mimetypes
import numpy as np
import streamlit as st
from typing import Optional  # <-- agrégalo al inicio del archivo

USERS_CSV = "users.csv"

@st.cache_data
def _load_users(path=USERS_CSV):
    return pd.read_csv(path)

def validar_usuario(usuario: str, clave: str) -> bool:
    df = _load_users()
    return bool(len(df[(df['usuario'] == usuario) & (df['clave'] == clave)]) > 0)

def get_nombre(usuario: str) -> str:
    df = _load_users()
    fila = df[df['usuario'] == usuario]
    return fila['nombre'].values[0] if not fila.empty else usuario

def login(logo_data_uri: Optional[str] = None, titulo: str = "Iniciar sesión") -> bool:
    if 'usuario' in st.session_state:
        return True
    with st.container(border=True):
        col_logo, col_title = st.columns([1, 5])
        with col_logo:
            if logo_data_uri:
                st.image(logo_data_uri, width=56)
        with col_title:
            st.markdown(f"### {titulo}")

        with st.form('frmLogin'):
            u = st.text_input('Usuario')
            p = st.text_input('Password', type='password')
            ok = st.form_submit_button('Ingresar', type='primary')
            if ok:
                if validar_usuario(u, p):
                    st.session_state['usuario'] = u
                    st.rerun()
                else:
                    st.error("Usuario o clave inválidos", icon=":material/gpp_maybe:")
    return False

def user_header():
    """Muestra saludo y botón salir en la parte superior del dashboard."""
    if 'usuario' not in st.session_state:
        return
    nombre = get_nombre(st.session_state['usuario'])
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"👋 Bienvenido **{nombre}**")
    with col2:
        salir = st.button("Salir", use_container_width=True)
        if salir:
            st.session_state.clear()
            st.rerun()



# ================== UTILIDADES DE DATOS ==================
def _find_section(df, needle: str):
    """Devuelve el nombre exacto de la sección que contenga 'needle' (en minúsculas)."""
    needle = needle.lower()
    for s in df["Seccion"].dropna().unique():
        if needle in str(s).lower():
            return s
    return None

def render_valoracion(df, df_per, seccion_label: str, titulo_bloque: str):
    """Pinta Resultados (barras) + Evolución (línea) con el estilo actual."""
    st.markdown(f"#### {titulo_bloque}")

    # --- Resultados del periodo seleccionado ---
    df_val_per = df_per[df_per["Seccion"] == seccion_label].copy()
    if df_val_per.empty:
        fig_resultados = px.bar(pd.DataFrame({"Respuesta": [], "Porcentaje": []}),
                                x="Porcentaje", y="Respuesta")
    else:
        normalizador = limpiar_txt if 'limpiar_txt' in globals() else lambda s: str(s).strip().lower()
        df_val_per["item_norm"] = df_val_per["Item"].map(lambda s: normalizador(s) if pd.notna(s) else None)
        mapa = {
            "positivo":"Positivos","positiva":"Positivos","positivos":"Positivos","positivas":"Positivos",
            "neutral":"Neutros","neutra":"Neutros","neutros":"Neutros","neutras":"Neutros",
            "negativo":"Negativos","negativa":"Negativos","negativos":"Negativos","negativas":"Negativos",
        }
        df_val_per["grupo"] = df_val_per["item_norm"].map(mapa)
        res = (df_val_per.dropna(subset=["grupo"]).groupby("grupo", as_index=False)["Valor"].sum())
        res = (res.set_index("grupo").reindex(["Positivos","Neutros","Negativos"]).fillna(0)
                    .reset_index().rename(columns={"grupo":"Respuesta","Valor":"Porcentaje"}))
        res["Porcentaje"] = pd.to_numeric(res["Porcentaje"], errors="coerce").fillna(0)

        fig_resultados = px.bar(
            res, x="Porcentaje", y="Respuesta", orientation="h",
            color="Respuesta",
            color_discrete_map={"Positivos": PRIMARY, "Neutros": "#8FBAD3", "Negativos": "#F07B7B"},
            text="Porcentaje"
        )
        fig_resultados.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig_resultados.update_layout(
            height=275, showlegend=False, margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title=None, yaxis_title=None
        )
        fig_resultados.update_xaxes(ticksuffix="%")

def limpiar_txt(s):
    """Minúsculas, sin acentos y sin dobles espacios, para comparar secciones/items robustamente."""
    s = unidecode(str(s)).strip().lower()
    s = " ".join(s.split())
    return s

def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64  = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def cargar_csv(path):
    df = pd.read_csv(path)
    # Normaliza columnas por si vienen con espacios/acentos
    for c in ["Seccion","Item","Unidad","Fecha","Valor","Periodo"]:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: {c}")
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df["Periodo"] = pd.to_datetime(df["Periodo"], errors="coerce")  # YYYY-MM
    df["fecha_label"] = df["Fecha"].astype(str)  # ej. "Sep-25"
    # Versiones limpias para filtrar
    df["seccion_norm"] = df["Seccion"].map(limpiar_txt)
    df["item_norm"]    = df["Item"].map(limpiar_txt)
    return df

def pick_valoracion_sections(df):
    """
    Devuelve el nombre 'original' de la sección de valoración a usar.
    Acepta variantes: 'Valoracion/Valoración ... gobierno/presidente ...'
    """
    cand = [
        "valoracion de la gestion del gobierno 2023-2025",
        "valoracion de la gestion del presidente 2023-2025",
        "valoracion de la gestion del gobierno",
        "valoracion de la gestion del presidente",
    ]
    map_norm_to_orig = {limpiar_txt(s): s for s in df["Seccion"].unique()}
    for c in cand:
        for k,orig in map_norm_to_orig.items():
            if c in k:
                return orig
    # si no hay, None
    return None

def items_alias_pos_neu_neg():
    # Alias posibles de columnas Positiva/Neutral/Negativa
    return {
        "pos": {"positiva","positivas","positivo","positivos","pos"},
        "neu": {"neutral","neutrales","neutro","neutros","neu"},
        "neg": {"negativa","negativas","negativo","negativos","neg"},
        # del CSV: Positivos/Neutros/Negativos
        "pos_csv": {"positivos"}, "neu_csv": {"neutros"}, "neg_csv": {"negativos"},
    }

def filtrar_periodo(df, periodo_dt):
    return df.loc[df["Periodo"]==periodo_dt].copy()

def norm(s: str) -> str:
    return unidecode(str(s)).strip().lower()


def val_from_items(df_sec):
    """extrae valores de Positiva/Neutral/Negativa sin importar alias exacto."""
    val_pos = val_neu = val_neg = np.nan
    for _,r in df_sec.iterrows():
        it = r["item_norm"]
        if it == 'Positiva':
            val_pos = r["Valor"]
        elif it == 'Neutral':
            val_neu = r["Valor"]
        elif it == 'Negativa':
            val_neg = r["Valor"]
    return val_pos, val_neu, val_neg


# KPI helper
def kpi_box(col, titulo, valor, delta=None):
    with col:
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.markdown(f'<div class="box-title">{titulo}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{"" if pd.isna(valor) else f"{valor:.1f}%"}''</div>', unsafe_allow_html=True)
        if delta is not None and not pd.isna(delta):
            st.markdown(f'<div class="kpi-delta">Δ {delta:+.1f} pp</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
   

def bar_100_stacked(df_long, x_col, y_col, color_col, color_order=None, color_map=None,
                    height=300, angle=-30, legend_y=-0.28, y_title="%"):
    """
    df_long: columnas [x_col, y_col, color_col] en LONG
    Apila en 100% por cada valor de x_col.
    """
    # agrega por si hay duplicados por periodo-categoría
    data = (df_long.groupby([x_col, color_col], as_index=False)[y_col]
                  .sum())
    x_vals = list(data[x_col].drop_duplicates())
    series = list(color_order or data[color_col].drop_duplicates())

    fig = go.Figure()
    for s in series:
        sub = data[data[color_col] == s]
        y = [float(sub.loc[sub[x_col] == x, y_col].sum()) for x in x_vals]
        fig.add_bar(name=s, x=x_vals, y=y,
                    marker_color=(color_map or {}).get(s, None))

    fig.update_layout(
        template="plotly",
        barmode="relative",     # <- apilado
        barnorm="percent",      # <- 100%
        height=height,
        margin=dict(l=0, r=0, t=10, b=40),
        xaxis_title=None,
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="top", y=legend_y,
                    xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(type="category", tickangle=angle)
    return fig
