# app.py
import os
import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
import streamlit as st
import json, io, base64
from PIL import Image
from funciones import *    
import streamlit.components.v1 as components

# ================== CONFIG ==================
st.set_page_config(
    page_title="Pérdida de vegetación en Darién",
    page_icon="circle-white.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Rutas ----
LOGO_PATH = "circle-white.svg"
LANDCOVER_PATH = "landcover_darien.tif"
dirpath = "mask_loss/"

# ---- Iconos UI ----
logo_data_uri = img_to_data_uri(LOGO_PATH)
icon_prev  = img_to_data_uri("previous-svgrepo-com.svg")
icon_play  = img_to_data_uri("play-svgrepo-com.svg")
icon_pause = img_to_data_uri("pause-svgrepo-com.svg")
icon_next  = img_to_data_uri("next-svgrepo-com.svg")

# ================== DATA SOURCES ==================
RASTERS = {
    "2020 → 2021": f"{dirpath}Mask_Loss_2020_2021_adaptive.tif",
    "2021 → 2022": f"{dirpath}Mask_Loss_2021_2022_adaptive.tif",
    "2022 → 2023": f"{dirpath}Mask_Loss_2022_2023_adaptive.tif",
    "2023 → 2024": f"{dirpath}Mask_Loss_2023_2024_adaptive.tif",
    "2024 → 2025": f"{dirpath}Mask_Loss_2024_2025_adaptive.tif",
}
LABELS = list(RASTERS.keys())
MAX_PIXELS = 5_000_000  # controla submuestreo para fluidez

# Paleta Copernicus + utilidades
LANDCOVER_CLASSES = [
    (0,   "#282828", "Desconocido"),
    (20,  "#ffbb22", "Arbustos"),
    (30,  "#84F58C", "Vegetación herbácea"),
    (40,  "#EBEB86", "Cultivos / agricultura"),
    (50,  "#b727f5", "Urbano / construido"),
    (60,  "#b4b4b4", "Desnudo / vegetación escasa"),
    (70,  "#f0f0f0", "Nieve y hielo"),
    (80,  "#0032c8", "Cuerpos de agua permanentes"),
    (90,  "#0096a0", "Humedal herbáceo"),
    (100, "#fae6a0", "Musgo y líquenes"),
    (111, "#58481f", "Bosque cerrado, coníferas perennes"),
    (112, "#009900", "Bosque cerrado, hoja perenne de amplio espectro"),
    (113, "#70663e", "Bosque cerrado, hoja caduca de aguja"),
    (114, "#00cc00", "Bosque cerrado, hoja caduca de amplio espectro"),
    (115, "#4e751f", "Bosque cerrado, mixto"),
    (116, "#007800", "Bosque cerrado, otro"),
    (121, "#666000", "Bosque abierto, coníferas perennes"),
    (122, "#8db400", "Bosque abierto, hoja perenne de amplio espectro"),
    (123, "#8d7400", "Bosque abierto, hoja caduca de aguja"),
    (124, "#a0dc00", "Bosque abierto, hoja caduca de amplio espectro"),
    (125, "#929900", "Bosque abierto, mixto"),
    (126, "#648c00", "Bosque abierto, otro"),
    (200, "#000080", "Océanos, mares"),
]

# ================== STATE (solo lo que usas) ==================
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "interval" not in st.session_state:
    st.session_state.interval = 0.6


# ================== UTILS ==================
def _hex_to_rgb(h: str):
    h = h.strip()
    if not h.startswith("#"):
        h = "#" + h
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def rgba_to_dataurl(rgba: np.ndarray) -> str:
    im = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

@st.cache_data(show_spinner=False)
def load_any_as_rgba_and_bounds(path, max_pixels=MAX_PIXELS):
    """
    GeoTIFF → RGBA + bounds (S,W,N,E), reproyectado a EPSG:4326, submuestreo por stride.
    1 banda → máscara >0 en rojo; 3/4 bandas → respeta RGB(A).
    """
    with rasterio.open(path) as src:
        if src.count >= 3:
            r = src.read(1); g = src.read(2); b = src.read(3)
            a = np.full_like(r, 255, dtype=np.uint8)
            if src.count >= 4:
                a = src.read(4).astype(np.uint8)
            base = np.dstack([r, g, b, a]).astype(np.uint8)
        else:
            m = src.read(1).astype(float)
            if src.nodata is not None:
                m[m == src.nodata] = np.nan
            mask = ~np.isnan(m) & (m > 0)
            base = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
            base[..., 0][mask] = 255
            base[..., 1][mask] = 59
            base[..., 2][mask] = 48
            base[..., 3][mask] = 255

        if src.crs is None or (hasattr(src.crs, "is_geographic") and src.crs.is_geographic):
            arr = base; transform = src.transform
        else:
            dst_crs = "EPSG:4326"
            T, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            for i in range(4):
                reproject(
                    source=base[..., i], destination=arr[..., i],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=T, dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
            transform = T

        H, W = arr.shape[:2]
        step = int(np.ceil(np.sqrt((H * W) / max_pixels))) if H * W > max_pixels else 1
        if step > 1:
            arr = arr[::step, ::step, :].copy()
            transform = rasterio.Affine(
                transform.a * step, transform.b, transform.c,
                transform.d, transform.e * step, transform.f
            )

        left, top = transform.c, transform.f
        right = left + transform.a * arr.shape[1]
        bottom = top + transform.e * arr.shape[0]
        s, w, n, e = bottom, left, top, right
        return arr, (s, w, n, e)

@st.cache_data(show_spinner=False)
def load_landcover_rgba_and_bounds(path, max_pixels=MAX_PIXELS):
    """
    TIFF categórico → RGBA por LUT + bounds (S,W,N,E) + arr códigos (H,W).
    """
    with rasterio.open(path) as src:
        band = src.read(1); nodata = src.nodata

        if src.crs is None or (hasattr(src.crs, "is_geographic") and src.crs.is_geographic):
            arr = band; transform = src.transform
        else:
            dst_crs = "EPSG:4326"
            T, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            arr = np.empty((h, w), dtype=band.dtype)
            reproject(
                source=band, destination=arr,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=T, dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            transform = T

        H, W = arr.shape
        step = int(np.ceil(np.sqrt((H * W) / max_pixels))) if H * W > max_pixels else 1
        if step > 1:
            arr = arr[::step, ::step]
            transform = rasterio.Affine(
                transform.a * step, transform.b, transform.c,
                transform.d, transform.e * step, transform.f
            )

        left, top = transform.c, transform.f
        right = left + transform.a * arr.shape[1]
        bottom = top + transform.e * arr.shape[0]
        s, w, n, e = bottom, left, top, right

        m = arr.astype(float)
        if nodata is not None:
            m[m == nodata] = np.nan
        valid = ~np.isnan(m)

        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        present_codes = set()
        for code, hexcolor, _label in LANDCOVER_CLASSES:
            mask = (arr == code) & valid
            if np.any(mask):
                r, g, b = _hex_to_rgb(hexcolor)
                rgba[..., 0][mask] = r
                rgba[..., 1][mask] = g
                rgba[..., 2][mask] = b
                rgba[..., 3][mask] = 255
                present_codes.add(code)

        legend_present = [
            {"code": code, "label": label, "color": color}
            for (code, color, label) in LANDCOVER_CLASSES
            if code in present_codes
        ]

        return rgba, (s, w, n, e), legend_present, arr

# === Land cover (opcional) ===
if os.path.exists(LANDCOVER_PATH):
    LC_rgba, (LC_s, LC_w, LC_n, LC_e), LC_LEGEND, LC_CODES = load_landcover_rgba_and_bounds(LANDCOVER_PATH)
    LC_img = rgba_to_dataurl(LC_rgba)
    LC_BOUNDS = [LC_w, LC_s, LC_e, LC_n]  # [W,S,E,N] para JS
    LC_H, LC_WID = LC_CODES.shape
    LC_CODES_B64 = base64.b64encode(LC_CODES.astype(np.uint16).tobytes()).decode("ascii")
else:
    LC_img = None; LC_BOUNDS = None; LC_LEGEND = []; LC_CODES_B64 = None; LC_H = LC_WID = 0



#  ================== ESTILOS (UN SOLO BLOQUE) ==================
st.markdown(f"""
<style>
/* ===== Tipografía: local (./static) con fallback a Google ===== */
@font-face {{
  font-family: 'PoppinsLocal';
  src: url('Poppins-Regular.woff2') format('woff2'),
       url('Poppins-Regular.ttf') format('truetype');
  font-weight: 200;
  font-style: normal;
  font-display: swap;
}}
[data-testid="stAppViewContainer"] * {{
  font-family: 'PoppinsLocal','Poppins',sans-serif !important;
}}

/* ===== Fondo y contenedor principal ===== */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(90deg, #175CA1, #07A9E0 140%);
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{
  background: transparent;
  box-shadow: none;
}}

/* ===== Cabecera (logo + título) ===== */
.header-row {{ display:flex; align-items:center; gap:12px; }}
.header-row h1 {{ margin:0; font-size:25px; font-weight:400; color:#fff; }}
.header-row img {{ height:5vh; width:auto; }}

/* ===== Controles player debajo del mapa ===== */
.player-btn {{
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px 6px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}}
.player-btn:hover {{ background: rgba(255,255,255,.08); }}
.player-btn img {{
  width: 8px;
  height: 8px;
  pointer-events: none;
  filter: invert(1); /* iconos blancos sobre fondo oscuro */
}}

.controls {{
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: nowrap;
  width: 100%;
}}
.controls input[type="range"] {{ flex:1 1 auto; max-width:none; }}
#label {{
  min-width: 120px;          /* ancho fijo razonable para la etiqueta */
  text-align: center;
  color: #fff;
  font-size:5vh;
  font-weight: 600;
}}

/* Slider consistente */
#slider {{
  -webkit-appearance: none;
  appearance: none;
  height: 8px;
  border-radius: 6px;
  background: rgba(255,255,255,.35);
  outline: none;
  margin: 0;
}}
#slider::-webkit-slider-thumb {{
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #87090E;
  border: 2px solid #fff;
}}
#slider::-moz-range-thumb {{
  width: 18px; height: 18px; border-radius: 50%;
  background: #87090E; border: 2px solid #fff;
}}
.leaflet-control-attribution {{ display: none !important; }} /* ocultar créditos */

/* ===== Mapa + contenedor ===== */
.map-wrapper {{ display:flex; flex-direction:column; gap:10px; width:100%; }}
#map {{ height: 70vh; width:100%; border-radius:12px; overflow:hidden; }}
/* Evitar suavizado al reamostrar */
.leaflet-image-layer, .leaflet-tile, .leaflet-overlay-pane img {{
  image-rendering: pixelated !important;
  image-rendering: crisp-edges !important;
}}


/* ===== Ajustes generales ===== */
.block-container label:empty {{ margin:0; padding:0; }}
.main .block-container {{ padding-top: 1.2rem; }}
footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ display:none !important; }}
MainMenu {{ visibility: hidden; }}
main blockquote, .block-container {{ padding-top: 0.6rem; padding-bottom: 0.6rem; }}
html, body, [data-testid="stAppViewContainer"] {{ height: 100%; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ================== CABECERA ==================
st.markdown(
    f"""
    <div class="header-box">
      <div class="header-row">
        <img src="{logo_data_uri}" alt="TDP Logo" />
        <h1>Perdida de vegetación en Darién</h1>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# if st.button("🔄 Refrescar land cover"):
#     st.cache_data.clear()

# ================== CARGA DE FRAMES ==================
ALL = {}
for k, p in RASTERS.items():
    if not os.path.exists(p):
        st.error(f"No existe el archivo: {p}")
        st.stop()
    ALL[k] = load_any_as_rgba_and_bounds(p)

# Envolvente global con frames y (si existe) land cover
bounds_list = [ALL[label][1] for label in LABELS]  # (S,W,N,E)
S = min(s for (s, w, n, e) in bounds_list)
W = min(w for (s, w, n, e) in bounds_list)
N = max(n for (s, w, n, e) in bounds_list)
E = max(e for (s, w, n, e) in bounds_list)
if LC_BOUNDS:
    W = min(W, LC_BOUNDS[0]); S = min(S, LC_BOUNDS[1])
    E = max(E, LC_BOUNDS[2]); N = max(N, LC_BOUNDS[3])
GLOBAL_BOUNDS = [[S, W], [N, E]]

# Frames para JS
frames = []
for label in LABELS:
    rgba_i, (s_i, w_i, n_i, e_i) = ALL[label]
    frames.append({
        "label": label,
        "img": rgba_to_dataurl(rgba_i),
        "bounds": [w_i, s_i, e_i, n_i],  # [W,S,E,N]
    })
<style>

html = f"""
<style>
  @font-face {{
    font-family: 'PoppinsLocal';
    src: url('Poppins-Regular.woff2') format('woff2'),
         url('Poppins-Regular.ttf') format('truetype');
    font-weight: 400;
    font-style: normal;
  }}
  :root {{
    --fs-base: clamp(12px, 2vh, 16px);
    --fs-small: clamp(11px, 2vh, 14px);
    --fs-h3: clamp(13px, 2vh, 16px);
    --fs-h1: clamp(18px, 3.2vh, 28px);
    --icon: clamp(16px, 2.2vh, 22px);
  }}
  html, body, #map {{
    font-family: 'PoppinsLocal','Poppins',sans-serif !important;
    font-size: 2vh;
  }}

  /* ===== Mapa + controles ===== */
  .map-wrapper {{
    display: flex; flex-direction: column; gap: 10px; width: 100%;
  }}
  #map {{
    height: 100vh; width: 100%; border-radius: 12px; overflow: hidden;
  }}
  .controls {{
    display:flex; align-items:center; gap:10px; flex-wrap:nowrap; width:100%;
    position: relative; z-index: 1100; /* por encima del mapa */
  }}
  .controls input[type="range"] {{ flex: 1 1 auto; max-width: none; }}
  #label {{
    min-width: 120px; text-align: center; color: #fff;
    font-size: 2.5vh; font-weight: 600;
  }}
  .player-btn {{
    background:none;border:none;cursor:pointer;padding:4px 6px;
    display:inline-flex;align-items:center;justify-content:center;
    width:34px;height:34px;border-radius:8px;
  }}
  .player-btn:hover {{ background: rgba(255,255,255,.08); }}
  .player-btn img {{ width:22px;height:22px;filter:invert(1);display:block; pointer-events: none; }}

  /* Evitar suavizado al reamostrar */
  .leaflet-image-layer, .leaflet-tile, .leaflet-overlay-pane img {{
    image-rendering: pixelated !important;
    image-rendering: crisp-edges !important;
  }}

  /* ===== Leyenda Land Cover ===== */
  .leaflet-control.lc-legend, .leaflet-control.lc-legend * {{
    font-family: 'PoppinsLocal','Poppins',sans-serif !important;
    box-sizing: border-box;
  }}
  .leaflet-control.lc-legend {{
    background: rgba(255,255,255,0.7) !important;
    color: #1f2937 !important;
    border: 1px solid rgba(0,0,0,.08);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,.18);
    line-height: 1.25;
    max-height: 300px;
    max-width: min(50vw, 380px);
    overflow-y: auto;
    backdrop-filter: saturate(120%) blur(2px);
  }}
  .lc-legend .ttl {{
    margin: 0 0 6px 0;
    font-weight: 700;
    font-size: 1.5vh !important;
    color: #0f172a !important;
  }}
  .lc-legend .row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
  }}
  .lc-legend .swatch {{
    width: 14px; height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(0,0,0,.2);
    flex-shrink: 0;
  }}
  .lc-legend .row span {{
    font-size: 1.5vh !important;
    color: #1f2937 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .lc-legend::-webkit-scrollbar {{ width: 6px; }}
  .lc-legend::-webkit-scrollbar-thumb {{
    background: rgba(0,0,0,.75);
    border-radius: 3px;
  }}

  /* ===== Panel flotante dentro del mapa (Leaflet control) ===== */
  .panel-control {{
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,.08);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,.18);
    min-width: 50vh;
    font-size: 2.5vh;
    backdrop-filter: saturate(120%) blur(2px);
  }}
  /* Estilo del número de opacidad */
  #loss-opacity-val {{
    font-size:2.5vh;     /* ← tamaño relativo; puedes usar 14px, 1.2em, etc. */
    color: #0f172a;       /* color del texto */
    display: inline-block;
    margin-top: 4px;
  }}
  .panel-control h3 {{
    margin: 0 0 8px 0;
    font-size: 2.5vh;
    font-weight: 600;
    color: #1f2937;
  }}
  .panel-control .section {{ margin-bottom: 10px; }}
  .panel-control .section:last-child {{ margin-bottom: 0; }}
  .panel-control .control-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }}
  .panel-control input[type="range"] {{
    width: 100%;
  }}

  /* Inspector LC centrado abajo */
  .lc-info {{
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,.08);
    border-radius: 10px;
    padding: 6px 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,.15);
    font-family: 'PoppinsLocal','Poppins',sans-serif;
    font-size: 2.5vh;
    pointer-events: none;
    width: 100%;
    display: flex;
    justify-content: center;
    margin-bottom: 8px;
  }}
  .lc-info .sw {{
    display:inline-block; width:12px; height:12px;
    border:1px solid rgba(0,0,0,.25);
    border-radius: 3px; vertical-align: -2px; margin-right: 6px;
  }}
</style>

<div class="map-wrapper">
  <div id="map"></div>

  <!-- Controles de animación (debajo del mapa) -->
  <div class="controls">
    <button class="player-btn" id="prev-btn" type="button" title="Anterior">
      <img src="{icon_prev}" alt="Anterior" width="22" height="22">
    </button>

    <button class="player-btn" id="toggle-btn" type="button" title="Play/Pause">
      <img id="toggle-icon" src="{icon_play}" alt="Play" width="22" height="22">
    </button>

    <button class="player-btn" id="next-btn" type="button" title="Siguiente">
      <img src="{icon_next}" alt="Siguiente" width="22" height="22">
    </button>

    <input id="slider" type="range" min="0" max="{len(frames)-1}" step="1" value="{st.session_state.idx}" style="width:100%;">
    <span id="label"></span>
  </div>
</div>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// === Datos desde Python ===
const FRAMES = {json.dumps(frames, separators=(',',':'))};
const GLOBAL_BOUNDS = [[{S}, {W}], [{N}, {E}]];
const LC_IMG = {json.dumps(LC_img) if LC_img else 'null'};
const LC_BOUNDS = {json.dumps(LC_BOUNDS) if LC_BOUNDS else 'null'};
const LC_LEGEND = {json.dumps(LC_LEGEND, ensure_ascii=False)};
const LC_CODES_B64 = {json.dumps(LC_CODES_B64) if LC_CODES_B64 else 'null'};
const LC_GRID_W = {LC_WID if LC_img else 0};
const LC_GRID_H = {LC_H if LC_img else 0};

function bToLeaflet(b) {{ return [[b[1], b[0]], [b[3], b[2]]]; }}

let idx = {st.session_state.idx};
let playing = false;
let timer = null;

// Mapa
const map = L.map('map', {{ preferCanvas: true, zoomSnap: 1, zoomDelta: 1 }});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom: 19, crossOrigin: true
}}).addTo(map);
map.createPane('labels');
map.getPane('labels').style.zIndex = 650;
map.getPane('labels').style.pointerEvents='none';
L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom:19, pane:'labels', crossOrigin:true
}}).addTo(map);

map.createPane('lcPane');   map.getPane('lcPane').style.zIndex = 350;
map.createPane('lossPane'); map.getPane('lossPane').style.zIndex = 400;
map.removeControl(map.attributionControl);

// Overlays
let overlay = L.imageOverlay(FRAMES[idx].img, bToLeaflet(FRAMES[idx].bounds), {{
  opacity:1.0, interactive:false, crossOrigin:true, pane:'lossPane'
}}).addTo(map);
let rect = L.rectangle(bToLeaflet(FRAMES[idx].bounds), {{
  color:'#fff', weight:3, fill:false, pane:'lossPane'
}}).addTo(map);
map.fitBounds(GLOBAL_BOUNDS, {{ padding:[10,10] }});

let lcLayer = null;
if (LC_IMG && LC_BOUNDS) {{
  lcLayer = L.imageOverlay(LC_IMG, bToLeaflet(LC_BOUNDS), {{
    opacity:1.0, interactive:false, crossOrigin:true, pane:'lcPane'
  }});
}}

// ===== Leyenda LC =====
let lcLegendCtrl = null;
function createLcLegend() {{
  const ctrl = L.control({{ position: 'bottomright' }});
  ctrl.onAdd = function () {{
    const div = L.DomUtil.create('div', 'leaflet-control lc-legend');
    const rows = (Array.isArray(LC_LEGEND) ? LC_LEGEND : []).map(item =>
      `<div class="row"><span class="swatch" style="background:${{item.color}}"></span><span>${{item.code}} – ${{item.label}}</span></div>`
    ).join('');
    div.innerHTML = `<div class="ttl">Land cover</div>${{rows}}`;
    return div;
  }};
  return ctrl;
}}

// ===== Inspector LC centrado abajo =====
function decodeUint16FromBase64(b64) {{
  if (!b64) return null;
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Uint16Array(buf);
}}
const LC_CODES = decodeUint16FromBase64(LC_CODES_B64);
const LC_LOOKUP = (() => {{
  const m = {{}};
  (Array.isArray(LC_LEGEND) ? LC_LEGEND : []).forEach(it => {{ m[it.code] = {{label: it.label, color: it.color}}; }});
  return m;
}})();
function lcLatLngToRowCol(lat, lng) {{
  if (!LC_BOUNDS || !LC_GRID_W || !LC_GRID_H) return null;
  const W = LC_BOUNDS[0], S = LC_BOUNDS[1], E = LC_BOUNDS[2], N = LC_BOUNDS[3];
  if (lng < W || lng > E || lat < S || lat > N) return null;
  const col = Math.floor((lng - W) / (E - W) * LC_GRID_W);
  const row = Math.floor((N - lat) / (N - S) * LC_GRID_H);
  if (col < 0 || col >= LC_GRID_W || row < 0 || row >= LC_GRID_H) return null;
  return {{row, col}};
}}
const LcInfo = L.Control.extend({{
  options: {{ position: 'bottomleft' }},
  onAdd: function(map) {{
    const outer = L.DomUtil.create('div', 'lc-info-outer');
    const inner = L.DomUtil.create('div', 'lc-info', outer);
    inner.innerHTML = '<div class="lc-info-inner">Pasa el cursor sobre Land cover</div>';
    L.DomEvent.disableClickPropagation(inner);
    L.DomEvent.disableScrollPropagation(inner);
    this._inner = inner;
    return outer;
  }},
  getContainerEl: function() {{ return this._inner; }}
}});
const lcInfoCtrl = new LcInfo().addTo(map);
let _lastShown = '';
function updateLcInfo(latlng) {{
  if (!LC_CODES) {{
    lcInfoCtrl.getContainerEl().innerHTML = '<div class="lc-info-inner">Land cover no disponible</div>';
    return;
  }}
  if (!lcLayer || !map.hasLayer(lcLayer)) {{
    lcInfoCtrl.getContainerEl().innerHTML = '<div class="lc-info-inner">Land cover apagado</div>';
    return;
  }}
  const rc = lcLatLngToRowCol(latlng.lat, latlng.lng);
  if (!rc) {{
    lcInfoCtrl.getContainerEl().innerHTML = '<div class="lc-info-inner">Fuera del Land cover</div>';
    return;
  }}
  const idx1D = rc.row * LC_GRID_W + rc.col;
  const code = LC_CODES[idx1D];
  const meta = LC_LOOKUP[code];
  const html = meta
    ? `<div class="lc-info-inner"><span class="sw" style="background:${{meta.color}}"></span>${{code}} — ${{meta.label}}</div>`
    : `<div class="lc-info-inner">Código ${{code}}</div>`;
  if (html !== _lastShown) {{
    lcInfoCtrl.getContainerEl().innerHTML = html;
    _lastShown = html;
  }}
}}
map.on('mousemove', (e) => updateLcInfo(e.latlng));

// ===== Panel flotante dentro del mapa (Leaflet Control) =====
let lossOpacityValEl = null;
const PanelControl = L.Control.extend({{
  options: {{ position: 'topright' }},
  onAdd: function(map) {{
    const div = L.DomUtil.create('div', 'panel-control');
    div.innerHTML = `
      <div class="section">
        <h3>Capas</h3>
        <label class="control-row">
          <input id="lc-toggle" type="checkbox"> <span>Land cover</span>
        </label>
      </div>
      <div class="section">
        <h3>Opacidad pérdida de vegetación</h3>
        <input id="loss-opacity" type="range" min="0" max="1" step="0.05" value="1">
        <div><small id="loss-opacity-val">100%</small></div>
      </div>
    `;
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);

    const lcToggle = div.querySelector('#lc-toggle');
    const lossOpacityEl = div.querySelector('#loss-opacity');
    lossOpacityValEl = div.querySelector('#loss-opacity-val');

    if (lcToggle) {{
      lcToggle.addEventListener('change', () => {{
        if (!lcLayer) return;
        if (lcToggle.checked) {{
          lcLayer.addTo(map);
          if (!lcLegendCtrl && Array.isArray(LC_LEGEND) && LC_LEGEND.length > 0) {{
            lcLegendCtrl = createLcLegend();
            lcLegendCtrl.addTo(map);
          }}
        }} else {{
          map.removeLayer(lcLayer);
          if (lcLegendCtrl) {{
            map.removeControl(lcLegendCtrl);
            lcLegendCtrl = null;
          }}
        }}
      }});
    }}

    if (lossOpacityEl) {{
      const setLossOpacity = (v) => {{
        overlay.setOpacity(v);
        if (rect) rect.setStyle({{opacity: Math.max(0.3, v)}});
        if (lossOpacityValEl) lossOpacityValEl.textContent = Math.round(v*100) + '%';
      }};
      setLossOpacity(parseFloat(lossOpacityEl.value || '1'));
      lossOpacityEl.addEventListener('input', (e) => setLossOpacity(parseFloat(e.target.value)));
    }}

    return div;
  }}
}});
new PanelControl().addTo(map);

// ===== Animación (controles debajo del mapa) =====
const labelEl  = document.getElementById('label');
const sliderEl = document.getElementById('slider');
const prevBtn  = document.getElementById('prev-btn');
const nextBtn  = document.getElementById('next-btn');
const toggleBtn= document.getElementById('toggle-btn');
const toggleIcon= document.getElementById('toggle-icon');


function show(i) {{
  idx = ((i % FRAMES.length) + FRAMES.length) % FRAMES.length;
  overlay.setUrl(FRAMES[idx].img);
  const bnds = bToLeaflet(FRAMES[idx].bounds);
  overlay.setBounds(bnds);
  rect.setBounds(bnds);
  sliderEl.value = idx;
  labelEl.textContent = FRAMES[idx].label;
}}
show(idx);

function toggle() {{
  if (!playing) {{
    playing = true;
    toggleIcon.src = "{icon_pause}";
    timer = setInterval(() => show(idx + 1), {int(st.session_state.interval * 1000)});
  }} else {{
    playing = false;
    toggleIcon.src = "{icon_play}";
    clearInterval(timer);
  }}
}}
prevBtn.onclick   = (e) => {{ e.preventDefault(); show(idx - 1); }};
nextBtn.onclick   = (e) => {{ e.preventDefault(); show(idx + 1); }};
sliderEl.oninput  = (e) => {{ show(parseInt(e.target.value)); }};
toggleBtn.onclick = (e) => {{ e.preventDefault(); toggle(); }};  // ← ¡IMPRESCINDIBLE!

// ===== Redimensionado fiable del iframe + mapa (un solo bloque) =====
function totalOuterHeight(sel){{
  const el = document.querySelector(sel);
  if(!el) return 0;
  const cs = getComputedStyle(el);
  return el.offsetHeight + parseFloat(cs.marginTop||0) + parseFloat(cs.marginBottom||0);
}}
function setMapHeight(){{
  const headerH  = totalOuterHeight('.header-row');
  const controlsH= totalOuterHeight('.controls');
  const padding  = 24;
  const available = Math.max(380, window.innerHeight - (headerH + controlsH + padding));
  const mapDiv = document.getElementById('map');
  if (mapDiv) mapDiv.style.height = available + 'px';
  if (typeof map !== 'undefined' && map.invalidateSize) map.invalidateSize(true);
}}
function getDocHeight(){{
  const b = document.body, d = document.documentElement;
  return Math.max(
    b.scrollHeight, b.offsetHeight, b.getBoundingClientRect().height,
    d.clientHeight, d.scrollHeight, d.offsetHeight, d.getBoundingClientRect().height
  );
}}
function postResize(h){{
  if (window.parent && window.parent.Streamlit && window.parent.Streamlit.setFrameHeight) {{
    window.parent.Streamlit.setFrameHeight(h, {{ debounce: 0 }});
  }}
  if (window.parent && window.parent.postMessage) {{
    window.parent.postMessage({{ type: 'streamlit:setFrameHeight', height: h }}, '*');
  }}
}}
function resizeEverything(){{
  setMapHeight();
  requestAnimationFrame(()=>postResize(getDocHeight()));
}}

window.addEventListener('load',  resizeEverything);
window.addEventListener('resize', resizeEverything);
try {{ new ResizeObserver(()=>resizeEverything()).observe(document.body); }} catch(e) {{}}
setTimeout(resizeEverything, 100);
setTimeout(resizeEverything, 350);
setTimeout(resizeEverything, 900);
</script>
"""

components.html(html, height=600, scrolling=False)
