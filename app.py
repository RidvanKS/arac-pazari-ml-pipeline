"""
🚗 Araç Pazarı ML Pipeline — Streamlit Web Uygulaması
Üniversite Final Projesi
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval

# ════════════════════════════════════════════════════════════════
# SAYFA AYARLARI
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Araç Fırsat Analizi",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════ CSS ═══════════════════════════
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%); color: #f1f5f9; }
    h1, h2, h3, h4 { color: #f1f5f9 !important; }
    label, p, div { color: #e2e8f0; }

    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        padding: 2rem; border-radius: 20px; text-align: center;
        margin-bottom: 1.5rem; box-shadow: 0 10px 40px rgba(124,58,237,0.3);
    }
    .hero-banner h1 { color: white !important; margin: 0; font-size: 2.2rem; }
    .hero-banner p  { color: #e9d5ff; margin: 0.5rem 0 0; font-size: 1.05rem; }

    .result-card {
        background: #131826; border: 1px solid #2d3348; border-radius: 15px;
        padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .result-card h3 { margin-top: 0; }

    .badge {
        display: inline-block; padding: 0.7rem 1.4rem; border-radius: 12px;
        font-weight: 800; font-size: 1.2rem; margin: 0.3rem 0; letter-spacing: 0.5px;
    }
    .badge-altin   { background: #16a34a; color: white; }
    .badge-tuzak   { background: #f59e0b; color: white; }
    .badge-premium { background: #2563eb; color: white; }
    .badge-riskli  { background: #dc2626; color: white; }

    .info-box {
        background: #1e293b; border-left: 4px solid #3b82f6;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
        font-size: 0.92rem; color: #cbd5e1;
    }
    .price-big   { font-size: 2.3rem; font-weight: 800; color: #60a5fa; }
    .price-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

    .car-legend {
        display: flex; justify-content: center; gap: 1.2rem; flex-wrap: wrap;
        margin-top: 0.8rem; font-size: 0.85rem;
    }
    .legend-item { display:flex; align-items:center; gap:0.4rem; color:#cbd5e1; }
    .legend-dot  { width: 14px; height: 14px; border-radius: 50%; display:inline-block; }

    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
        color: #cbd5e1 !important; font-weight: 600;
    }
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white; border: none; border-radius: 12px;
        padding: 0.8rem 2rem; font-weight: 700; font-size: 1.05rem;
        width: 100%; box-shadow: 0 4px 15px rgba(59,130,246,0.4);
    }
    .stButton button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59,130,246,0.6); }

    .preset-btn button {
        background: #1e293b !important; border: 1px solid #475569 !important;
        font-size: 0.85rem !important; padding: 0.5rem !important;
    }

    div[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg,#3b82f6,#8b5cf6); }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# BUNDLE YÜKLEME (cache)
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bundle():
    bundle_path = Path("web_bundle/web_bundle.joblib")
    if not bundle_path.exists():
        st.error(f"❌ Bundle bulunamadı: {bundle_path}")
        st.stop()
    return joblib.load(bundle_path)


@st.cache_data
def load_df():
    return pd.read_parquet("data_exports/df_model_clean.parquet")


# ⭐⭐⭐ EKSİK OLAN BLOK — BURAYA EKLE ⭐⭐⭐
bundle = load_bundle()
df = load_df()

model1 = bundle["model1"]
model2 = bundle["model2"]
model3 = bundle["model3"]
model1_meta = bundle["model1_meta"]
model2_meta = bundle["model2_meta"]
model3_meta = bundle["model3_meta"]
encoders = bundle["encoders"]
hierarchy = bundle["hierarchy"]
ui_options = bundle["ui_options"]
thresholds = bundle["thresholds"]
firsat_styles = bundle["firsat_styles"]
segment_lookup = bundle["segment_lookup"]
global_stats = bundle["global_stats"]
damage_map = bundle["damage_map"]
extra_meta = bundle.get("extra_meta", {})


# ─── SHAP explainer runtime'da yeniden oluştur ───
@st.cache_resource
def build_explainers():
    expl = {}
    try:
        expl["model2"] = shap.TreeExplainer(model2)
    except Exception as e:
        st.warning(f"Model 2 explainer kurulamadı: {e}")
    return expl

explainers = build_explainers()
# ⭐⭐⭐ EKSİK OLAN BLOK BİTTİ ⭐⭐⭐

# ════════════════════════════════════════════════════════════════
# PIPELINE FONKSİYONLARI (notebook ile birebir aynı)
# ════════════════════════════════════════════════════════════════
def get_feature_list(meta, model=None):
    for k in ["model1_features", "model2_features", "model3_features", "features"]:
        if k in meta and isinstance(meta[k], list) and len(meta[k]) > 0:
            return meta[k]
    if model is not None:
        for attr in ["feature_names_in_", "feature_name_"]:
            if hasattr(model, attr):
                return list(getattr(model, attr))
    raise KeyError(f"Feature listesi yok: {list(meta.keys())}")


def build_feature_row(user_input, encoders, df_ref, damage_map):
    yil_simdi = datetime.now().year
    row = {}

    row["yil"]              = user_input["yil"]
    row["kilometre"]        = user_input["kilometre"]
    row["motor_hacmi_num"]  = user_input.get("motor_hacmi_num", 1600)
    row["motor_gucu_num"]   = user_input.get("motor_gucu_num", 120)
    row["arac_yasi"]        = max(0, yil_simdi - user_input["yil"])
    row["tramer_tutari"]    = user_input.get("tramer_tutari", 0)

    encode_pairs = [
        ("marka","marka_enc"),("seri","seri_enc"),("model","model_enc"),
        ("vites_tipi","vites_tipi_enc"),("yakit_tipi","yakit_tipi_enc"),
        ("kasa_tipi","kasa_tipi_enc"),("renk","renk_enc"),
        ("cekis","cekis_enc"),("kimden","kimden_enc"),("il","il_enc"),
    ]
    for orig, enc in encode_pairs:
        val = user_input.get(orig)
        m = encoders.get(f"{orig}_to_enc", {})
        fallback = int(np.median(list(m.values()))) if m else 0
        row[enc] = m.get(val, fallback)

    parca_listesi = ["kaput","tavan","on_tampon","arka_tampon",
                     "sol_on_camurluk","sag_on_camurluk","sol_on_kapi","sag_on_kapi",
                     "sol_arka_kapi","sag_arka_kapi","sol_arka_camurluk","sag_arka_camurluk",
                     "bagaj_kapagi"]
    parca_durumlari = user_input.get("parca_durumlari", {})
    sayim = {"orijinal":0,"boyali":0,"degismis":0,"lokal_boyali":0,"belirtilmemis":0}
    parca_risk_listesi = []

    for parca in parca_listesi:
        durum = parca_durumlari.get(parca, "belirtilmemis")
        skor  = damage_map.get(durum, 0)
        row[f"{parca}_risk"] = skor
        parca_risk_listesi.append(skor)
        if durum in sayim:
            sayim[durum] += 1

    row["orijinal_sayi"]      = sayim["orijinal"]
    row["boyali_sayi"]        = sayim["boyali"]
    row["degismis_sayi"]      = sayim["degismis"]
    row["lokal_boyali_sayi"]  = sayim["lokal_boyali"]
    row["belirtilmemis_sayi"] = sayim["belirtilmemis"]
    row["parca_risk_toplam"]   = sum(parca_risk_listesi)
    row["parca_risk_ortalama"] = float(np.mean(parca_risk_listesi))
    row["parca_risk_max"]      = max(parca_risk_listesi)
    row["degismis_parca_var"]  = int(sayim["degismis"] > 0)
    row["hasar_skoru"]         = sum(parca_risk_listesi)

    return pd.DataFrame([row])


def add_market_and_interaction_features(row_df, user_input, tahmini_fiyat,
                                         encoders, segment_lookup, global_stats):
    r = row_df.iloc[0].to_dict()
    liste_raw = user_input.get("liste_fiyati")
    liste = liste_raw if liste_raw is not None else tahmini_fiyat

    r["liste_fiyati"]          = liste
    r["tahmini_piyasa_fiyati"] = tahmini_fiyat
    r["fiyat_fark_pct"]        = float(np.clip((liste-tahmini_fiyat)/tahmini_fiyat*100, -100, 200))
    r["fiyat_orani"]           = float(np.clip(liste/max(tahmini_fiyat,1), 0.3, 3.0))
    r["mutlak_fiyat_farki"]    = abs(liste - tahmini_fiyat)
    r["log_fiyat_orani"]       = float(np.log1p(r["fiyat_orani"]))

    seg_key = (user_input["marka"], user_input["seri"])
    seg = segment_lookup.get(seg_key, global_stats)
    seg_mean = seg["mean_price"]
    seg_std  = seg["std_price"] if seg["std_price"] > 0 else 0
    seg_median_price = seg.get("median_price", seg_mean)

    if seg_std > 0:
        r["seg_fiyat_zscore"] = float(np.clip((liste-seg_mean)/seg_std, -4, 4))
    else:
        r["seg_fiyat_zscore"] = 0.0

    r["seg_fiyat_oran"] = float(np.clip(liste/max(seg_median_price,1), 0.3, 3.0))
    r["seg_km_oran"]    = float(np.clip(r["kilometre"]/max(seg["mean_km"],1), 0, 5))
    r["seg_yas_fark"]    = r["arac_yasi"] - seg["mean_yas"]
    r["seg_rekabet_log"] = float(np.log1p(seg["count"]))
    r["fiyat_avantaj_skoru"] = (1-r["seg_fiyat_oran"])*0.6 + (1-min(r["seg_km_oran"],2)/2)*0.4

    il_count    = encoders.get("il_pazar_buyuklugu", {}).get(user_input.get("il"), 0)
    marka_count = encoders.get("marka_populerlik", {}).get(user_input["marka"], 0)
    r["il_pazar_buyuklugu"] = float(np.log1p(il_count))
    r["marka_populerlik"]   = float(np.log1p(marka_count))

    r["fiyat_km_orani"]   = float(np.log1p(liste/max(r["kilometre"],1)))
    r["fiyat_guc_orani"]  = float(np.log1p(liste/max(r["motor_gucu_num"],1)))
    r["yas_km_etkilesim"] = r["arac_yasi"] * float(np.log1p(r["kilometre"]))
    r["hasar_yogunlugu"]  = float(np.log1p(r.get("tramer_tutari",0)))*0.5 + r.get("parca_risk_toplam",0)*0.5
    r["kimden_fiyat_etkilesim"] = r.get("kimden_enc",0) * r["fiyat_fark_pct"]
    yillik_km = r["kilometre"]/max(r["arac_yasi"],1)
    r["yillik_km_log"] = float(np.log1p(yillik_km))

    return pd.DataFrame([r])


def run_pipeline(user_input):
    base_row = build_feature_row(user_input, encoders, df, damage_map)
    m1_features = get_feature_list(model1_meta, model1)
    X1 = base_row.reindex(columns=m1_features, fill_value=0)
    tahmini_fiyat = float(model1.predict(X1)[0])

    liste_raw = user_input.get("liste_fiyati")
    liste = liste_raw if liste_raw is not None else tahmini_fiyat
    fark_pct = ((liste-tahmini_fiyat)/tahmini_fiyat)*100

    enriched = add_market_and_interaction_features(
        base_row, user_input, tahmini_fiyat, encoders, segment_lookup, global_stats)

    m2_features = get_feature_list(model2_meta, model2)
    X2 = enriched.reindex(columns=m2_features, fill_value=0)
    m2_proba = model2.predict_proba(X2)[0]
    m2_pred  = int(np.argmax(m2_proba))
    speed_labels = model2_meta.get("speed_labels", ["Hizli","Yavas"])

    enriched["tahmin_hizli_olasiligi"] = float(m2_proba[0])
    enriched["tahmin_yavas_olasiligi"] = float(m2_proba[1])
    m3_features = get_feature_list(model3_meta, model3)
    X3 = enriched.reindex(columns=m3_features, fill_value=0)
    m3_proba = model3.predict_proba(X3)[0]
    m3_pred_idx = int(np.argmax(m3_proba))
    m3_classes_raw = list(model3.classes_)

    label_map = model3_meta.get("firsat_class_map", {})
    inverse_map = {v:k for k,v in label_map.items()}
    label_list = model3_meta.get("firsat_labels", [])

    def resolve(c):
        if c in inverse_map: return inverse_map[c]
        try:
            idx = int(c)
            if 0 <= idx < len(label_list): return label_list[idx]
        except: pass
        return str(c)

    m3_classes = [resolve(c) for c in m3_classes_raw]

    return {
        "model1": {
            "tahmini_piyasa_fiyati": tahmini_fiyat,
            "liste_fiyati": liste_raw,
            "fiyat_fark_pct": fark_pct,
        },
        "model2": {
            "tahmin": speed_labels[m2_pred],
            "olasilik_hizli": float(m2_proba[0]),
            "olasilik_yavas": float(m2_proba[1]),
        },
        "model3": {
            "firsat_kategorisi": m3_classes[m3_pred_idx],
            "olasiliklar": {m3_classes[i]: float(m3_proba[i]) for i in range(len(m3_classes))},
        },
        "_X2": X2,
        "_enriched": enriched,
    }


# ════════════════════════════════════════════════════════════════
# ARABA SVG (DİNAMİK RENKLİ)
# ════════════════════════════════════════════════════════════════
DURUM_RENK = {
    "orijinal":     "#7ec97e",
    "lokal_boyali": "url(#Gradient_local)",
    "boyali":       "#fdb724",
    "degismis":     "#dc2626",
    "belirtilmemis":"#6b7280",
}

PARCA_LABELS = {
    "kaput":            "🚪 Motor Kaputu",
    "tavan":            "🏠 Tavan",
    "on_tampon":        "⬆️ Ön Tampon",
    "arka_tampon":      "⬇️ Arka Tampon",
    "sol_on_camurluk":  "↖️ Sol Ön Çamurluk",
    "sag_on_camurluk":  "↗️ Sağ Ön Çamurluk",
    "sol_on_kapi":      "🚪 Sol Ön Kapı",
    "sag_on_kapi":      "🚪 Sağ Ön Kapı",
    "sol_arka_kapi":    "🚪 Sol Arka Kapı",
    "sag_arka_kapi":    "🚪 Sağ Arka Kapı",
    "sol_arka_camurluk":"↙️ Sol Arka Çamurluk",
    "sag_arka_camurluk":"↘️ Sağ Arka Çamurluk",
    "bagaj_kapagi":     "📦 Bagaj Kapağı",
}

DURUM_LABELS = {
    "orijinal":     "🟢 Orijinal (sağlam)",
    "lokal_boyali": "🟡 Lokal Boyalı (küçük rötuş)",
    "boyali":       "🟠 Boyalı (komple boya)",
    "degismis":     "🔴 Değişmiş (parça yenilenmiş)",
    "belirtilmemis":"⚪ Belirtilmemiş",
}


def render_car_svg(parca_durumlari):
    """Tıklanabilir SVG — JavaScript ile state yönetimi."""
    import json
    durumlar_json = json.dumps(parca_durumlari)
    # Her state değişikliğinde version değişir → localStorage cache invalidate
    version = str(hash(durumlar_json))   # ← YENİ SATIR

    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .car-container {
            background: #0f172a;
            padding: 1rem;
            border-radius: 15px;
            border: 1px solid #2d3348;
        }
        .hint {
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        svg path[data-parca] {
            cursor: pointer;
            transition: all 0.2s ease;
        }
        svg path[data-parca]:hover {
            opacity: 0.7;
            stroke: #60a5fa !important;
            stroke-width: 2.5 !important;
        }
        .car-legend {
            display: flex;
            justify-content: center;
            gap: 1.2rem;
            flex-wrap: wrap;
            margin-top: 0.8rem;
            font-size: 0.85rem;
        }
        .legend-item { display: flex; align-items: center; gap: 0.4rem; color: #cbd5e1; }
        .legend-dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
        .toast {
            position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
            background: #1e293b; color: #f1f5f9; padding: 0.5rem 1rem;
            border-radius: 8px; border: 1px solid #3b82f6;
            font-size: 0.85rem; opacity: 0; transition: opacity 0.3s;
            pointer-events: none; z-index: 999;
        }
        .toast.show { opacity: 1; }
    </style>
    </head>
    <body>
    <div class="car-container">
      <div class="hint">👆 <b>Parçalara tıkla</b> — durum sırayla değişir: 🟢→🟡→🟠→🔴</div>
      <div id="toast" class="toast"></div>
      <svg id="carSvg" version="1.1" style="height:380px;width:100%;display:block;margin:auto;" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 430">
        <defs>
          <linearGradient id="Gradient_local" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#fff"/>
            <stop offset="10%" style="stop-color:#ffdb4d"/>
            <stop offset="20%" style="stop-color:#fff"/>
            <stop offset="30%" style="stop-color:#ffdb4d"/>
            <stop offset="40%" style="stop-color:#fff"/>
            <stop offset="50%" style="stop-color:#ffdb4d"/>
            <stop offset="60%" style="stop-color:#fff"/>
            <stop offset="70%" style="stop-color:#ffdb4d"/>
            <stop offset="80%" style="stop-color:#fff"/>
            <stop offset="90%" style="stop-color:#ffdb4d"/>
            <stop offset="100%" style="stop-color:#fff"/>
          </linearGradient>
        </defs>
        <g fill="none" fill-rule="evenodd">
          <g transform="translate(156.5 219.5) rotate(-90) translate(-188.5 -144.5)">
            <path data-parca="sol_on_kapi" d="m106.51 55.944c-0.52167 0.93363-0.66394 2.0147-0.33197 2.9974 0.85364 2.506 2.5609 4.5207 4.7899 5.7492l4.6476 2.506c9.5798 5.2087 20.345 7.9113 31.158 7.9113h13.421l3.13-17.248c1.3279-7.2233 1.9918-14.643 1.9918-21.965v-25.847c-2.4187 0-8.0622-0.049138-13.706-0.049138-4.3156 0-7.7776 0-10.196 0.049138-1.8496 0-3.794 1.081-5.3115 2.9483-1.1856 1.425-2.2764 2.9483-3.2723 4.5207-1.0433 1.6707-1.6599 2.9483-2.229 4.1767-0.71137 1.4741-1.3753 2.8992-2.6084 4.619-1.2805 1.769-2.798 3.4397-4.5053 4.9138-3.4146 2.9483-6.3075 6.388-8.5838 10.319l-0.047424 0.049138v0.049138l-8.3467 14.299zm5.027-0.88449c2.0393-1.769 4.6476-2.7517 7.3508-2.7517h40.548c0.80622 0 1.4227 0.73707 1.3279 1.5724l-2.4187 16.412c-0.23712 1.5724-1.5176 2.7026-3.0352 2.6535l-10.149-0.19655c-9.4849-0.19655-18.733-2.85-26.937-7.7638l-3.13-1.8673c-1.8021-1.081-3.2723-2.6535-4.2208-4.5699-0.61652-1.1793-0.33197-2.6535 0.66394-3.4888z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sol_arka_kapi" d="m166.04 58.376l-3.0398 16.732 2.9448-0.14764c18.476-0.98425 36.62-6.7421 52.483-16.634 3.4197-2.1653 5.9845-5.561 7.0769-9.5472 3.2297-11.467 3.9897-23.72 2.1848-35.531l-0.28498-1.7224c-0.14249-0.88582-0.85493-1.5256-1.7574-1.5256h-57.613v25.886c0 7.5295-0.66495 15.108-1.9948 22.49zm-0.28498 12.352l3.8472-15.994c0.42747-1.7224 1.8524-2.9527 3.5622-3.0512l34.055-1.9193v-0.049212c0-2.559 1.9948-4.6752 4.5121-4.6752h7.3144c0.61745 0 1.1874 0.14764 1.7574 0.34449z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="on_tampon" d="m340 201.35c3.7652 1.6125 7.7313 2.4968 11.798 2.4968h15.864c3.2632 0 5.9742-2.8609 6.1248-6.5021 0.70284-16.958 1.1547-34.643 1.1547-53.005v-0.41614c0-18.518-0.40162-36.36-1.1547-53.422-0.15061-3.6412-2.8616-6.5021-6.1248-6.5021h-15.864c-4.0664 0-8.0325 0.83227-11.798 2.4968v114.85z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sol_on_camurluk" d="m59.277 54.594s6.1176 1.9503 15.529 0.97517l5.3646 0.29255s9.2234 7.5088 12.047 7.2163c2.8235-0.34131 7.7175-8.4352 7.7175-8.4352l10.635-19.503c-0.14118 0.24379-15.2 3.4131-22.164-1.414-6.0234-4.1932-10.682-10.824-11.482-17.846l-0.79999-4.8759s-9.3175-0.39007-12.329 6.5824c-3.0117 6.9725-6.5411 9.4592-6.5411 9.4592s-0.94116 10.629 0.79999 13.262c1.6941 2.5842 1.2235 14.286 1.2235 14.286z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sag_on_camurluk" transform="translate(267.88 27.237) scale(-1) rotate(180) translate(-267.88 -27.237)" d="m234.26 49.983l53.188-9.0296s15.014-4.4657 16.577-8.6861c1.563-4.2204 2.3681-7.0176 1.563-10.109-0.80516-3.0917-2.8418-10.502-2.8418-10.502s3.3154-6.1833-0.61572-6.1833c-3.9311 0-15.958-0.98148-15.958-0.98148s2.3211 32.474-25.531 32.907c-25.568 0.39668-24.904-28.637-24.904-28.637h-5.8815s5.7309 23.212 0 41.222h4.4042z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sag_arka_camurluk" transform="translate(267.88 261.26) scale(-1 1) rotate(180) translate(-267.88 -261.26)" d="m234.26 284.01l53.188-9.0296s15.014-4.4657 16.577-8.6861c1.563-4.2204 2.3681-7.0176 1.563-10.109-0.80516-3.0917-2.8418-10.502-2.8418-10.502s3.3154-6.1833-0.61572-6.1833c-3.9311 0-15.958-0.98148-15.958-0.98148s2.3211 32.474-25.531 32.907c-25.568 0.39668-24.904-28.637-24.904-28.637h-5.8815s5.7309 23.212 0 41.222h4.4042z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sag_on_kapi" d="m114.9 247.46l0.047425 0.098276c2.2764 3.9311 5.1693 7.4199 8.5838 10.319 1.7547 1.4741 3.2723 3.1448 4.5053 4.9138 1.233 1.7198 1.897 3.1448 2.6084 4.619 0.61652 1.2285 1.233 2.5552 2.229 4.1767 0.99592 1.5724 2.0867 3.0957 3.2723 4.5207 1.5176 1.8181 3.462 2.8992 5.3115 2.9483 2.4661 0.049138 5.8806 0.049138 10.196 0.049138 5.5961 0 11.287-0.049138 13.706-0.049138v-25.847c0-7.3707-0.66394-14.741-1.9918-21.965l-3.13-17.248h-13.469c-10.813 0-21.578 2.7517-31.158 7.9113l-4.6476 2.506c-2.2764 1.2285-3.9362 3.2431-4.7899 5.7492-0.33197 0.98276-0.1897 2.0638 0.33197 2.9974l8.3941 14.299z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sag_arka_kapi" d="m169.03 253.22v25.886h57.66c0.85493 0 1.6149-0.63976 1.7574-1.5256l0.28498-1.7224c1.8049-11.811 1.0449-24.065-2.1848-35.531-1.1399-3.9862-3.6572-7.3819-7.0769-9.5472-15.911-9.9409-34.055-15.65-52.531-16.634l-2.9448-0.14764 3.0398 16.732c1.3299 7.3819 1.9948 14.961 1.9948 22.49z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="sol_arka_camurluk" d="m58.43 234.06s6.5264-2.0807 16.567-1.0403l5.7232-0.3121s9.8398-8.0106 12.852-7.6985c3.0122 0.36412 8.2333 8.999 8.2333 8.999l11.346 20.807c-0.15061-0.26008-16.216-3.6412-23.646 1.5085-6.426 4.4735-11.396 11.548-12.25 19.038l-0.85345 5.2017s-9.9402 0.41614-13.153-7.0223c-3.213-7.4385-6.9782-10.091-6.9782-10.091s-1.0041-11.34 0.85345-14.149c1.8073-2.7569 1.3053-15.241 1.3053-15.241z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="kaput" d="m230 100s14.961 40.833 0 87.129h53.968s20.633-8.1667 18.876-43.07c-1.7571-34.904-18.876-44.059-18.876-44.059h-53.968z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="bagaj_kapagi" d="m95.64 100.03h-23.897s-10.743-1.3004-10.743 13.004v65.594s1.7069 8.7909 8.4843 8.7909h26.156s-8.5345-37.712 0-87.389z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="tavan" d="m126.16 111s-10.794 28.349-1.1547 64.501h63.658s8.7855-32.771 0-64.501h-62.503z" stroke="#374151" stroke-width="0.5"/>
            <path data-parca="arka_tampon" d="m36.941 86.497c-3.7652-1.6125-7.7313-2.4968-11.798-2.4968h-15.864c-3.2632 0-5.9742 2.8609-6.1248 6.5021-0.70284 16.958-1.1547 34.643-1.1547 53.005v0.41614c0 18.518 0.40162 36.36 1.1547 53.422 0.15061 3.6412 2.8616 6.5021 6.1248 6.5021h15.864c4.0664 0 8.0325-0.83228 11.798-2.4968v-114.85z" stroke="#374151" stroke-width="0.5"/>
          </g>
        </g>
      </svg>
      <div class="car-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#7ec97e"></span> Orijinal</div>
        <div class="legend-item"><span class="legend-dot" style="background:#fdb724"></span> Boyalı</div>
        <div class="legend-item"><span class="legend-dot" style="background:#dc2626"></span> Değişmiş</div>
        <div class="legend-item"><span class="legend-dot" style="background:linear-gradient(90deg,#fff 50%,#ffdb4d 50%)"></span> Lokal Boyalı</div>
        <div class="legend-item"><span class="legend-dot" style="background:#6b7280"></span> Belirtilmemiş</div>
      </div>
    </div>

    <script>
      // Başlangıç durumu (Python'dan gelen)
      const COLORS = {
        "orijinal":     "#7ec97e",
        "lokal_boyali": "url(#Gradient_local)",
        "boyali":       "#fdb724",
        "degismis":     "#dc2626",
        "belirtilmemis":"#6b7280"
      };
      const LABELS = {
        "orijinal":     "🟢 Orijinal",
        "lokal_boyali": "🟡 Lokal Boyalı",
        "boyali":       "🟠 Boyalı",
        "degismis":     "🔴 Değişmiş",
        "belirtilmemis":"⚪ Belirtilmemiş"
      };
      const PARCA_NAMES = {
        "kaput":"Motor Kaputu","tavan":"Tavan","on_tampon":"Ön Tampon","arka_tampon":"Arka Tampon",
        "sol_on_camurluk":"Sol Ön Çamurluk","sag_on_camurluk":"Sağ Ön Çamurluk",
        "sol_on_kapi":"Sol Ön Kapı","sag_on_kapi":"Sağ Ön Kapı",
        "sol_arka_kapi":"Sol Arka Kapı","sag_arka_kapi":"Sağ Arka Kapı",
        "sol_arka_camurluk":"Sol Arka Çamurluk","sag_arka_camurluk":"Sağ Arka Çamurluk",
        "bagaj_kapagi":"Bagaj Kapağı"
      };
      const CYCLE = ["orijinal", "lokal_boyali", "boyali", "degismis"];

      // localStorage'dan oku, yoksa Python'dan geleni kullan
     let durumlar = __DURUMLAR__;
const PYTHON_VERSION = "__VERSION__";

try {
    const savedVersion = localStorage.getItem("parca_durumlari_version");
    const saved = localStorage.getItem("parca_durumlari");
    
    // Sadece version uyuşuyorsa localStorage'ı kullan
    if (saved && savedVersion === PYTHON_VERSION) {
        durumlar = JSON.parse(saved);
    } else {
        // Version uyuşmuyor → Python'dan geleni kabul et, localStorage'ı güncelle
        localStorage.setItem("parca_durumlari", JSON.stringify(durumlar));
        localStorage.setItem("parca_durumlari_version", PYTHON_VERSION);
    }
} catch(e) {}
      // Renkleri uygula
      function applyColors() {
        document.querySelectorAll("path[data-parca]").forEach(path => {
          const parca = path.getAttribute("data-parca");
          const durum = durumlar[parca] || "belirtilmemis";
          path.setAttribute("fill", COLORS[durum]);
        });
      }

      // Toast göster
      function showToast(msg) {
        const toast = document.getElementById("toast");
        toast.textContent = msg;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 1500);
      }

      // Tıklama
      document.querySelectorAll("path[data-parca]").forEach(path => {
        path.addEventListener("click", function() {
          const parca = this.getAttribute("data-parca");
          const current = durumlar[parca] || "orijinal";
          let idx = CYCLE.indexOf(current);
          if (idx === -1) idx = -1;
          const next = CYCLE[(idx + 1) % CYCLE.length];
          durumlar[parca] = next;

          // localStorage'a kaydet
         // localStorage'a kaydet
            try { 
                localStorage.setItem("parca_durumlari", JSON.stringify(durumlar));
                localStorage.setItem("parca_durumlari_version", PYTHON_VERSION);  // ← YENİ
            } catch(e) {}
          // Renk değiştir
          this.setAttribute("fill", COLORS[next]);

          // Toast
          showToast(`${PARCA_NAMES[parca]}: ${LABELS[next]}`);

          // Streamlit'e bildir (hidden input + change event)
          const hiddenInput = window.parent.document.querySelector('input[aria-label="parca_state_sync"]');
          if (hiddenInput) {
            const setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, "value").set;
            setter.call(hiddenInput, JSON.stringify(durumlar));
            hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
          }
        });
      });

      // İlk render
      applyColors();
    </script>
    </body>
    </html>
    """
    html = html.replace("__DURUMLAR__", durumlar_json)
    return html

# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
if "parca_durumlari" not in st.session_state:
    st.session_state.parca_durumlari = {p: "orijinal" for p in PARCA_LABELS.keys()}

if "result" not in st.session_state:
    st.session_state.result = None
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

if "js_sync_counter" not in st.session_state:
    st.session_state.js_sync_counter = 0
    # ════════════════════════════════════════════════════════════════
# TIKLAMA OLAYINI YAKALA (URL query param)
# ════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🚗 Araç Pazarı Akıllı Fırsat Analizi</h1>
    <p>Yapay zekâ ile aracın gerçek piyasa değerini öğren — fırsat mı tuzak mı?</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DEMO PRESET BUTONLARI
# ════════════════════════════════════════════════════════════════
st.markdown("### 🎯 Hızlı Demo")
demo_cols = st.columns(4)

def apply_preset(preset):
    for k, v in preset.items():
        st.session_state[k] = v

DEMOLAR = [
    {"label":"🟢 Altın Fırsat", "marka":"Volkswagen", "yil":2021, "km":40000, "liste":1300000, "hasar":{}},
    {"label":"💎 Premium",      "marka":"Toyota",     "yil":2023, "km":20000, "liste":1700000, "hasar":{}},
    {"label":"⚠️ Tuzak",        "marka":"Fiat",       "yil":2017, "km":180000,"liste":800000,
     "hasar":{"kaput":"boyali","on_tampon":"degismis","sol_on_kapi":"degismis"}},
    {"label":"🔴 Riskli",       "marka":"Renault",    "yil":2015, "km":220000,"liste":700000,
     "hasar":{"kaput":"degismis","tavan":"boyali","arka_tampon":"boyali","bagaj_kapagi":"degismis"}},
]

for i, demo in enumerate(DEMOLAR):
    with demo_cols[i]:
        with st.container():
            st.markdown('<div class="preset-btn">', unsafe_allow_html=True)
            if st.button(demo["label"], key=f"demo_{i}", use_container_width=True):
                if demo["marka"] in hierarchy:
                    seri = list(hierarchy[demo["marka"]].keys())[0]
                    mdl = hierarchy[demo["marka"]][seri][0]
                    st.session_state.demo_marka = demo["marka"]
                    st.session_state.demo_seri = seri
                    st.session_state.demo_model = mdl
                    st.session_state.demo_yil = demo["yil"]
                    st.session_state.demo_km = demo["km"]
                    st.session_state.demo_liste = demo["liste"]
                    new_hasar = {p:"orijinal" for p in PARCA_LABELS.keys()}
                    new_hasar.update(demo["hasar"])
                    st.session_state.parca_durumlari = new_hasar
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════
# ANA İÇERİK — 2 KOLON
# ════════════════════════════════════════════════════════════════
col_form, col_result = st.columns([1, 1.2], gap="large")


# ─────── SOL: FORM ───────
with col_form:
    st.markdown("## 📋 Araç Bilgileri")
    st.markdown('<div class="info-box">💡 Aracınızın bilgilerini girin, yapay zekâ piyasa fiyatını ve fırsat durumunu söylesin.</div>', unsafe_allow_html=True)

    # === Marka / Seri / Model (hiyerarşik) ===
    markalar = sorted(hierarchy.keys())
    default_marka = st.session_state.get("demo_marka", markalar[0])
    if default_marka not in markalar:
        default_marka = markalar[0]

    marka = st.selectbox("🏭 Marka", markalar, index=markalar.index(default_marka), key="sel_marka")

    seriler = sorted(hierarchy[marka].keys())
    default_seri = st.session_state.get("demo_seri", seriler[0])
    if default_seri not in seriler:
        default_seri = seriler[0]
    seri = st.selectbox("🚙 Seri", seriler, index=seriler.index(default_seri), key="sel_seri")

    modeller = sorted(hierarchy[marka][seri])
    default_model = st.session_state.get("demo_model", modeller[0])
    if default_model not in modeller:
        default_model = modeller[0]
    model_adi = st.selectbox("🔧 Model", modeller, index=modeller.index(default_model), key="sel_model")

    # === Temel bilgiler ===
    c1, c2 = st.columns(2)
    with c1:
        yil = st.number_input("📅 Model Yılı", min_value=1990, max_value=2025,
                              value=st.session_state.get("demo_yil", 2018), step=1)
    with c2:
        km = st.number_input("🛣️ Kilometre", min_value=0, max_value=1_000_000,
                             value=st.session_state.get("demo_km", 100000), step=1000)

    c3, c4 = st.columns(2)
    with c3:
        yakit = st.selectbox("⛽ Yakıt", ui_options.get("yakit_tipleri", ["Benzin"]))
    with c4:
        vites = st.selectbox("⚙️ Vites", ui_options.get("vites_tipleri", ["Otomatik"]))

    c5, c6 = st.columns(2)
    with c5:
        kasa = st.selectbox("🚘 Kasa Tipi", ui_options.get("kasa_tipleri", ["Sedan"]))
    with c6:
        renk = st.selectbox("🎨 Renk", ui_options.get("renkler", ["Beyaz"]))

    c7, c8 = st.columns(2)
    with c7:
        il = st.selectbox("📍 Şehir", ui_options.get("iller", ["İstanbul"]))
    with c8:
        kimden = st.selectbox("👤 Kimden", ui_options.get("kimden_tipleri", ["Sahibinden"]))

    cekis = st.selectbox("🔩 Çekiş", ui_options.get("cekis_tipleri", ["Önden Çekiş"]))

    c9, c10 = st.columns(2)
    with c9:
        motor_hacmi = st.number_input("🔧 Motor Hacmi (cc)", 600, 6000,
                                       value=1600, step=100)
    with c10:
        motor_gucu = st.number_input("⚡ Motor Gücü (HP)", 50, 800, value=120, step=5)

    tramer = st.number_input("💸 Tramer Tutarı (TL) — yoksa 0", 0, 1_000_000, value=0, step=1000,
                             help="Aracın trafik kazası geçmişindeki toplam hasar bedeli")

    st.markdown("### 💰 Liste Fiyatı (Opsiyonel)")
    st.markdown('<div class="info-box">📌 <b>Boş bırakırsan</b>: Piyasa değerini tahmin ederiz.<br>📌 <b>Doldurulursa</b>: O fiyatla satılırsa fırsat mı, tuzak mı analiz ederiz.</div>', unsafe_allow_html=True)

    use_liste = st.checkbox("Liste fiyatı girmek istiyorum", value=True)
    if use_liste:
        liste_fiyati = st.number_input("📋 Liste Fiyatı (TL)", 50_000, 10_000_000,
                                        value=st.session_state.get("demo_liste", 700000),
                                        step=10000)
    else:
        liste_fiyati = None


# ─────── SAĞ: ARABA + HASAR ───────
with col_result:
    st.markdown("## 🚗 Araç Hasar Durumu")
    st.markdown('<div class="info-box">🎨 Her parça için durumu seç. Araç görseli renklerle güncellenir.<br>🟢 Orijinal | 🟡 Lokal boya | 🟠 Boyalı | 🔴 Değişmiş</div>', unsafe_allow_html=True)

    # SVG göster
    # SVG göster
    components.html(render_car_svg(st.session_state.parca_durumlari), height=520, scrolling=False)

    # Durumları senkronize et (JS'den Python'a)
        # JS → Python state senkronizasyonu (her rerun'da fresh)
    if "js_sync_counter" not in st.session_state:
        st.session_state.js_sync_counter = 0

    sync_value = streamlit_js_eval(
        js_expressions="localStorage.getItem('parca_durumlari') || ''",
        key=f"parca_sync_{st.session_state.js_sync_counter}"
    )
    if sync_value:
        try:
            import json
            new_durumlar = json.loads(sync_value)
            if isinstance(new_durumlar, dict) and new_durumlar != st.session_state.parca_durumlari:
                st.session_state.parca_durumlari = new_durumlar
        except:
            pass

    # Hızlı sıfırlama
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("🔄 Hepsini Orijinal Yap", use_container_width=True):
            st.session_state.parca_durumlari = {p:"orijinal" for p in PARCA_LABELS.keys()}
            # localStorage'ı KOMPLE temizle (version dahil)
            streamlit_js_eval(
                js_expressions="localStorage.removeItem('parca_durumlari'); localStorage.removeItem('parca_durumlari_version'); 'cleared'",
                key=f"clear_orijinal_{st.session_state.get('reset_counter', 0)}"
            )
            st.session_state.reset_counter = st.session_state.get("reset_counter", 0) + 1
            st.session_state.js_sync_counter += 1   # Sync'i de invalidate et
            st.rerun()
    with cc2:
        if st.button("🗑️ Hepsini Sıfırla", use_container_width=True):
            st.session_state.parca_durumlari = {p:"belirtilmemis" for p in PARCA_LABELS.keys()}
            streamlit_js_eval(
                js_expressions="localStorage.removeItem('parca_durumlari'); localStorage.removeItem('parca_durumlari_version'); 'cleared'",
                key=f"clear_sifir_{st.session_state.get('reset_counter', 0)}"
            )
            st.session_state.reset_counter = st.session_state.get("reset_counter", 0) + 1
            st.session_state.js_sync_counter += 1
            st.rerun()

      

# ════════════════════════════════════════════════════════════════
# ANALİZ BUTONU
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# ANALİZ BUTONU (2 aşamalı: önce sync, sonra pipeline)
# ════════════════════════════════════════════════════════════════
st.markdown("---")
analiz_btn = st.button("🔮 ARACI ANALİZ ET", use_container_width=True)

# 1. AŞAMA: Buton tıklandı → senkronize et + pending işaretle
if analiz_btn:
    st.session_state.js_sync_counter += 1   # Force fresh JS read
    st.session_state.pending_analysis = True
    st.rerun()

# 2. AŞAMA: Rerun sonrası — state taze, pipeline çalıştır
if st.session_state.get("pending_analysis", False):
    st.session_state.pending_analysis = False  # flag'i kapat

    user_input = {
        "marka": marka, "seri": seri, "model": model_adi,
        "yil": yil, "kilometre": km,
        "il": il, "renk": renk,
        "yakit_tipi": yakit, "vites_tipi": vites,
        "kasa_tipi": kasa, "cekis": cekis, "kimden": kimden,
        "motor_hacmi_num": motor_hacmi, "motor_gucu_num": motor_gucu,
        "tramer_tutari": tramer,
        "liste_fiyati": liste_fiyati,
        "parca_durumlari": dict(st.session_state.parca_durumlari),  # ✅ TAZE STATE
    }

    with st.spinner("🤖 Yapay zekâ aracı analiz ediyor..."):
        try:
            st.session_state.result = run_pipeline(user_input)
            
            # Debug: state'in doğru gönderildiğini logla
            hasarli_sayi = sum(1 for v in user_input["parca_durumlari"].values() 
                                if v in ["boyali", "degismis", "lokal_boyali"])
            st.success(f"✅ Analiz tamamlandı! ({hasarli_sayi} hasarlı parça hesaba katıldı)")
        except Exception as e:
            st.error(f"❌ Analiz hatası: {e}")
            st.session_state.result = None

# ════════════════════════════════════════════════════════════════
# SONUÇLAR
# ════════════════════════════════════════════════════════════════
if st.session_state.result is not None:
    result = st.session_state.result

    st.markdown("# 📊 Analiz Sonuçları")

    # ───── Model 1: Fiyat ─────
    m1 = result["model1"]
    tahmin = m1["tahmini_piyasa_fiyati"]
    liste = m1["liste_fiyati"]
    fark = m1["fiyat_fark_pct"]

    st.markdown(f"""
    <div class="result-card">
        <h3>💰 Adım 1 — Piyasa Değeri Tahmini</h3>
        <p style="color:#94a3b8;">Yapay zekâmız bu aracın özelliklerine göre piyasa değerini hesapladı:</p>
        <div class="price-label">TAHMİNİ PİYASA DEĞERİ</div>
        <div class="price-big">{tahmin:,.0f} TL</div>
    </div>
    """, unsafe_allow_html=True)

    if liste is not None:
        # Fark yorumu
        if fark < -15:
            fark_emoji = "🟢"
            fark_yorum = "<b>Piyasanın altında</b> — fiyat avantajlı görünüyor"
            fark_color = "#16a34a"
        elif fark > 15:
            fark_emoji = "🔴"
            fark_yorum = "<b>Piyasanın üstünde</b> — fiyat yüksek görünüyor"
            fark_color = "#dc2626"
        else:
            fark_emoji = "🟡"
            fark_yorum = "<b>Piyasaya yakın</b> — makul bir fiyat"
            fark_color = "#f59e0b"

        st.markdown(f"""
        <div class="result-card">
            <h3>📋 Liste Fiyatı Karşılaştırması</h3>
            <table style="width:100%;border-spacing:0;color:#e2e8f0;">
                <tr>
                    <td style="padding:0.5rem 0;color:#94a3b8;">İlan / Liste Fiyatı:</td>
                    <td style="text-align:right;font-weight:700;font-size:1.2rem;">{liste:,.0f} TL</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem 0;color:#94a3b8;">Tahmini Piyasa Değeri:</td>
                    <td style="text-align:right;font-weight:700;font-size:1.2rem;">{tahmin:,.0f} TL</td>
                </tr>
                <tr style="border-top:1px solid #2d3348;">
                    <td style="padding:0.7rem 0;color:#94a3b8;">Fark:</td>
                    <td style="text-align:right;font-weight:800;font-size:1.4rem;color:{fark_color};">
                        {fark:+.1f}% {fark_emoji}
                    </td>
                </tr>
            </table>
            <div class="info-box" style="margin-top:1rem;border-left-color:{fark_color};">
                {fark_emoji} {fark_yorum}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ───── Model 2: Satış Hızı ─────
    m2 = result["model2"]
    p_hizli = m2["olasilik_hizli"]
    p_yavas = m2["olasilik_yavas"]
    hiz_emoji = "⚡" if m2["tahmin"] == "Hizli" else "🐢"
    hiz_yorum = ("Bu araç piyasada <b>hızlı satılır</b> — talep yüksek görünüyor."
                 if m2["tahmin"] == "Hizli" else
                 "Bu araç piyasada <b>yavaş satılabilir</b> — alıcı bulması zaman alabilir.")

    st.markdown(f"""
    <div class="result-card">
        <h3>⏱️ Adım 2 — Satış Hızı Tahmini</h3>
        <p style="color:#94a3b8;">Bu araç ne kadar sürede satılır?</p>
        <div style="text-align:center;font-size:3rem;margin:1rem 0;">{hiz_emoji}</div>
        <div style="text-align:center;font-size:1.8rem;font-weight:800;color:{'#16a34a' if m2['tahmin']=='Hizli' else '#f59e0b'};">
            {m2['tahmin'].upper()}
        </div>
        <div class="info-box">
            {hiz_yorum}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Olasılık barları
    st.markdown("**📊 Olasılık Dağılımı**")
    cba, cbb = st.columns(2)
    with cba:
        st.markdown(f"⚡ **Hızlı satış**: {p_hizli:.1%}")
        st.progress(p_hizli)
    with cbb:
        st.markdown(f"🐢 **Yavaş satış**: {p_yavas:.1%}")
        st.progress(p_yavas)

    # ───── Model 3: Fırsat Kategorisi ─────
    m3 = result["model3"]
    firsat = m3["firsat_kategorisi"]
    firsat_olasiliklar = m3["olasiliklar"]

    style_map = {
        "Altin_Firsat": {"emoji":"🟢","label":"ALTIN FIRSAT","class":"badge-altin",
                         "desc":"Bu araç fiyat–değer dengesi açısından <b>çok avantajlı</b>. Piyasanın altında ve hızlı satış beklentisi var."},
        "Tuzak":        {"emoji":"⚠️","label":"TUZAK","class":"badge-tuzak",
                         "desc":"Fiyat <b>cazip görünüyor</b> ama satış yavaş. <b>Dikkat!</b> Gizli bir sorun olabilir — detaylı incele."},
        "Premium":      {"emoji":"💎","label":"PREMIUM","class":"badge-premium",
                         "desc":"Fiyat piyasa üstünde ama <b>popüler bir model</b> — yine de hızlı satılır. Marka değeri var."},
        "Riskli":       {"emoji":"🔴","label":"RİSKLİ","class":"badge-riskli",
                         "desc":"Hem <b>pahalı</b> hem <b>yavaş satılıyor</b>. Yatırım açısından kaçınılması önerilir."},
    }
    s = style_map.get(firsat, {"emoji":"❓","label":firsat,"class":"badge-altin","desc":""})

    st.markdown(f"""
    <div class="result-card" style="text-align:center;">
        <h3>🎯 Adım 3 — Fırsat Kategorisi (Nihai Karar)</h3>
        <p style="color:#94a3b8;">3 modelin birleşik analizine göre bu araç:</p>
        <div style="font-size:5rem;margin:1rem 0;">{s['emoji']}</div>
        <div class="badge {s['class']}" style="font-size:1.6rem;padding:1rem 2rem;">
            {s['label']}
        </div>
        <div class="info-box" style="margin-top:1.5rem;text-align:left;">
            {s['desc']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 sınıf olasılık grafiği
    st.markdown("**📊 Tüm Kategorilerin Olasılıkları**")
    sirali = sorted(firsat_olasiliklar.items(), key=lambda x: -x[1])
    for kat, olasilik in sirali:
        st_info = style_map.get(kat, {"emoji":"•","label":kat})
        col_lbl, col_bar = st.columns([1, 3])
        with col_lbl:
            st.markdown(f"{st_info['emoji']} **{st_info['label']}**")
        with col_bar:
            st.progress(olasilik)
            st.caption(f"{olasilik:.1%}")

    # ───── SHAP Açıklaması ─────
    st.markdown("---")
    st.markdown("## 🔍 Karar Açıklaması (Yapay Zekâ Neden Böyle Karar Verdi?)")
    st.markdown('<div class="info-box">🧠 Aşağıdaki grafik, modelin bu kararı verirken hangi özellikleri ne kadar dikkate aldığını gösterir.</div>', unsafe_allow_html=True)

    try:
        if "model2" in explainers:
            X2 = result["_X2"]

        
            explainer = explainers["model2"]
            shap_values = explainer.shap_values(X2)
            if isinstance(shap_values, list):
                shap_use = shap_values[1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_use = shap_values
                base_val = explainer.expected_value

            # Top 10 feature etkisi
            shap_arr = np.array(shap_use[0]).flatten()
            feat_imp = pd.Series(shap_arr, index=X2.columns).sort_values(key=lambda s: s.abs(), ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["#16a34a" if v < 0 else "#dc2626" for v in feat_imp.values]
            ax.barh(range(len(feat_imp)), feat_imp.values[::-1], color=colors[::-1])
            ax.set_yticks(range(len(feat_imp)))
            ax.set_yticklabels(feat_imp.index[::-1], color="white")
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_xlabel("Etki Yönü ve Büyüklüğü", color="white")
            ax.set_title("Bu Aracı Etkileyen TOP 10 Faktör", color="white", pad=15)
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0f172a")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#475569")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            <div class="info-box">
                🟢 <b>Yeşil çubuklar</b> → "Hızlı satış" yönünde etkileyen faktörler<br>
                🔴 <b>Kırmızı çubuklar</b> → "Yavaş satış" yönünde etkileyen faktörler<br>
                Çubuk uzunluğu, o faktörün karara olan <b>etki gücünü</b> gösterir.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ SHAP açıklayıcısı yüklü değil")
    except Exception as e:
        st.warning(f"SHAP grafiği oluşturulamadı: {e}")

    # ───── Performans Bilgileri ─────
    with st.expander("📈 Modellerin Doğruluk Oranları"):
        perf = extra_meta.get("performance", {})
        cmm1, cmm2, cmm3 = st.columns(3)
        with cmm1:
            st.metric("💰 Fiyat Modeli (R²)",
                      f"{perf.get('model1', {}).get('r2', 0):.3f}",
                      help="1.00'a ne kadar yakınsa o kadar iyi tahmin yapıyor.")
            st.metric("Hata Oranı (MAPE)",
                      f"%{perf.get('model1', {}).get('mape', 0):.2f}")
        with cmm2:
            st.metric("⏱️ Hız Modeli (F1)",
                      f"{perf.get('model2', {}).get('f1', 0):.3f}",
                      help="0.50 üzeri iyidir, 0.65+ çok iyi.")
            st.metric("Doğruluk",
                      f"%{perf.get('model2', {}).get('accuracy', 0)*100:.1f}")
        with cmm3:
            st.metric("🎯 Fırsat Modeli (F1)",
                      f"{perf.get('model3', {}).get('f1_macro', 0):.3f}",
                      help="4 sınıf için 0.65+ başarılıdır.")
            st.metric("ROC AUC",
                      f"{perf.get('model3', {}).get('roc_auc_ovr', 0):.3f}")


# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;padding:2rem 0;font-size:0.9rem;">
    🚗 <b>Araç Pazarı ML Pipeline</b> — Üniversite Final Projesi<br>
    💡 3 ML Modeli (Fiyat → Satış Hızı → Fırsat) ile akıllı analiz<br>
    📊 ****.com verisi · LightGBM + XGBoost + Stacking Ensemble · SHAP açıklaması
</div>
""", unsafe_allow_html=True)