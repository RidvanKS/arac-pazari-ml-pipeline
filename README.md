# 🚗 İkinci El Otomobil Akıllı Asistanı

Türk ikinci el otomobil pazarındaki **~500.000 ilan** üzerinde eğitilmiş, üç farklı makine öğrenmesi modelini tek arayüzde birleştiren akıllı alıcı asistanı. Bir aracın **gerçek piyasa fiyatını**, **ne kadar sürede satılabileceğini** ve **fırsat mı yoksa tuzak mı** olduğunu tahmin eder.

> 🎓 Hitit Üniversitesi Bilgisayar Mühendisliği 

🔗 **Canlı Demo:** [ridvan.streamlit.app](#)

---

## 🧠 Modeller

Sistem, sıralı çalışan üç bağımsız modelden oluşur:

| # | Model | Algoritma | Görev |
|---|-------|-----------|-------|
| 1 | **Fiyat Tahmini** | LightGBM | İlanın olması gereken piyasa fiyatını tahmin eder |
| 2 | **Satış Hızı Tahmini** | XGBoost | Aracın kaç günde satılabileceğini öngörür (F1 ≈ 0.56) |
| 3 | **Fırsat / Tuzak Sınıflandırıcı** | Stacking Ensemble | İlanı 5 sınıftan birine yerleştirir: **Altın Fırsat, Premium, Piyasa Uyumlu, Riskli, Tuzak** |

Model 3, ilk iki modelin çıktılarını da girdi olarak kullanarak çok katmanlı bir karar mantığı kurar.

---

## ✨ Öne Çıkan Özellikler

- 🎯 **Üç boyutlu analiz** — fiyat, hız ve risk birlikte değerlendirilir
- 📊 **SHAP açıklanabilirliği** — modelin neden öyle karar verdiği görselleştirilir
- 🖥️ **Streamlit arayüzü** — teknik olmayan kullanıcılar için sade ve anlaşılır
- 🛡️ **Alıcı odaklı yaklaşım** — tuzak ilanlardan kaçınmaya yardımcı olur

---

## 🛠️ Kullanılan Teknolojiler

- **Dil:** Python 3.x
- **ML:** LightGBM, XGBoost, scikit-learn
- **Açıklanabilirlik:** SHAP (XGBoost 2.x native `pred_contribs`)
- **Arayüz:** Streamlit
- **Veritabanı:** PostgreSQL
- **Dağıtım:** Streamlit Community Cloud

---

## 🚀 Kurulum ve Çalıştırma

```bash

git clone https://github.com/RidvanKS/arac-pazari-ml-pipeline.git
cd arac-pazari-ml-pipeline


python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
streamlit run app.py
```

---

## 📂 Proje Yapısı

```
.
├── app.py                  # Streamlit arayüzü
├── data_exports/           # İşlenmiş veri setleri
├── web_bundle/             # Web kaynakları
├── requirements.txt        # Python bağımlılıkları
├── runtime.txt             # Streamlit Cloud Python sürümü
└── .devcontainer/          # Geliştirme ortamı yapılandırması
```

---

## 📈 Çözülen Teknik Problemler

Proje boyunca karşılaşılan ve çözülen kritik problemler:

- **XGBoost 2.x uyumsuzluğu** — SHAP açıklayıcısı yeni sürümle çalışmıyordu; çözüm olarak XGBoost'un native `pred_contribs` parametresi entegre edildi.
- **Satış süresi censoring hatası** — Model 2'de ilan süresi yanlış başlangıç tarihinden hesaplanıyordu; veri akışı düzeltildi.
- **Model 3 sınıf tasarım hatası** — "Normal" sınıfı eğitime dahil edilmediği için olasılık dağılımları yanıltıcıydı; sınıf yapısı yeniden kurgulandı.

---

## 👤 Geliştirici

**Rıdvan Koçak**
Hitit Üniversitesi — Bilgisayar Mühendisliği

- 📧 kocak.ridvan@hotmail.com
- 🐙 [github.com/RidvanKS](https://github.com/RidvanKS)
