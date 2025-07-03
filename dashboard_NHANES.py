import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import io
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from io import BytesIO

# ----------------------
# Load Model dan Template
# ----------------------
@st.cache_resource
def load_model():
    with open("NHANES_ALLMERGED_smoteenn_model_raw(terbaru).pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_template():
    df = pd.read_excel("NHANES_ALLMERGED_smoteenn.xlsx")
    df = df.drop(columns=["target"], errors="ignore")
    return df

model = load_model()
X_reference = load_template()
model_features = [col.strip() for col in model.get_booster().feature_names]

@st.cache_data
def create_template_df():
    # DataFrame kosong dengan header yang tepat
    return pd.DataFrame(columns=model_features)

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Mengonversi DataFrame ke bytes Excel (XLSX) untuk di-download.
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Template")
    return buffer.getvalue()

# ----------------------
# Feature Descriptions
# ----------------------
feature_descriptions = {
    "RXQASA_L_RXQ510": "Apakah pasien pernah mendapat diagnosis asma?",
    "DIETARY DR1IFF_L_DR1SELE": "Jumlah asupan energi dari makanan (hari 1)",
    "BPQ_L_BPQ020": "Apakah pasien pernah diberitahu punya tekanan darah tinggi?",
    "DEMOGRAFI_INDFMPIR": "Rasio pendapatan keluarga terhadap ambang kemiskinan",
    "DEMOGRAFI_RIDAGEYR": "Usia pasien (tahun)",
    "PAQ_L_PAD680": "Seberapa sering pasien melakukan aktivitas fisik berat?",
    "BPQ_L_BPQ101D": "Minum obat tekanan darah saat ini",
    "SLQ_L_SLD012": "Apakah pasien mengalami gangguan tidur?",
    "DEMOGRAFI_DMDeduc2": "Tingkat pendidikan tertinggi yang diselesaikan",
    "BMX_L_BMXHIP": "Lingkar pinggul (cm)",
    "PAQ_L_PAD790Q": "Durasi aktivitas ringan per hari",
    "BMX_L_BMXWAIST": "Lingkar pinggang (cm)",
    "DIQ_L_DIQ010": "Apakah pasien pernah didiagnosis diabetes?",
    "DEMOGRAFI_SDMVSTRA": "Strata survei NHANES",
    "DEMOGRAFI_WTINT2YR": "Bobot sampel survei 2 tahun",
    "CBC_L_LBXPLTSI": "Jumlah trombosit dalam darah",
    "TCHOL_L_LBXTC": "Kadar kolesterol total (mg/dL)",
    "SMQ_L_SMQ020": "Status merokok",
    "BMX_L_BMDSTATS": "Status massa tulang",
    "PAQ_L_PAD810Q": "Apakah pasien jalan kaki minimal 10 menit?",
}

nhanes_links = {
    "DEMOGRAFI": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm",
    "DIETARY (DR1IFF_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.htm",
    "DIETARY (DR1TOT_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1TOT_L.htm",
    "Body Measures (BMX_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm",
    "Blood Pressure - Oscillometric Measurements (BPXO_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.htm",
    "Complete Blood Count with 5-Part Differential in Whole Blood (CBC_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm",
    "Cholesterol-High-Density Lipoprotein (HDL_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.htm",
    "High-Sensitivity C-Reactive Protein (HSCRP_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HSCRP_L.htm",
    "Cholesterol - Total (TCHOL_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.htm",
    "Blood Pressure & Cholesterol (BPQ_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.htm",
    "Diabetes (DIQ_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm",
    "Smoking - Cigarette Use (SMQ_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.htm",
    "Physical Activity (PAQ_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.htm",
    "Preventive Aspirin Use (RXQASA_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/RXQASA_L.htm",
    "Sleep Disorders (SLQ_L)": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.htm",
}

# ----------------------
# Judul dan Threshold
# ----------------------
st.title("🩺 Prediksi Risiko Penyakit Jantung")
st.markdown("Silakan download template terlebih dahulu")

# threshold = st.slider("🔧 Ubah Threshold Prediksi", 0.0, 1.0, 0.5, step=0.01)
# Threshold tetap
threshold = 0.07

template_bytes = df_to_excel_bytes(create_template_df())
st.download_button(
    label="📥 Download Template Excel",
    data=template_bytes,
    file_name="template_input_pasien.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("Silakan pilih metode input data pasien dan klik tombol **Prediksi Risiko** untuk melihat hasil prediksi")

# ----------------------
# Input Data Pasien
# ----------------------
st.subheader("📋 Input Data Pasien")

input_method = st.radio("Pilih metode input data:", ["Upload Excel", "Input Manual (Sampel)"])

df_input = None
if input_method == "Upload Excel":
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx):", type=["xlsx"])
    if uploaded_file:
        df_input = pd.read_excel(uploaded_file)
        df_input.columns = df_input.columns.str.strip()
elif input_method == "Input Manual (Sampel)":
    df_input = X_reference.sample(10).reset_index(drop=True)
    df_input = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# ----------------------
# Tombol Prediksi
# ----------------------
if df_input is not None:
    if st.button("🚀 Prediksi Risiko"):
        if set(model_features).issubset(set(df_input.columns)):
            X_input = df_input[model_features]
            probabilities = model.predict_proba(X_input)[:, 1]
            predictions = (probabilities > threshold).astype(int)

            result_df = df_input.copy()
            result_df['Probabilitas Risiko (%)'] = probabilities * 100
            result_df['Hasil Prediksi'] = np.where(predictions == 1, "Berisiko", "Tidak Berisiko")

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(result_df)

            # Visualisasi Distribusi
            st.subheader("📊 Visualisasi Distribusi Prediksi")
            label_counts = pd.Series(predictions).value_counts().sort_index()
            label_counts.index = label_counts.index.map({0: "Tidak Berisiko", 1: "Berisiko"})

            fig_bar, ax_bar = plt.subplots()
            bars = ax_bar.bar(label_counts.index, label_counts.values, color=["green", "red"])
            ax_bar.set_ylabel("Jumlah Pasien")
            ax_bar.set_title("Distribusi Prediksi Risiko")

            for bar in bars:
                height = bar.get_height()
                ax_bar.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(fig_bar)

            # SHAP Summary Plot
            st.subheader("🔍 Penjelasan Model dengan SHAP")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_input, show=False)
            st.pyplot(fig)

            # Penjelasan Fitur
            # with st.expander("📘 Kamus Fitur (Klik untuk lihat penjelasan nama kolom)"):
            #     for col in X_input.columns:
            #         if col in feature_descriptions:
            #             st.markdown(f"**{col}**: {feature_descriptions[col]}")
            #         else:
            #             st.markdown(f"**{col}**: _(Deskripsi belum tersedia)_")
            with st.expander("📘 Link Dokumentasi Fitur NHANES"):
                st.markdown("Berikut adalah link ke dokumentasi NHANES berdasarkan kategori datanya:")
                for kategori, url in nhanes_links.items():
                    st.markdown(f"- [{kategori}]({url})")


            # # SHAP Force Plot (Pasien Tertentu)
            # st.subheader("👤 SHAP Force Plot untuk Pasien Tertentu")
            # shap.initjs()
            # # Pilih index pasien (baris)
            # selected_index = st.number_input("Pilih index pasien", min_value=0, max_value=len(X_input)-1, step=1)

            # # Buat force plot HTML
            # force_plot = shap.plots.force(
            #     base_value=explainer.expected_value,
            #     shap_values=shap_values[selected_index],
            #     features=X_input.iloc[selected_index],
            #     matplotlib=False
            # )

            # # Tampilkan force plot di Streamlit
            # components.html(shap.getjs() + force_plot.html(), height=300)

            # # Evaluasi Threshold
            # st.subheader("📉 Evaluasi Model terhadap Threshold")
            # thresholds = np.arange(0.0, 1.01, 0.01)
            # precisions, recalls, f1s, accuracies = [], [], [], []

            # for t in thresholds:
            #     y_pred = (probabilities > t).astype(int)
            #     precisions.append(precision_score(predictions, y_pred, zero_division=0))
            #     recalls.append(recall_score(predictions, y_pred, zero_division=0))
            #     f1s.append(f1_score(predictions, y_pred, zero_division=0))
            #     accuracies.append(accuracy_score(predictions, y_pred))

            # fig_metric, ax_metric = plt.subplots()
            # ax_metric.plot(thresholds, precisions, label='Precision')
            # ax_metric.plot(thresholds, recalls, label='Recall')
            # ax_metric.plot(thresholds, f1s, label='F1 Score')
            # ax_metric.plot(thresholds, accuracies, label='Accuracy')
            # ax_metric.axvline(threshold, color='gray', linestyle='--', label=f"Threshold = {threshold}")
            # ax_metric.set_xlabel("Threshold")
            # ax_metric.set_ylabel("Score")
            # ax_metric.set_title("Evaluasi Model terhadap Threshold")
            # ax_metric.legend()
            # st.pyplot(fig_metric)

        else:
            st.warning("❗ File tidak valid. Pastikan semua kolom berikut tersedia di file Anda:")
            st.code(", ".join(model_features))
