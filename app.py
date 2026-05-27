import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Stunting Dashboard", layout="wide")

# ======================
# LOAD DATA & MODEL
# ======================
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/data_final.csv')

@st.cache_resource
def load_model():
    return pickle.load(open('model/model_stunting.pkl', 'rb'))

df = load_data()
model = load_model()

# ======================
# SESSION STATE
# ======================
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# ======================
# API CUACA
# ======================
def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=-6.7&longitude=108.5&current_weather=true"
        res = requests.get(url)
        data = res.json()
        temp = data['current_weather']['temperature']
        wind = data['current_weather']['windspeed']
    except:
        temp, wind = 0, 0
    return temp, wind

# ======================
# SIDEBAR
# ======================
menu = st.sidebar.radio("Navigation", [
    "Overview",
    "Data",
    "Environment",
    "Prediction",
    "Insight"
])

# ======================
# OVERVIEW
# ======================
if menu == "Overview":
    st.title("Stunting Prediction Dashboard")

    # tanggal realtime
    now = datetime.now()
    st.caption(f"Tanggal: {now.strftime('%d-%m-%Y %H:%M:%S')}")

    # gabung data
    df_hist = pd.DataFrame(st.session_state.riwayat)

    if not df_hist.empty:
        df_hist['status_gizi'] = df_hist['hasil'].map({
            "Tidak Stunting": 0,
            "Stunting": 1
        })
        df_all = pd.concat([df[['status_gizi']], df_hist[['status_gizi']]])
    else:
        df_all = df[['status_gizi']]

    total = len(df_all)
    stunting = df_all['status_gizi'].sum()
    normal = total - stunting

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", total)
    col2.metric("Tidak Stunting", int(normal))
    col3.metric("Stunting", int(stunting))

    # PIE (kecil)
    st.subheader("Distribusi Status Gizi")
    status_map = {0: "Tidak Stunting", 1: "Stunting"}
    pie = df_all['status_gizi'].map(status_map).value_counts()

    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(pie, labels=pie.index, autopct='%1.1f%%')
    st.pyplot(fig)

    # fungsi bar biar semua kategori muncul
    def plot_bar(data, kategori, title):
        data = data.value_counts().reindex(kategori, fill_value=0)
        fig, ax = plt.subplots()
        ax.bar([str(i) for i in data.index], data.values)

        for i, v in enumerate(data.values):
            ax.text(i, v + 0.5, str(v), ha='center')

        ax.set_title(title)
        st.pyplot(fig)

    st.subheader("Distribusi Variabel Dataset")

    col1, col2 = st.columns(2)

    with col1:
        plot_bar(df['pendidikan_ibu'], [0,1,2,3], "Pendidikan Ibu")
        st.caption("0=Tidak Sekolah | 1=SD | 2=SMP | 3=SMA")

        plot_bar(df['pendapatan_keluarga'], [0,1,2], "Pendapatan")
        st.caption("0=Rendah | 1=Sedang | 2=Tinggi")

        plot_bar(df['frekuensi_makan'], [0,1,2], "Frekuensi Makan")
        st.caption("0=Jarang | 1=Cukup | 2=Sering")

    with col2:
        plot_bar(df['sanitasi_rumah'], [0,1], "Sanitasi")
        st.caption("0=Buruk | 1=Baik")

        plot_bar(df['air_bersih'], [0,1], "Air Bersih")
        st.caption("0=Tidak Layak | 1=Layak")

        plot_bar(df['asi_eksklusif'], [0,1], "ASI Eksklusif")
        st.caption("0=Tidak | 1=Ya")

# ======================
# DATA
# ======================
elif menu == "Data":
    st.title("Dataset")

    st.subheader("📊 Data Balita")
    st.dataframe(df[['umur_bulan','jenis_kelamin','tinggi_badan_cm','status_gizi']])

    st.markdown("### Keterangan Data Balita")
    st.table(pd.DataFrame({
        "Variabel": ["Jenis Kelamin", "Status Gizi"],
        "0": ["Perempuan", "Tidak Stunting"],
        "1": ["Laki-laki", "Stunting"]
    }))

    st.subheader("📋 Data Survey")
    st.dataframe(df[
        ['pendidikan_ibu','pendapatan_keluarga',
         'asi_eksklusif','frekuensi_makan',
         'sanitasi_rumah','air_bersih']
    ])

    st.markdown("### Keterangan Data Survey")
    st.table(pd.DataFrame({
        "Variabel": ["Pendidikan Ibu","Pendapatan","ASI","Frekuensi","Sanitasi","Air"],
        "0": ["Tidak Sekolah","Rendah","Tidak","Jarang","Buruk","Tidak Layak"],
        "1": ["SD","Sedang","Ya","Cukup","Baik","Layak"],
        "2": ["SMP","Tinggi","-","Sering","-","-"],
        "3": ["SMA","-","-","-","-","-"]
    }))

# ======================
# ENVIRONMENT
# ======================
elif menu == "Environment":
    st.title("Kondisi Lingkungan")

    now = datetime.now()
    st.caption(f"Tanggal: {now.strftime('%d-%m-%Y %H:%M:%S')}")

    temp, wind = get_weather()

    col1, col2 = st.columns(2)
    col1.metric("Temperature", f"{temp} °C")
    col2.metric("Wind Speed", f"{wind} km/h")

# ======================
# PREDICTION
# ======================
elif menu == "Prediction":
    st.title("Prediction")

    temp, wind = get_weather()

    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input("Umur (bulan)", 0, 60)
        jk = st.selectbox("Jenis Kelamin", ["0 (Perempuan)","1 (Laki-laki)"])
        tinggi = st.number_input("Tinggi Badan (cm)")
        pendidikan = st.selectbox("Pendidikan Ibu", [
            "0 (Tidak Sekolah)","1 (SD)","2 (SMP)","3 (SMA)"
        ])
        pendapatan = st.selectbox("Pendapatan", [
            "0 (Rendah)","1 (Sedang)","2 (Tinggi)"
        ])

    with col2:
        asi = st.selectbox("ASI Eksklusif", ["0 (Tidak)","1 (Ya)"])
        frek = st.selectbox("Frekuensi Makan", [
            "0 (Jarang)","1 (Cukup)","2 (Sering)"
        ])
        sanitasi = st.selectbox("Sanitasi", ["0 (Buruk)","1 (Baik)"])
        air = st.selectbox("Air Bersih", ["0 (Tidak Layak)","1 (Layak)"])

    jk = int(jk[0])
    pendidikan = int(pendidikan[0])
    pendapatan = int(pendapatan[0])
    asi = int(asi[0])
    frek = int(frek[0])
    sanitasi = int(sanitasi[0])
    air = int(air[0])

    kelompok_umur = umur // 12
    rasio = tinggi / umur if umur != 0 else 0

    st.info(f"Suhu: {temp}°C | Angin: {wind} km/h")

    if st.button("Run Prediction"):

    # input ke model
        data = [[umur,jk,tinggi,pendidikan,pendapatan,
                asi,frek,sanitasi,air,temp,wind,kelompok_umur,rasio]]

        # prediksi
        pred = model.predict(data)[0]
        hasil = "Stunting" if pred == 1 else "Tidak Stunting"

        # alasan
        alasan = []

        if tinggi < 80:
            alasan.append("Tinggi badan rendah menunjukkan indikasi gangguan pertumbuhan")

        if asi == 0:
            alasan.append("Tidak mendapatkan ASI eksklusif dapat mengurangi nutrisi")

        if sanitasi == 0:
            alasan.append("Sanitasi buruk meningkatkan risiko infeksi")

        # suhu
        if temp > 30:
            alasan.append("Suhu tinggi (musim panas) dapat menyebabkan dehidrasi")
        elif temp < 22:
            alasan.append("Suhu rendah (musim hujan/dingin) dapat menurunkan imun")

        # angin
        if wind > 20:
            alasan.append("Angin kencang meningkatkan penyebaran penyakit")
        elif wind < 5:
            alasan.append("Sirkulasi udara buruk karena angin rendah")

        # simpan
        st.session_state.riwayat.append({
            "tanggal": datetime.now(),
            "hasil": hasil,
            "suhu": temp,
            "angin": wind,
            "alasan": alasan
        })

        st.success(hasil)

# ======================
# INSIGHT
# ======================
elif menu == "Insight":
    st.title("Insight")

    if len(st.session_state.riwayat) == 0:
        st.warning("Belum ada data")
    else:
        df_hist = pd.DataFrame(st.session_state.riwayat)
        df_hist['tanggal'] = pd.to_datetime(df_hist['tanggal'])

        st.dataframe(df_hist)

        idx = st.number_input("Pilih Index", 0, len(df_hist)-1)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Detail"):
                data = df_hist.iloc[idx]

                st.write("Hasil:", data["hasil"])
                st.write("Tanggal:", data["tanggal"])
                st.write("Suhu:", data["suhu"], "°C")
                st.write("Wind Speed:", data["angin"], "km/h")
                st.write("Alasan:", data["alasan"])

        with col2:
            if st.button("Hapus"):
                st.session_state.riwayat.pop(idx)
                st.rerun()

        # faktor penyebab
        all_alasan = []
        for item in st.session_state.riwayat:
            all_alasan.extend(item["alasan"])

        if len(all_alasan) > 0:
            count = Counter(all_alasan)
            df_vis = pd.DataFrame(count.items(), columns=["Faktor","Jumlah"])
            st.bar_chart(df_vis.set_index("Faktor"))