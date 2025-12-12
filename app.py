import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background Gradient */
body {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%) !important;
}

/* Main Container */
.main-container {
    background: white;
    padding: 40px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    max-width: 700px;
    margin: auto;
    margin-top: 40px;
}

/* Title */
.title-text {
    text-align: center;
    font-weight: 700;
    font-size: 34px;
    color: #333;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #666;
    margin-bottom: 25px;
}

/* Button Style */
button[kind="primary"] {
    background-color: #6c63ff !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


# ====== LOAD MODEL ======
model, le = pickle.load(open("model.pkl", "rb"))


# ====== LAYOUT ======
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>🌤️ Weather & Outfit Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistem Rekomendasi Cuaca dan Aktivitas Menggunakan Metode Klasifikasi</p>", unsafe_allow_html=True)


# ====== INPUT ======
temp_max = st.number_input("🌡️ Suhu Maksimum (°C)", 0, 50, 24)
temp_min = st.number_input("❄️ Suhu Minimum (°C)", -10, 40, 17)
wind = st.number_input("💨 Kecepatan Angin (m/s)", 0.0, 30.0, 3.22)
precip = st.number_input("🌧️ Curah Hujan (mm)", 0.0, 50.0, 0.10)


# ====== PREDIKSI ======
if st.button("🔍 Prediksi"):
    data = pd.DataFrame({
        "temp_max": [temp_max],
        "temp_min": [temp_min],
        "wind": [wind],
        "precipitation": [precip]
    })

    pred = model.predict(data)[0]
    weather = le.inverse_transform([pred])[0]

    st.success(f"🌦️ Cuaca Diperkirakan: **{weather.upper()}**")

    # Rekomendasi aktivitas & pakaian
    aktivitas = {
        "sun": "Aktivitas outdoor seperti piknik, jogging, atau bersepeda.",
        "rain": "Aktivitas indoor seperti nonton atau membaca buku.",
        "drizzle": "Tetap bisa keluar dengan membawa jaket kecil."
    }

    pakaian = {
        "sun": "Kaos santai, topi, sunscreen.",
        "rain": "Sweater, Payung / jas hujan, sepatu anti air.",
        "drizzle": "Hoodie atau jaket ringan."
    }

    st.info(f"🎯 **Rekomendasi Aktivitas:** {aktivitas.get(weather)}")
    st.warning(f"👕 **Rekomendasi Pakaian:** {pakaian.get(weather)}")

    # ====== DIAGRAM INTERAKTIF (PLOTLY) ======
    st.subheader("📊 Diagram Kondisi Cuaca")

    df_plot = pd.DataFrame({
        "Kategori": ["Suhu Maks", "Suhu Min", "Kecepatan Angin", "Curah Hujan"],
        "Nilai": [temp_max, temp_min, wind, precip]
    })

    fig = px.line(
        df_plot,
        x="Kategori",
        y="Nilai",
        markers=True,
        title="Diagram Interaktif Kondisi Cuaca",
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.2,
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


st.markdown("</div>", unsafe_allow_html=True)
