from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path("best.pt")


st.set_page_config(
    page_title="Deteksi Jenis Sampah",
    page_icon="recycle",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def predict_waste(image):
    model = load_model()
    results = model.predict(
        source=image,
        imgsz=224,
        verbose=False,
    )

    result = results[0]
    probs = result.probs
    top1 = probs.top1
    confidence = float(probs.top1conf)
    prediction = model.names[top1]

    all_probs = {}
    for idx, score in enumerate(probs.data.cpu().numpy()):
        all_probs[model.names[idx]] = round(float(score) * 100, 2)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": all_probs,
    }


def get_confidence_style(confidence):
    if confidence < 30:
        return "#f62828", "Kepercayaan Rendah"
    if confidence <= 50:
        return "#ef9f1d", "Kepercayaan Sedang"
    return "#2BB732", "Kepercayaan Tinggi"


def show_recommendation(prediction):
    st.divider()
    st.subheader("Rekomendasi Penanganan")

    if prediction == "Organic":
        st.info(
            """
            **Sampah ini termasuk sampah organik.**

            - Buang ke tempat sampah organik.
            - Dapat diolah menjadi kompos atau pupuk.
            - Pisahkan dari sampah anorganik agar proses pengolahan lebih mudah.
            """
        )
    elif prediction == "Recyclable":
        st.success(
            """
            **Sampah ini dapat didaur ulang.**

            - Buang ke tempat sampah daur ulang.
            - Bersihkan terlebih dahulu jika memungkinkan.
            - Dapat diolah kembali menjadi produk baru.
            """
        )
    elif prediction == "Non-Recyclable":
        st.warning(
            """
            **Sampah ini tidak dapat didaur ulang dengan proses umum.**

            - Buang ke tempat sampah residu.
            - Jangan dicampur dengan sampah yang masih dapat didaur ulang.
            - Kurangi penggunaan produk sejenis untuk mengurangi limbah.
            """
        )
    elif prediction == "Hazardous":
        st.error(
            """
            **Sampah ini termasuk limbah B3 (Bahan Berbahaya dan Beracun).**

            - Jangan dibuang bersama sampah rumah tangga biasa.
            - Simpan secara terpisah dan aman.
            - Serahkan ke fasilitas atau tempat pengelolaan limbah B3 yang tersedia.
            """
        )
    else:
        st.info("Belum ada rekomendasi khusus untuk kelas ini.")


def show_probabilities(probabilities):
    st.divider()
    st.subheader("Probabilitas Semua Kelas")

    sorted_probs = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, value in sorted_probs:
        st.write(f"**{class_name}** : {value}%")
        st.progress(min(value / 100, 1.0))


st.title("Sistem Deteksi Jenis Sampah")
st.divider()
st.subheader("Deteksi Cepat, Pilah Tepat!")

st.sidebar.title("Tentang Sistem")
st.sidebar.info(
    """
    Frontend : Streamlit

    Model : YOLOv8 Classification

    Mode : Prediksi langsung tanpa backend API
    """
)

if not MODEL_PATH.exists():
    st.error("File model best.pt tidak ditemukan di folder project.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Preview")
        st.divider()
        st.markdown("#### Informasi Jenis Sampah")
        st.markdown("**Sampah Organik**: Sampah yang berasal dari sisa makhluk hidup.")
        st.markdown(
            "**Sampah Hazardous/B3**: Sampah yang mengandung zat beracun, "
            "mudah terbakar, atau korosif."
        )
        st.markdown(
            "**Sampah Recyclable**: Sampah anorganik yang bernilai ekonomis "
            "dan dapat diolah kembali menjadi barang baru."
        )
        st.markdown(
            "**Sampah Non-Recyclable**: Sampah yang tidak memiliki nilai "
            "ekonomis atau tidak bisa diolah kembali dengan teknologi daur ulang saat ini."
        )
        st.image(image, use_container_width=True)

    with col2:
        st.header("Hasil Prediksi")
        st.divider()

        if st.button("Deteksi", type="primary"):
            try:
                with st.spinner("Memproses gambar..."):
                    result = predict_waste(image)

                prediction = result["prediction"]
                confidence = result["confidence"]
                text_color, status = get_confidence_style(confidence)

                st.success(f"Jenis Sampah : {prediction}")
                st.markdown(
                    f"""
                    <h4>Confidence</h4>
                    <h2 style="color:{text_color};">{confidence}%</h2>
                    <p style="color:{text_color};">{status}</p>
                    """,
                    unsafe_allow_html=True,
                )

                show_recommendation(prediction)
                show_probabilities(result["probabilities"])
            except Exception as error:
                st.error(f"Error: {error}")
else:
    st.info("Silakan upload gambar terlebih dahulu.")

st.divider()
st.caption("YOLOv8 Classification - Streamlit")
