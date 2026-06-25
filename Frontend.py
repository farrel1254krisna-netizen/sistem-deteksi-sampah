import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(
    page_title="Deteksi Jenis Sampah",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Sistem Deteksi Jenis Sampah")
st.divider()
st.subheader("**Deteksi Cepat, Pilah Tepat!**")

st.sidebar.title("Tentang Sistem")

st.sidebar.info("""
Frontend : Streamlit

Backend : FastAPI

Model : YOLOv8 Classification
""")

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1,col2 = st.columns(2)

    with col1:

        st.header("Preview")
        st.divider()
        st.markdown("#### Informasi Jenis Sampah :")
        st.markdown("**Sampah Organik**: Sampah yang berasal dari sisa makhluk hidup.")
        st.markdown("**Sampah Hazardous/B3**: Sampah yang mengandung zat beracun, mudah terbakar, atau korosif. ")
        st.markdown("**Sampah Recyclable**: Sampah ini biasanya berupa sampah anorganik yang bernilai ekonomis dan dapat diolah kembali menjadi barang baru.")
        st.markdown("**Sampah Non-Recyclable**: Sampah ini tidak memiliki nilai ekonomis atau tidak bisa diolah kembali dengan teknologi daur ulang saat ini.")
        st.image(
            image,
            use_container_width=True
        )   
    with col2:

        st.header("Hasil Prediksi")
        st.divider()
        if st.button("Deteksi"):

            try:

                response = requests.post(
                 API_URL,
                files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
                }   
            )   
                result = response.json()
                st.write("Response diTerima:")
                
                
                if result["success"]:

                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    st.success(
                        f"Jenis Sampah : {prediction}"
                    )
                    if confidence < 30:
                        bg_color = "#ffebee"
                        text_color ="#f62828"
                        status = "Kepercayaan Rendah"
                    elif confidence <= 50:
                        bg_color = "#fff8e1"
                        text_color = "#ef9f1d"
                        status = "Kepercayaan Sedang"
                    else:
                        bg_color = "#e8f5e9"
                        text_color = "#2BB732"
                        status = "Kepercayaan Tinggi"

                    st.markdown(
                        f"""
                            <h4>Confidence</h4>
                            <h2 style="color:{text_color};">
                                {confidence}%
                            </h2>
                            <p style="color:{text_color};">
                                {status}
                            </p>
                        """,
                        unsafe_allow_html=True
                    )
                    st.divider()
                    st.subheader("💡 Rekomendasi Penanganan")

                    if prediction == "Organic":
                        st.info(
                            """
                            **Sampah ini termasuk sampah organik.**
                            
                            🌱 Buang ke tempat sampah organik.
                            
                            🌱 Dapat diolah menjadi kompos atau pupuk.
                            
                            🌱 Pisahkan dari sampah anorganik agar proses pengolahan lebih mudah.
                            """
                        )

                    elif prediction == "Recyclable":
                        st.success(
                            """
                            **Sampah ini dapat didaur ulang.**
                            
                            ♻️ Buang ke tempat sampah daur ulang.
                            
                            ♻️ Bersihkan terlebih dahulu jika memungkinkan.
                            
                            ♻️ Dapat diolah kembali menjadi produk baru.
                            """
                        )

                    elif prediction == "Non-Recyclable":
                        st.warning(
                            """
                            **Sampah ini tidak dapat didaur ulang dengan proses umum.**
                            
                            🗑️ Buang ke tempat sampah residu.
                            
                            🗑️ Jangan dicampur dengan sampah yang masih dapat didaur ulang.
                            
                            🗑️ Kurangi penggunaan produk sejenis untuk mengurangi limbah.
                            """
                        )

                    elif prediction == "Hazardous":
                        st.error(
                            """
                            **Sampah ini termasuk limbah B3 (Bahan Berbahaya dan Beracun).**
                            
                            ⚠️ Jangan dibuang bersama sampah rumah tangga biasa.
                            
                            ⚠️ Simpan secara terpisah dan aman.
                            
                            ⚠️ Serahkan ke fasilitas atau tempat pengelolaan limbah B3 yang tersedia.
                            """
                        )
                
                    st.divider()

                    st.subheader("Probabilitas Semua Kelas")

                    probabilities = result["probabilities"]

                    sorted_probs = sorted(
                        probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    for kelas, nilai in sorted_probs:
                        st.write(f"**{kelas}** : {nilai}%")
                        st.progress(min(nilai / 100, 1.0))

                else:
                    st.error(
                    result.get(
                    "error",
                    "Terjadi kesalahan"
                    )
                )

            except Exception as e:

                st.error(
            f"Error: {str(e)}"
        )
st.divider()
st.caption(
    "YOLOv8 Classification • FastAPI • Streamlit"
)