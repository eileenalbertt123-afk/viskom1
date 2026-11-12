import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

   from detector import detect, CLASSES, DEFAULT_CONF
from utils import resize_max, draw_detections, count_per_class

st.set_page_config(page_title="Deteksi Sampah - Bank Sampah", layout="wide")

st.title("Deteksi & Klasifikasi Sampah pada Tumpukan")
st.write("Unggah foto, lalu sistem menandai tiap objek dan menghitung per kategori.")


col_in, col_out = st.columns(2)

with col_in:
    file = st.file_uploader("Foto tumpukan sampah", type=["jpg", "jpeg", "png"])
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.025)
    run = st.button("Deteksi", type="primary")
    if file is not None:
        image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        st.image(image, caption="Foto asli", use_container_width=True)

with col_out:
    if run:
        if file is None:
            st.error("Silakan unggah foto tumpukan sampah dulu.")
        else:
            with st.spinner("Mendeteksi..."):
                img = resize_max(image)
                dets = detect(img, conf)
                out_img = draw_detections(img, dets)
                summary = count_per_class(dets, CLASSES)
                detail = [[d["label"], d["confidence"], str(d["bbox"])] for d in dets]

            st.image(out_img, caption="Hasil deteksi", use_container_width=True)
            st.subheader("Ringkasan")
            st.dataframe(pd.DataFrame(summary, columns=["Kategori", "Jumlah"]),
                         hide_index=True, use_container_width=True)
            st.subheader("Detail")
            st.dataframe(pd.DataFrame(detail, columns=["Label", "Confidence", "BBox"]),
                         hide_index=True, use_container_width=True)

with st.expander("Keterbatasan sistem"):
    st.markdown(
        "- Hanya 5 kategori: plastik, kertas/kardus, logam, organik, kaca\n"
        "- Tumpukan sangat padat atau tertutup total tidak dijamin terdeteksi\n"
        "- Tidak membedakan sub-jenis plastik (PET, HDPE, PP)\n"
        "- Hanya berbasis citra, tanpa sensor berat atau material\n"
        "- Performa terbaik pada foto terang, fokus, dan sudut wajar\n"
        "- Prototipe, belum diuji langsung di bank sampah"
    )