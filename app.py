import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ==================================
# KONFIGURASI HALAMAN
# ==================================

st.set_page_config(
    page_title="Koreksi Jawaban",
    layout="wide"
)

# ==================================
# CSS
# ==================================

st.markdown("""
<style>

.main .block-container{
    max-width:100%;
    padding-top:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

div[role="radiogroup"]{
    display:flex !important;
    flex-wrap:nowrap !important;
    flex-direction:row !important;
    gap:2px !important;
}

div[role="radiogroup"] label{
    min-width:45px !important;
    width:45px !important;
    height:30px !important;

    border:1px solid #666 !important;
    border-radius:1px !important;

    padding:0px !important;
    margin:0px !important;

    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
}

div[role="radiogroup"] p{
    font-size:11px !important;
    margin:0px !important;
}

div[role="radiogroup"] p{
    margin:0px !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_model():
    return YOLO("models/omr_yolov8_v1.pt")

model = load_model()

# ==================================
# FUNGSI KOREKSI
# ==================================

def koreksi_gambar(image_path, kunci_jawaban):

    results = model.predict(
        source=image_path,
        conf=0.50,
        save=True,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            detections.append({

                "label": label,
                "x": center_x,
                "y": center_y

            })

    detections.sort(
        key=lambda d: d["x"]
    )

    columns = []

    for i in range(5):

        start = i * 7
        end = start + 7

        columns.append(
            detections[start:end]
        )

    answers = {}

    for col_idx, column in enumerate(columns):

        column.sort(
            key=lambda d: d["y"]
        )

        for row_idx in range(7):

            nomor = col_idx * 7 + row_idx + 1

            if row_idx < len(column):

                answers[nomor] = (
                    column[row_idx]["label"]
                )

            else:

                answers[nomor] = "-"

    benar = 0

    detail = []

    for i in range(35):

        siswa = answers.get(i + 1, "-")

        kunci = kunci_jawaban[i]

        if siswa == kunci:

            benar += 1

            detail.append(
                f"Q{i+1} : BENAR"
            )

        else:

            detail.append(
                f"Q{i+1} : SALAH ({siswa} ≠ {kunci})"
            )

    salah = 35 - benar

    nilai = (
        benar / 35
    ) * 100

    return (
        benar,
        salah,
        nilai,
        detail,
        answers
    )

# ==================================
# JUDUL
# ==================================

st.title(
    "Aplikasi Koreksi Jawaban Pilihan Ganda"
)

# ==================================
# BARIS ATAS
# ==================================

col_upload, col_button = st.columns([4,1])

with col_upload:

    uploaded_file = st.file_uploader(
        "Upload Lembar Jawaban",
        type=["jpg","jpeg","png"]
    )

with col_button:

    st.write("")
    st.write("")

    tombol_koreksi = st.button(
        "KOREKSI",
        use_container_width=True
    )

# ==================================
# PREVIEW GAMBAR
# ==================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        width=400
    )

st.divider()

# ==================================
# INPUT KUNCI JAWABAN
# ==================================

st.subheader("Input Kunci Jawaban")

jawaban_kunci = []

kolom_utama = st.columns(5)

for kolom in range(5):

    with kolom_utama[kolom]:

        for nomor in range(
            kolom * 7 + 1,
            kolom * 7 + 8
        ):

            nomor_col, pilihan_col = st.columns(
                [1, 4],
                gap="small"
            )

            with nomor_col:

                st.markdown(
                    f"""
                    <div style="
                    height:35px;
                    display:flex;
                    align-items:center;
                    justify-content:flex-end;
                    font-weight:bold;
                    padding-right:5px;
                    ">
                    {nomor}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with pilihan_col:

                pilihan = st.radio(
                    "",
                    ["A", "B", "C", "D"],
                    horizontal=True,
                    index=None,
                    key=f"kunci_{nomor}",
                    label_visibility="collapsed"
                )

            jawaban_kunci.append(
                pilihan
            )

st.divider()

# ==================================
# HASIL DETEKSI SISWA
# ==================================

st.subheader(
    "Hasil Deteksi Siswa"
)

hasil_container = st.container()

for i, jawaban in enumerate(jawaban_kunci):

    if jawaban is None:

        st.error(
            f"Kunci nomor {i+1} belum diisi"
        )

        st.stop()

# ==================================
# PROSES KOREKSI
# ==================================

if tombol_koreksi:

    if uploaded_file is None:

        st.error(
            "Upload gambar terlebih dahulu"
        )

    else:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            image_path = tmp.name

        benar, salah, nilai, detail, answers = (
            koreksi_gambar(
                image_path,
                jawaban_kunci
            )
        )

        st.success(
            f"Benar : {benar}"
        )

        st.error(
            f"Salah : {salah}"
        )

        st.info(
            f"Nilai : {nilai:.2f}"
        )

        with hasil_container:

            cols = st.columns(5)

            for nomor in range(1, 36):

                kolom = (nomor - 1) // 7

                siswa = answers.get(
                    nomor,
                    "-"
                )

                kunci = jawaban_kunci[
                    nomor - 1
                ]

                warna = "🟢"

                if siswa != kunci:
                    warna = "🔴"

                with cols[kolom]:

                    st.write(
                        f"{nomor}. {siswa} {warna}"
                    )