from ultralytics import YOLO

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(
    "models/omr_yolov8_v1.pt"
)

# ==========================
# PREDIKSI
# ==========================
results = model.predict(
    source="images/IMG_20260524_163355.jpg",
    conf=0.50,
    save=True
)

# ==========================
# AMBIL DETEKSI
# ==========================
detections = []

for result in results:

    boxes = result.boxes

    for box in boxes:

        cls_id = int(box.cls[0])

        label = model.names[cls_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        detections.append({

            "label": label,
            "conf": confidence,
            "x": center_x,
            "y": center_y

        })

# ==========================
# URUTKAN BERDASARKAN X
# ==========================
detections.sort(
    key=lambda d: d["x"]
)

# ==========================
# PASTIKAN 35 DETEKSI
# ==========================
if len(detections) != 35:

    print(
        f"\nERROR! Jumlah deteksi = {len(detections)}"
    )

    print(
        "Harus tepat 35 deteksi."
    )

    exit()

# ==========================
# BAGI MENJADI 5 KOLOM
# ==========================
columns = []

for i in range(5):

    start = i * 7
    end = start + 7

    column = detections[start:end]

    columns.append(column)

# ==========================
# URUTKAN BERDASARKAN Y
# ==========================
answers = {}

question_number = 1

for column in columns:

    column.sort(
        key=lambda d: d["y"]
    )

    for det in column:

        answers[
            question_number
        ] = det["label"]

        question_number += 1

# ==========================
# CETAK HASIL JAWABAN
# ==========================
print("\n===== HASIL JAWABAN =====\n")

for q in sorted(answers):

    print(
        f"Q{q} = {answers[q]}"
    )

# ==========================
# INPUT KUNCI JAWABAN
# ==========================
print("\n==============================")
print("INPUT KUNCI JAWABAN")
print("==============================")

print(
    "\nMasukkan 35 jawaban dipisahkan koma"
)

print(
    "Contoh:"
)

print(
    "A,D,C,B,A,D,C,B,A,D,C,B,A,D,C,B,A,D,C,B,A,D,C,B,A,D,C,B,A,D,C,B,A,D,C"
)

key_input = input(
    "\nKunci Jawaban : "
)

key_list = [
    x.strip().upper()
    for x in key_input.split(",")
]

# ==========================
# VALIDASI KUNCI
# ==========================
if len(key_list) != 35:

    print(
        "\nERROR!"
    )

    print(
        f"Jumlah kunci = {len(key_list)}"
    )

    print(
        "Harus tepat 35 soal."
    )

    exit()

# ==========================
# HITUNG NILAI
# ==========================
benar = 0

salah = 0

print(
    "\n=============================="
)

print(
    "DETAIL PENILAIAN"
)

print(
    "==============================\n"
)

for nomor in range(1, 36):

    jawaban_siswa = answers.get(
        nomor,
        "-"
    )

    kunci = key_list[
        nomor - 1
    ]

    if jawaban_siswa == kunci:

        benar += 1

        status = "BENAR"

    else:

        salah += 1

        status = "SALAH"

    print(
        f"Q{nomor:<2} | "
        f"Siswa: {jawaban_siswa} | "
        f"Kunci: {kunci} | "
        f"{status}"
    )

# ==========================
# HITUNG SKOR
# ==========================
nilai = (
    benar / 35
) * 100

# ==========================
# HASIL AKHIR
# ==========================
print(
    "\n=============================="
)

print(
    "HASIL PENILAIAN"
)

print(
    "=============================="
)

print(
    f"Benar : {benar}"
)

print(
    f"Salah : {salah}"
)

print(
    f"Nilai : {nilai:.2f}"
)

print(
    "=============================="
)