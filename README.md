# AnalogToBi

AnalogToBi adalah pipeline untuk memproses rangkaian analog dari netlist SPICE menjadi representasi bipartite graph, melatih model GPT dan GAT, lalu melakukan inference dengan grammar-guided decoding.

## Gambaran Alur

Urutan kerja yang paling umum adalah:

1. Netlist `.cir` di `Dataset/` diproses menjadi graph bipartit.
2. Graph diubah menjadi urutan token `node -> edge -> node -> ...`.
3. Token diberi label tipe rangkaian, lalu dibagi menjadi data latih dan validasi.
4. Data `Training_renamed.npy` dan `Validation_renamed.npy` dipakai untuk pretraining GPT.
5. Model hasil training dipakai untuk inference terstruktur dengan grammar.
6. Hasil generasi bisa dievaluasi dengan metrik validity, novelty, dan exact matching.

## Representasi Data

Format urutan token yang dipakai proyek ini adalah:

```text
CIRCUIT_Opamp -> VSS -> M_SB -> NM1 -> M_D -> VOUT1 -> ... -> TRUNCATE
```

Komponen utamanya:

- `CIRCUIT_*` = token tipe rangkaian
- `NM*`, `PM*`, `NPN*`, `PNP*`, `R*`, `C*`, `L*`, `DIO*` = node device
- `VIN*`, `VOUT*`, `VB*`, `VDD`, `VSS`, `NET*` = node net/port
- `M_*`, `B_*`, `R_C`, `C_C`, `L_C`, `D_*` = token koneksi/pin
- `TRUNCATE` = penanda akhir sequence

## Setup Environment

```bash
conda env create -f environment.yml
conda activate AnalogToBi
```

## Struktur Data

Folder `Dataset/` berisi sample mentah. Umumnya satu folder bernomor berisi:

| File | Fungsi |
|---|---|
| `{ID}.cir` | Netlist SPICE asli |
| `Graph_Bipart{ID}.csv` | Hasil graph bipartit |
| `Book{ID}.png` | Cuplikan sumber/paper |
| `Cadence{ID}.png` | Cuplikan schematic Cadence |
| `Pagenumber{ID}.txt` | Referensi halaman |
| `Port{ID}.txt` | Informasi port netlist |

## Alur Kerja

### 1. Bangun graph bipartit

```bash
python PREPROCESSING_Bipartite.py
```

Script ini membaca netlist SPICE dari `Dataset/` dan membentuk representasi graph bipartit.

### 2. Ubah graph menjadi sequence token

```bash
python PREPROCESSING_Augmentation_Bipart.py
```

Script ini melakukan traversing graph dan menghasilkan beberapa variasi sequence yang valid untuk satu rangkaian.

### 3. Tambahkan token tipe rangkaian

```bash
python PREPROCESSING_Add_Circuit_Types.py
```

Token seperti `CIRCUIT_Opamp` dipasang di awal sequence agar model belajar secara conditional berdasarkan jenis rangkaian.

### 4. Bentuk dataset GPT

```bash
python PREPROCESSING_GPT_dataset.py
```

Membuat split train/validation untuk pretraining GPT.

### 5. Augmentasi renaming device

```bash
python PREPROCESSING_Renaming.py --input Training.npy --output Training_renamed.npy
python PREPROCESSING_Renaming.py --input Validation.npy --output Validation_renamed.npy
```

Langkah ini mengacak penomoran device, misalnya `NM1` menjadi `NM5`, tanpa mengubah topologi.

### 6. Bentuk dataset GAT

```bash
python PREPROCESSING_GAT_dataset.py
```

Membuat dataset untuk classifier GAT dengan split berbasis ID rangkaian.

### 7. Renaming untuk dataset GAT

```bash
python PREPROCESSING_Renaming.py --input Training_GAT.npy --output Training_GAT_renamed.npy
python PREPROCESSING_Renaming.py --input Validation_GAT.npy --output Validation_GAT_renamed.npy
```

### 8. Pretrain GPT

```bash
python GPT_Pretrain.py
```

Model ini belajar memprediksi token rangkaian secara autoregressive.

### 9. Train GAT classifier

```bash
python GAT_Train.py
```

Model ini memprediksi kategori rangkaian dari graph bipartit.

### 10. Inference dengan grammar

```bash
python GPT_Inference_Grammar.py CIRCUIT_Opamp
```

Contoh di atas menghasilkan topologi untuk kategori OpAmp.

### 11. Evaluasi hasil

```bash
python GAT_Inference_ALL.py
python METRIC_Validity.py
python METRIC_Novelty.py
python METRIC_Valid_n_Novel.py
python METRIC_ExactMatching.py
```

## Perintah Terminal untuk User Baru

Jika ingin mulai dari nol, urutan praktisnya:

```bash
conda env create -f environment.yml
conda activate AnalogToBi
cd AnalogToBi
python PREPROCESSING_Bipartite.py
python PREPROCESSING_Augmentation_Bipart.py
python PREPROCESSING_Add_Circuit_Types.py
python PREPROCESSING_GPT_dataset.py
python PREPROCESSING_Renaming.py --input Training.npy --output Training_renamed.npy
python PREPROCESSING_Renaming.py --input Validation.npy --output Validation_renamed.npy
python GPT_Pretrain.py
python GPT_Inference_Grammar.py CIRCUIT_Opamp
```

Jika ingin melatih classifier GAT juga:

```bash
python PREPROCESSING_GAT_dataset.py
python PREPROCESSING_Renaming.py --input Training_GAT.npy --output Training_GAT_renamed.npy
python PREPROCESSING_Renaming.py --input Validation_GAT.npy --output Validation_GAT_renamed.npy
python GAT_Train.py
```

## File Penting

- [`PREPROCESSING_Bipartite.py`](./PREPROCESSING_Bipartite.py)
- [`PREPROCESSING_Augmentation_Bipart.py`](./PREPROCESSING_Augmentation_Bipart.py)
- [`PREPROCESSING_Add_Circuit_Types.py`](./PREPROCESSING_Add_Circuit_Types.py)
- [`PREPROCESSING_GPT_dataset.py`](./PREPROCESSING_GPT_dataset.py)
- [`PREPROCESSING_Renaming.py`](./PREPROCESSING_Renaming.py)
- [`GPT_Pretrain.py`](./GPT_Pretrain.py)
- [`GPT_Inference_Grammar.py`](./GPT_Inference_Grammar.py)
- [`GAT_Train.py`](./GAT_Train.py)
- [`METRIC_Validity.py`](./METRIC_Validity.py)
- [`METRIC_Novelty.py`](./METRIC_Novelty.py)
- [`METRIC_Valid_n_Novel.py`](./METRIC_Valid_n_Novel.py)
- [`METRIC_ExactMatching.py`](./METRIC_ExactMatching.py)

## Catatan

- Jalankan perintah dari folder `AnalogToBi/` agar path relatif cocok.
- Beberapa skrip mengharapkan file hasil preprocessing seperti `Training.npy`, `Validation.npy`, `Training_renamed.npy`, dan `Validation_renamed.npy` sudah tersedia.
- Untuk inference dan evaluasi, pastikan model checkpoint yang dibutuhkan juga sudah dilatih terlebih dulu.
