# Image Generation dengan GAN — Overhead-MNIST (kelas `ship`)

Proyek ini membangun dan membandingkan dua arsitektur GAN untuk men-generate citra satelit kapal (`ship`) berukuran 28×28 grayscale dari dataset [Overhead-MNIST](https://www.kaggle.com/datasets/datamunge/overheadmnist/data). Evaluasi memakai **FID (Fréchet Inception Distance)**.

## Dataset

- Sumber: Overhead-MNIST (versi 2), hanya kelas **`ship`**.
- Data `train` + `test` digabung: **8.012 citra** (7.124 train + 888 test), 28×28 grayscale.
- **Tanpa split train/val/test** — GAN dievaluasi lewat kemiripan distribusi citra fake terhadap real (FID), bukan akurasi pada data yang belum dilihat.
- Preprocessing: piksel di-scale ke `[-1, 1]` dengan `(x - 127.5) / 127.5`, menyesuaikan activation `tanh` pada output generator.

## Model

### (b) Baseline — Conditional GAN (Dense/Linear)

Mengikuti diagram pada soal: noise + `Embedding(label)` dikonkatenasi, lalu melewati layer `Dense`.

- **Generator:** 128 → 256 → 512 → 1024 → 784 (`LeakyReLU(0.2)`, output `tanh`)
- **Discriminator:** 512 → 1024 → 1024 → 512 → 1 (logit)
- Hyperparameter: `lr=2e-4`, `batch_size=128`, 60 epoch

### (c) Improved — DCGAN (Convolutional)

| Aspek | Baseline | Improved |
|---|---|---|
| Layer | `Dense` | `Conv2D` / `Conv2DTranspose` |
| Upsampling | Dense bertahap lalu reshape | `Dense` → reshape 7×7 → `Conv2DTranspose` (7→14→28) |
| Normalisasi | — | `BatchNormalization` di generator |
| Label conditioning | Ada | Dihapus (hanya 1 kelas) |
| Regularisasi discriminator | — | `Dropout(0.3)` |
| Hyperparameter | Fixed | Grid search `learning_rate` × `batch_size` |

### Training

- Loss: `BinaryCrossentropy(from_logits=True)` dengan one-sided label smoothing (label real = 0.9)
- Optimizer: Adam (`beta_1=0.5`)
- FID dipantau tiap 10 epoch, dan bobot generator dengan FID terbaik disimpan sebagai checkpoint akhir.

## Evaluasi: FID

FID dihitung dari fitur 2048-dim **InceptionV3** (pretrained ImageNet). Citra di-resize ke 299×299 dan channel grayscale direplikasi menjadi 3 channel.

### Hasil tuning (10 epoch, 200 sampel)

| Learning rate | Batch size | FID |
|---|---|---|
| **1e-4** | **64** | **261.49** |
| 2e-4 | 64 | 264.10 |
| 2e-4 | 128 | 321.06 |
| 1e-4 | 128 | 322.39 |

### Perbandingan akhir (500 sampel)

| Model | LR | Batch | Epoch | FID ↓ |
|---|---|---|---|---|
| Baseline (CGAN, Dense) | 2e-4 | 128 | 60 | 356.02 |
| **Improved (DCGAN, Conv)** | 1e-4 | 64 | 80 | **152.37** |

## Kesimpulan

- DCGAN menurunkan FID dari **356.02 menjadi 152.37**. Penyebab utamanya adalah penggantian `Dense` dengan `Conv2D`, yang memanfaatkan struktur spasial citra.
- Loss generator dan discriminator pada DCGAN stabil (sekitar 0.8 dan 1.37) tanpa tanda mode collapse.
- Model belum sepenuhnya layak pakai: FID masih tinggi dan secara visual masih ada noise.

## Keterbatasan

- InceptionV3 dilatih pada foto natural, sedangkan citra di sini berupa citra satelit grayscale kecil (*out-of-domain*), sehingga nilai FID absolut cenderung tinggi dan sebaiknya dipakai untuk perbandingan relatif antar model.
- Grid search kecil dan hanya 10 epoch per kandidat, sehingga belum tentu menemukan hyperparameter optimal.
