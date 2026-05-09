#Ganti background
import cv2
from histogram_operations import HistogramProcessor as hp

# 1. Load gambar objek (foreground) dan background baru
img_objek = cv2.imread('foto_produk.jpg', 0) # Grayscale
img_bg_baru = cv2.imread('pemandangan.jpg', 0)

# 2. Pastikan ukuran background sama dengan foreground
h, w = img_objek.shape[:2]
img_bg_baru = hp.resize_manual(img_bg_baru, h, w)

# 3. Ganti background (Contoh: hapus background putih/terang)
hasil = hp.replace_background(
    foreground=img_objek,
    background=img_bg_baru,
    threshold=240,   # Piksel di atas 240 dianggap putih
    mode='bright'    # Mode terang
)

# 4. Tampilkan hasil
hp.show_comparison(img_objek, hasil, "Sebelum", "Sesudah Ganti Background")


import cv2
import numpy as np
from histogram_operations import HistogramProcessor as hp

# 1. Load gambar asli
img = cv2.imread('pemandangan.jpg', 0)
h, w = img.shape

# 2. Buat Mask (Contoh: Mask lingkaran di tengah)
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), 100, 255, -1) 

# 3. Terapkan Masking
# Bagian di luar lingkaran akan menjadi hitam
img_masked = hp.apply_mask(img, mask)

# 4. Tampilkan
hp.show_comparison(img, img_masked, "Original", "Hasil Masking Lingkaran")

