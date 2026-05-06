import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  IMAGE OPERATIONS CHEATSHEET — Manual NumPy (No CV2 Built-ins)
#  Semua operasi dibuat secara manual menggunakan indexing dan loop.
#  Boleh menggunakan fungsi MATEMATIKA numpy: cos, sin, sqrt, pi, zeros, dll.
# =============================================================================
#
#  DAFTAR FUNGSI:
#  ─────────────────────────────────────────────────────────────────────────────
#  [1] FLIP & CERMIN
#      flip_horizontal(img)
#      flip_vertical(img)
#      flip(img, mode)              ← overloading: 'h','v','b'
#
#  [2] ROTASI
#      rotate_90_cw(img)            ← 90° searah jarum jam
#      rotate_90_ccw(img)           ← 90° berlawanan jarum jam
#      rotate_180(img)
#      rotate(img, angle_deg)       ← overloading: sudut sembarang
#
#  [3] TRANSLASI & CROP
#      translasi(img, tx, ty)       ← geser gambar (tx kolom, ty baris)
#      crop_rect(img, x, y, w, h)  ← crop area persegi dari titik (x,y)
#      crop_setengah_lingkaran(img, center, radius, sisi)
#                                   ← crop setengah lingkaran
#                                      sisi: 'kanan','kiri','atas','bawah'
#
#  [4] MERGE & SPLIT
#      merge_horizontal(c1, c2)
#      merge_vertical(c1, c2)
#      split_circular(img, center, radius)
#      merge_circular(dalam, luar, mask)
#
#  [5] OPERASI ARITMATIKA
#      tambah_scalar(img, n)        negatif(img)
#      kurang_scalar(img, n)        rata_rata_citra(a, b)
#      kali_scalar(img, n)          tambah_citra(a, b)
#      bagi_scalar(img, n)          kurang_citra(a, b)
#                                   kali_citra(a, b)
#
#  [6] OPERASI BOOLEAN (BITWISE)
#      bitwise_and(a, b)            bitwise_or(a, b)
#      bitwise_xor(a, b)            bitwise_not(img)
#      threshold(img, nilai, maks)
#
#  [7] HISTOGRAM
#      get_histogram(img)
#      plot_histogram(img, title)
#
#  [8] MORFOLOGI
#      dilate(img, kernel_size)
#      erode(img, kernel_size)
#
#  [9] VISUALISASI
#      show(img, title)
#      show_compare(img1, img2, title1, title2)
#
#  [10] KONVOLUSI (paling bawah)
#      convolve(img, kernel)        ← custom kernel sembarang ukuran
#      blur_avg(img, ukuran)
#      blur_gaussian(img)
#      sharpen(img)
#      edge_sobel(img)
#      edge_laplacian(img)
#
#  [11] EQUALISASI HISTOGRAM (paling bawah)
#      equalize_histogram(img)
#  ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
#  1. FLIP (PENCERMINAN)
# =============================================================================

def flip_horizontal(img):
    """
    Flip / cerminkan gambar secara horizontal (kiri-kanan).
    Cara kerja: Balikan urutan kolom setiap baris.

    Args:
        img: np.array gambar (grayscale atau RGB)

    Returns:
        np.array gambar yang di-flip horizontal

    Contoh:
        >>> hasil = flip_horizontal(img)
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                hasil[i, w - 1 - j] = img[i, j]
    else:
        hasil = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                hasil[i, w - 1 - j] = img[i, j]
    return hasil


def flip_vertical(img):
    """
    Flip / cerminkan gambar secara vertikal (atas-bawah).
    Cara kerja: Balikan urutan baris setiap kolom.

    Args:
        img: np.array gambar (grayscale atau RGB)

    Returns:
        np.array gambar yang di-flip vertikal

    Contoh:
        >>> hasil = flip_vertical(img)
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
        for i in range(h):
            hasil[h - 1 - i] = img[i]
    else:
        hasil = np.zeros_like(img)
        for i in range(h):
            hasil[h - 1 - i] = img[i]
    return hasil


def flip(img, mode='horizontal'):
    """
    Simulated function overloading untuk flip.
    Pilih mode untuk menentukan arah flip.

    Args:
        img  : np.array gambar
        mode : 'horizontal' | 'h'  -> cermin kiri-kanan
               'vertical'   | 'v'  -> cermin atas-bawah
               'both'       | 'b'  -> keduanya (rotasi 180)

    Contoh:
        >>> flip(img, 'h')            # horizontal
        >>> flip(img, 'vertical')     # vertical
        >>> flip(img, 'b')            # both
    """
    if mode in ('horizontal', 'h', 1):
        return flip_horizontal(img)
    elif mode in ('vertical', 'v', 0):
        return flip_vertical(img)
    elif mode in ('both', 'b', -1):
        return flip_vertical(flip_horizontal(img))
    return img


# =============================================================================
#  2. ROTATE (ROTASI)
# =============================================================================

def rotate_90_cw(img):
    """
    Rotasi 90 derajat searah jarum jam (Clockwise).
    Cara kerja: baris ke-i, kolom ke-j -> kolom ke-(n-1-i), baris ke-j.

    Args:
        img: np.array gambar

    Returns:
        np.array gambar hasil rotasi

    Contoh:
        >>> hasil = rotate_90_cw(img)
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        hasil = np.zeros((w, h, img.shape[2]), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                hasil[j, h - 1 - i] = img[i, j]
    else:
        hasil = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                hasil[j, h - 1 - i] = img[i, j]
    return hasil


def rotate_90_ccw(img):
    """
    Rotasi 90 derajat berlawanan arah jarum jam (Counter-Clockwise).
    Cara kerja: baris ke-i, kolom ke-j -> kolom ke-i, baris ke-(w-1-j).

    Args:
        img: np.array gambar

    Returns:
        np.array gambar hasil rotasi

    Contoh:
        >>> hasil = rotate_90_ccw(img)
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        hasil = np.zeros((w, h, img.shape[2]), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                hasil[w - 1 - j, i] = img[i, j]
    else:
        hasil = np.zeros((w, h), dtype=img.dtype)
        for i in range(h):
            for j in range(w):
                hasil[w - 1 - j, i] = img[i, j]
    return hasil


def rotate_180(img):
    """
    Rotasi 180 derajat.

    Args:
        img: np.array gambar

    Contoh:
        >>> hasil = rotate_180(img)
    """
    return flip_vertical(flip_horizontal(img))


def rotate(img, angle_deg):
    """
    Simulated function overloading untuk rotasi.
    Rotasi sembarang sudut menggunakan inverse mapping (bilinear-ish).

    Args:
        img      : np.array gambar
        angle_deg: Sudut rotasi dalam derajat (+ = CCW, - = CW)
                   Shortcut khusus: 90, -90, 180 menggunakan fungsi exact.

    Contoh:
        >>> rotate(img, 90)    # putar 90 CCW
        >>> rotate(img, -90)   # putar 90 CW
        >>> rotate(img, 180)   # putar 180
        >>> rotate(img, 45)    # putar 45 derajat sembarang
    """
    # Gunakan fungsi exact untuk sudut standar (lebih cepat)
    if angle_deg == 90:
        return rotate_90_ccw(img)
    elif angle_deg == -90 or angle_deg == 270:
        return rotate_90_cw(img)
    elif angle_deg == 180 or angle_deg == -180:
        return rotate_180(img)

    # Rotasi sembarang menggunakan inverse mapping
    h, w = img.shape[:2]
    rad = np.pi * (angle_deg / 180)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    new_w = int(w * abs(cos_a) + h * abs(sin_a))
    new_h = int(h * abs(cos_a) + w * abs(sin_a))

    cx_in, cy_in = w / 2, h / 2
    cx_out, cy_out = new_w / 2, new_h / 2

    if len(img.shape) == 3:
        out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((new_h, new_w), dtype=img.dtype)

    for row in range(new_h):
        for col in range(new_w):
            dx = col - cx_out
            dy = row - cy_out
            src_x = int(cos_a * dx + sin_a * dy + cx_in)
            src_y = int(-sin_a * dx + cos_a * dy + cy_in)
            if 0 <= src_x < w and 0 <= src_y < h:
                out[row, col] = img[src_y, src_x]
    return out



# =============================================================================
#  3. TRANSLASI & CROP
# =============================================================================

def translasi(img, tx, ty):
    """
    Geser (translasi) gambar sebesar tx kolom dan ty baris.
    Piksel yang keluar batas akan dipotong, area kosong diisi 0 (hitam).

    Args:
        img: np.array gambar (grayscale atau RGB)
        tx : Pergeseran horizontal (+ = kanan, - = kiri)
        ty : Pergeseran vertikal   (+ = bawah, - = atas)

    Returns:
        np.array gambar hasil translasi

    Contoh:
        >>> hasil = translasi(img, 50, 30)    # geser kanan 50, bawah 30
        >>> hasil = translasi(img, -20, 0)    # geser kiri 20
        >>> hasil = translasi(img, 0, -40)    # geser atas 40
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
    else:
        hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            ni = i + ty   # baris baru
            nj = j + tx   # kolom baru
            if 0 <= ni < h and 0 <= nj < w:
                hasil[ni, nj] = img[i, j]
    return hasil


def crop_rect(img, x, y, lebar, tinggi):
    """
    Crop area persegi panjang dari gambar.

    Args:
        img   : np.array gambar
        x     : Kolom awal (horizontal, dari kiri)
        y     : Baris awal (vertikal, dari atas)
        lebar : Lebar area crop (jumlah kolom)
        tinggi: Tinggi area crop (jumlah baris)

    Returns:
        np.array potongan gambar

    Contoh:
        >>> potongan = crop_rect(img, 50, 30, 200, 150)
        # Ambil area dari (x=50, y=30) selebar 200px dan setinggi 150px
    """
    h, w = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + lebar)
    y2 = min(h, y + tinggi)
    return img[y1:y2, x1:x2]


def crop_setengah_lingkaran(img, center=None, radius=None, sisi='kanan'):
    """
    Crop setengah lingkaran dari gambar.
    Piksel yang tidak masuk area setengah lingkaran akan diisi hitam (0).

    Cara kerja:
        Piksel (i, j) dipertahankan jika KEDUA syarat terpenuhi:
          1. Jarak Euclidean dari center <= radius
          2. Berada di sisi yang ditentukan (kanan/kiri/atas/bawah)

    Args:
        img   : np.array gambar
        center: (cx, cy) pusat lingkaran. Default = tengah gambar.
                Contoh titik tengah kanan: center=(w-1, h//2)
        radius: Jari-jari. Default = setengah sisi terpendek gambar.
        sisi  : 'kanan'  -> ambil x >= cx  (setengah kanan)
                'kiri'   -> ambil x <= cx  (setengah kiri)
                'atas'   -> ambil y <= cy  (setengah atas)
                'bawah'  -> ambil y >= cy  (setengah bawah)

    Returns:
        np.array gambar dengan area setengah lingkaran, sisanya hitam

    Contoh:
        # Setengah lingkaran kanan dari tengah gambar
        >>> hasil = crop_setengah_lingkaran(img)

        # Setengah lingkaran dari titik tengah sisi kanan gambar
        >>> h, w = img.shape[:2]
        >>> hasil = crop_setengah_lingkaran(img, center=(w-1, h//2), radius=h//2, sisi='kiri')

        # Setengah lingkaran atas
        >>> hasil = crop_setengah_lingkaran(img, sisi='atas')
    """
    h, w = img.shape[:2]

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    if radius is None:
        radius = min(h, w) // 2

    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
    else:
        hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((j - cx)**2 + (i - cy)**2)
            dalam_lingkaran = dist <= radius

            if sisi == 'kanan':
                dalam_sisi = j >= cx
            elif sisi == 'kiri':
                dalam_sisi = j <= cx
            elif sisi == 'atas':
                dalam_sisi = i <= cy
            elif sisi == 'bawah':
                dalam_sisi = i >= cy
            else:
                dalam_sisi = True  # full circle kalau sisi tidak dikenali

            if dalam_lingkaran and dalam_sisi:
                hasil[i, j] = img[i, j]

    return hasil


# =============================================================================
#  4. MERGE & SPLIT
# =============================================================================

def merge_horizontal(citra_1, citra_2):
    """
    Gabungkan dua citra secara horizontal (bersebelahan kiri-kanan).
    Tinggi output = max(tinggi_1, tinggi_2).

    Args:
        citra_1: np.array gambar kiri
        citra_2: np.array gambar kanan

    Returns:
        np.array gambar gabungan horizontal

    Contoh:
        >>> combined = merge_horizontal(img1, img2)
    """
    c1, c2 = np.array(citra_1), np.array(citra_2)
    tinggi = max(c1.shape[0], c2.shape[0])
    lebar = c1.shape[1] + c2.shape[1]

    if len(c1.shape) == 3:
        hasil = np.zeros((tinggi, lebar, c1.shape[2])).astype(int)
    else:
        hasil = np.zeros((tinggi, lebar)).astype(int)

    hasil[0:c1.shape[0], 0:c1.shape[1]] = c1
    hasil[0:c2.shape[0], c1.shape[1]:c1.shape[1] + c2.shape[1]] = c2
    return hasil


def merge_vertical(citra_1, citra_2):
    """
    Gabungkan dua citra secara vertikal (bertumpuk atas-bawah).
    Lebar output = max(lebar_1, lebar_2).

    Args:
        citra_1: np.array gambar atas
        citra_2: np.array gambar bawah

    Returns:
        np.array gambar gabungan vertikal

    Contoh:
        >>> combined = merge_vertical(img1, img2)
    """
    c1, c2 = np.array(citra_1), np.array(citra_2)
    tinggi = c1.shape[0] + c2.shape[0]
    lebar = max(c1.shape[1], c2.shape[1])

    if len(c1.shape) == 3:
        hasil = np.zeros((tinggi, lebar, c1.shape[2])).astype(int)
    else:
        hasil = np.zeros((tinggi, lebar)).astype(int)

    hasil[0:c1.shape[0], 0:c1.shape[1]] = c1
    hasil[c1.shape[0]:c1.shape[0] + c2.shape[0], 0:c2.shape[1]] = c2
    return hasil


def split_circular(img, center=None, radius=None):
    """
    Pisahkan citra menjadi dua bagian: dalam dan luar lingkaran.
    Cara kerja: Euclidean Distance dari setiap piksel ke pusat.

    Args:
        img   : np.array gambar
        center: (x, y) koordinat pusat lingkaran. Default = tengah gambar.
        radius: Jari-jari lingkaran. Default = 1/4 dari sisi terpendek.

    Returns:
        (dalam, luar, mask) -> tiga np.array

    Contoh:
        >>> dalam, luar, mask = split_circular(img)
        >>> dalam, luar, mask = split_circular(img, center=(100,100), radius=50)
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 4

    cx, cy = center
    if len(img.shape) == 3:
        dalam = np.zeros_like(img)
        luar = np.zeros_like(img)
    else:
        dalam = np.zeros_like(img)
        luar = np.zeros_like(img)

    mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((j - cx)**2 + (i - cy)**2)
            if dist <= radius:
                dalam[i, j] = img[i, j]
                mask[i, j] = 1
            else:
                luar[i, j] = img[i, j]
    return dalam, luar, mask


def merge_circular(dalam, luar, mask):
    """
    Gabungkan kembali bagian dalam lingkaran dan luar lingkaran.

    Args:
        dalam: np.array bagian dalam lingkaran
        luar : np.array bagian luar lingkaran
        mask : np.array mask (1=dalam, 0=luar)

    Returns:
        np.array gambar gabungan

    Contoh:
        >>> hasil = merge_circular(dalam, luar, mask)
    """
    h, w = dalam.shape[:2]
    if len(dalam.shape) == 3:
        hasil = np.zeros_like(dalam)
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 1:
                    hasil[i, j] = dalam[i, j]
                else:
                    hasil[i, j] = luar[i, j]
    else:
        hasil = np.zeros_like(dalam)
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 1:
                    hasil[i, j] = dalam[i, j]
                else:
                    hasil[i, j] = luar[i, j]
    return hasil



# =============================================================================
#  4. OPERASI ARITMATIKA (SCALAR & ANTAR CITRA)
# =============================================================================

def _clip(val):
    """Clamp nilai ke rentang [0, 255]."""
    if val < 0: return 0
    if val > 255: return 255
    return int(val)


def tambah_scalar(img, nilai):
    """
    Tambahkan nilai scalar ke setiap piksel.
    Rumus: output[i,j] = clip(img[i,j] + nilai)

    Contoh:
        >>> hasil = tambah_scalar(img, 50)
    """
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    if len(img.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img.shape[2]):
                    hasil[i, j, c] = _clip(int(img[i, j, c]) + nilai)
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _clip(int(img[i, j]) + nilai)
    return hasil


def kurang_scalar(img, nilai):
    """
    Kurangi setiap piksel dengan nilai scalar.
    Rumus: output[i,j] = clip(img[i,j] - nilai)

    Contoh:
        >>> hasil = kurang_scalar(img, 30)
    """
    return tambah_scalar(img, -nilai)


def kali_scalar(img, nilai):
    """
    Kalikan setiap piksel dengan nilai scalar (float).
    Rumus: output[i,j] = clip(img[i,j] * nilai)

    Contoh:
        >>> hasil = kali_scalar(img, 1.5)   # terangkan
        >>> hasil = kali_scalar(img, 0.5)   # gelapkan
    """
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    if len(img.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img.shape[2]):
                    hasil[i, j, c] = _clip(int(img[i, j, c]) * nilai)
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _clip(int(img[i, j]) * nilai)
    return hasil


def bagi_scalar(img, nilai):
    """
    Bagi setiap piksel dengan nilai scalar.
    Rumus: output[i,j] = clip(img[i,j] / nilai)

    Contoh:
        >>> hasil = bagi_scalar(img, 2)
    """
    if nilai == 0:
        raise ValueError("Pembagi tidak boleh 0")
    return kali_scalar(img, 1.0 / nilai)


def tambah_citra(img1, img2):
    """
    Tambahkan dua citra piksel per piksel.
    Rumus: output[i,j] = clip(img1[i,j] + img2[i,j])

    Contoh:
        >>> hasil = tambah_citra(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _clip(int(img1[i, j, c]) + int(img2[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _clip(int(img1[i, j]) + int(img2[i, j]))
    return hasil


def kurang_citra(img1, img2):
    """
    Kurangi dua citra piksel per piksel.
    Rumus: output[i,j] = clip(img1[i,j] - img2[i,j])

    Contoh:
        >>> hasil = kurang_citra(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _clip(int(img1[i, j, c]) - int(img2[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _clip(int(img1[i, j]) - int(img2[i, j]))
    return hasil


def kali_citra(img1, img2):
    """
    Kalikan dua citra piksel per piksel (hasilnya dibagi 255).
    Rumus: output[i,j] = clip((img1[i,j] * img2[i,j]) / 255)

    Contoh:
        >>> hasil = kali_citra(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _clip((int(img1[i, j, c]) * int(img2[i, j, c])) // 255)
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _clip((int(img1[i, j]) * int(img2[i, j])) // 255)
    return hasil


def rata_rata_citra(img1, img2):
    """
    Rata-rata dua citra (blending 50/50).
    Rumus: output[i,j] = (img1[i,j] + img2[i,j]) / 2

    Contoh:
        >>> hasil = rata_rata_citra(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = (int(img1[i, j, c]) + int(img2[i, j, c])) // 2
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = (int(img1[i, j]) + int(img2[i, j])) // 2
    return hasil


def negatif(img):
    """
    Citra negatif (inversi intensitas).
    Rumus: output[i,j] = 255 - img[i,j]

    Contoh:
        >>> hasil = negatif(img)
    """
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    if len(img.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img.shape[2]):
                    hasil[i, j, c] = 255 - int(img[i, j, c])
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = 255 - int(img[i, j])
    return hasil


# =============================================================================
#  5. OPERASI BOOLEAN (BITWISE)
# =============================================================================

def _and_bit(a, b):
    hasil = 0
    for bit in range(8):
        if (a >> bit & 1) and (b >> bit & 1):
            hasil |= (1 << bit)
    return hasil

def _or_bit(a, b):
    hasil = 0
    for bit in range(8):
        if (a >> bit & 1) or (b >> bit & 1):
            hasil |= (1 << bit)
    return hasil

def _xor_bit(a, b):
    hasil = 0
    for bit in range(8):
        if ((a >> bit) & 1) != ((b >> bit) & 1):
            hasil |= (1 << bit)
    return hasil

def _not_bit(a):
    hasil = 0
    for bit in range(8):
        if not ((a >> bit) & 1):
            hasil |= (1 << bit)
    return hasil


def bitwise_and(img1, img2):
    """
    AND bitwise per piksel. Kegunaan: masking area tertentu.

    Contoh:
        >>> hasil = bitwise_and(img, mask)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _and_bit(int(img1[i, j, c]), int(img2[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _and_bit(int(img1[i, j]), int(img2[i, j]))
    return hasil


def bitwise_or(img1, img2):
    """
    OR bitwise per piksel. Kegunaan: gabungkan dua region.

    Contoh:
        >>> hasil = bitwise_or(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _or_bit(int(img1[i, j, c]), int(img2[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _or_bit(int(img1[i, j]), int(img2[i, j]))
    return hasil


def bitwise_xor(img1, img2):
    """
    XOR bitwise per piksel. Kegunaan: deteksi perbedaan dua citra.

    Contoh:
        >>> hasil = bitwise_xor(img1, img2)
    """
    h, w = img1.shape[:2]
    hasil = np.zeros_like(img1)
    if len(img1.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img1.shape[2]):
                    hasil[i, j, c] = _xor_bit(int(img1[i, j, c]), int(img2[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _xor_bit(int(img1[i, j]), int(img2[i, j]))
    return hasil


def bitwise_not(img):
    """
    NOT bitwise per piksel (inversi bit-per-bit, sama dengan negatif untuk 8-bit).

    Contoh:
        >>> hasil = bitwise_not(img)
    """
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    if len(img.shape) == 3:
        for i in range(h):
            for j in range(w):
                for c in range(img.shape[2]):
                    hasil[i, j, c] = _not_bit(int(img[i, j, c]))
    else:
        for i in range(h):
            for j in range(w):
                hasil[i, j] = _not_bit(int(img[i, j]))
    return hasil


def threshold(img, nilai=127, maks=255):
    """
    Thresholding biner: piksel >= nilai -> maks, lainnya -> 0.
    Menghasilkan citra biner (hitam-putih).

    Args:
        img  : np.array gambar grayscale
        nilai: nilai ambang batas (default 127)
        maks : nilai output untuk piksel di atas threshold (default 255)

    Contoh:
        >>> biner = threshold(img_gray, 127)
        >>> biner = threshold(img_gray, 100, 255)
    """
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            hasil[i, j] = maks if int(img[i, j]) >= nilai else 0
    return hasil


# =============================================================================
#  6. HISTOGRAM
# =============================================================================

def get_histogram(img):
    """
    Hitung histogram intensitas piksel (0-255) secara manual.

    Args:
        img: np.array gambar (grayscale atau RGB)

    Returns:
        Grayscale -> np.array shape (256,)
        RGB       -> list of 3 np.array shape (256,)

    Contoh:
        >>> hist = get_histogram(img_gray)
        >>> hists = get_histogram(img_rgb)  # [hist_B, hist_G, hist_R]
    """
    if len(img.shape) == 3:
        hists = []
        for c in range(3):
            hist = np.zeros(256)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    hist[int(img[i, j, c])] += 1
            hists.append(hist)
        return hists
    else:
        hist = np.zeros(256)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                hist[int(img[i, j])] += 1
        return hist


# equalize_histogram() -> lihat bagian bawah file (KONVOLUSI & EQUALISASI)


# =============================================================================
#  5. KONVOLUSI
# =============================================================================

# convolve(), blur_avg(), blur_gaussian(), sharpen(), edge_sobel(), edge_laplacian()
# -> lihat bagian bawah file (KONVOLUSI & EQUALISASI)


# =============================================================================
#  6. DILASI & EROSI (Morphology)
# =============================================================================

def dilate(img, kernel_size=3):
    """
    Dilasi morfologi: ambil nilai maksimum di sekitar jendela kernel.

    Args:
        img        : np.array gambar biner atau grayscale
        kernel_size: ukuran kernel persegi (default 3x3)

    Contoh:
        >>> hasil = dilate(img_biner, kernel_size=3)
    """
    h, w = img.shape[:2]
    pad = kernel_size // 2
    hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            max_val = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    ni, nj = i - pad + ki, j - pad + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        if img[ni, nj] > max_val:
                            max_val = img[ni, nj]
            hasil[i, j] = max_val
    return hasil


def erode(img, kernel_size=3):
    """
    Erosi morfologi: ambil nilai minimum di sekitar jendela kernel.

    Args:
        img        : np.array gambar biner atau grayscale
        kernel_size: ukuran kernel persegi (default 3x3)

    Contoh:
        >>> hasil = erode(img_biner, kernel_size=3)
    """
    h, w = img.shape[:2]
    pad = kernel_size // 2
    hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            min_val = 255
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    ni, nj = i - pad + ki, j - pad + kj
                    if 0 <= ni < h and 0 <= nj < w:
                        if img[ni, nj] < min_val:
                            min_val = img[ni, nj]
            hasil[i, j] = min_val
    return hasil


# =============================================================================
#  7. HELPER VISUALISASI
# =============================================================================

def show(img, title="Image"):
    """
    Tampilkan satu gambar.

    Contoh:
        >>> show(img, "Hasil Flip")
    """
    plt.figure(figsize=(5, 5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:, :, ::-1])  # BGR -> RGB (hanya slicing)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_compare(img1, img2, title1="Before", title2="After"):
    """
    Tampilkan dua gambar berdampingan untuk perbandingan.

    Contoh:
        >>> show_compare(img, hasil, "Original", "Hasil Flip")
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1 if len(img1.shape) == 2 else img1[:, :, ::-1],
               cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2 if len(img2.shape) == 2 else img2[:, :, ::-1],
               cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()


def plot_histogram(img, title="Histogram"):
    """
    Plot histogram intensitas gambar.

    Contoh:
        >>> plot_histogram(img_gray, "Histogram Asli")
    """
    plt.figure(figsize=(8, 4))
    if len(img.shape) == 2:
        hist = get_histogram(img)
        plt.plot(hist)
        plt.title(f"{title} (Grayscale)")
    else:
        hists = get_histogram(img)
        for i, col in enumerate(['blue', 'green', 'red']):
            plt.plot(hists[i], color=col)
        plt.title(f"{title} (RGB)")
    plt.xlim([0, 256])
    plt.show()


# =============================================================================
#  10. KONVOLUSI — Custom Kernel & Filter Preset
#  Taruh paling bawah karena paling berat komputasinya.
# =============================================================================

def convolve(img, kernel):
    """
    Konvolusi 2D manual (sliding window). Mendukung kernel custom sembarang ukuran.
    Cara kerja: Geser kernel di atas setiap piksel, kalikan elemen, jumlahkan.

    Args:
        img   : np.array gambar grayscale (2D) atau RGB (3D)
        kernel: np.array kernel NxM, misal dari soal:
                  [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]  -> lalu dibagi jumlah elemen untuk blur

    Returns:
        np.array hasil konvolusi, dtype uint8, nilai di-clip ke [0, 255]

    Contoh custom kernel dari soal:
        # Kernel blur rata-rata 3x3
        k = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0
        hasil = convolve(img, k)

        # Kernel Sobel Gx
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        hasil = convolve(img, kx)

        # Kernel 5x5 Gaussian
        k5 = np.array([[1,4,6,4,1],
                        [4,16,24,16,4],
                        [6,24,36,24,6],
                        [4,16,24,16,4],
                        [1,4,6,4,1]]) / 256.0
        hasil = convolve(img, k5)
    """
    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
        for c in range(img.shape[2]):
            hasil[:, :, c] = convolve(img[:, :, c], kernel)
        return hasil

    h, w = img.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # Zero-padding manual
    h_pad, w_pad = h + 2 * ph, w + 2 * pw
    padded = np.zeros((h_pad, w_pad), dtype=float)
    for i in range(h):
        for j in range(w):
            padded[i + ph, j + pw] = img[i, j]

    output = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            total = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    total += padded[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = total

    # Clip ke [0, 255] manual
    for i in range(h):
        for j in range(w):
            if output[i, j] < 0:
                output[i, j] = 0
            elif output[i, j] > 255:
                output[i, j] = 255

    return output.astype(np.uint8)


# --- Filter Preset (menggunakan convolve) ---

def blur_avg(img, ukuran=3):
    """
    Blur rata-rata (Average Blur). Kernel semua 1 dibagi ukuran^2.

    Args:
        img   : np.array gambar
        ukuran: ukuran kernel persegi (default 3 -> kernel 3x3)

    Contoh:
        >>> hasil = blur_avg(img)
        >>> hasil = blur_avg(img, 5)   # kernel 5x5
    """
    k = np.ones((ukuran, ukuran)) / float(ukuran * ukuran)
    return convolve(img, k)


def blur_gaussian(img):
    """
    Gaussian Blur 3x3 (aproksimasi kernel Gaussian).

    Contoh:
        >>> hasil = blur_gaussian(img)
    """
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float) / 16.0
    return convolve(img, k)


def sharpen(img):
    """
    Sharpening / penajaman tepi dengan kernel laplacian of gaussian.

    Contoh:
        >>> hasil = sharpen(img)
    """
    k = np.array([[ 0, -1,  0],
                  [-1,  5, -1],
                  [ 0, -1,  0]], dtype=float)
    return convolve(img, k)


def edge_sobel(img):
    """
    Deteksi tepi Sobel. Output = sqrt(Gx^2 + Gy^2).

    Args:
        img: np.array gambar grayscale

    Contoh:
        >>> tepi = edge_sobel(img_gray)
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)
    gx = convolve(img, kx).astype(float)
    gy = convolve(img, ky).astype(float)
    hasil = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            v = int(np.sqrt(gx[i, j]**2 + gy[i, j]**2))
            hasil[i, j] = v if v < 255 else 255
    return hasil


def edge_laplacian(img):
    """
    Deteksi tepi Laplacian.

    Contoh:
        >>> tepi = edge_laplacian(img_gray)
    """
    k = np.array([[ 0,  1,  0],
                  [ 1, -4,  1],
                  [ 0,  1,  0]], dtype=float)
    return convolve(img, k)


# =============================================================================
#  11. EQUALISASI HISTOGRAM (taruh bawah karena butuh get_histogram)
# =============================================================================

def equalize_histogram(img):
    """
    Equalisasi histogram untuk meningkatkan kontras (CDF-based).
    Cara kerja:
        1. Hitung histogram H[v] untuk setiap intensitas v
        2. Hitung CDF: CDF[v] = sum(H[0..v])
        3. Normalisasi: new[v] = (CDF[v] - CDF_min) / (total - CDF_min) * 255
        4. Buat LUT (Look-Up Table), mapping lama -> baru
        5. Apply LUT ke setiap piksel

    Args:
        img: np.array gambar grayscale atau RGB

    Contoh:
        >>> eq = equalize_histogram(img_gray)
        >>> eq = equalize_histogram(img_bgr)  # per channel
    """
    if len(img.shape) == 3:
        hasil = np.zeros_like(img)
        for c in range(3):
            hasil[:, :, c] = equalize_histogram(img[:, :, c])
        return hasil

    h, w = img.shape
    total = h * w

    # Step 1: Histogram
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[int(img[i, j])] += 1

    # Step 2: CDF
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for v in range(1, 256):
        cdf[v] = cdf[v - 1] + hist[v]

    # Step 3: Cari CDF minimum (bukan 0)
    cdf_min = 0
    for v in range(256):
        if cdf[v] > 0:
            cdf_min = cdf[v]
            break

    # Step 4: LUT
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(256):
        val = (cdf[v] - cdf_min) / (total - cdf_min) * 255
        lut[v] = int(min(max(val, 0), 255))

    # Step 5: Apply
    hasil = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            hasil[i, j] = lut[int(img[i, j])]
    return hasil
