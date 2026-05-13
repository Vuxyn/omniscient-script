import numpy as np
import cv2

# =============================================================================
#  IMAGE ENHANCEMENT CHEATSHEET (MODUL 3) — Manual vs Built-in
#  Implementasi operasi penajaman, filtering, dan edge detection.
#  Tersedia dalam versi Manual (Loop), Standard PCDLib, dan Built-in (OpenCV).
# =============================================================================
#
#  DAFTAR FUNGSI:
#  ─────────────────────────────────────────────────────────────────────────────
#  [1] PADDING (Utilitas Dasar)
#      padding_manual(img, pad)       ← Padding manual dengan np.zeros
#      padding_builtin(img, pad)      ← Padding menggunakan np.pad
#
#  [2] SPATIAL FILTERING (Sliding Window)
#      filter_spasial_manual(...)     ← Mean/Median/Modus dengan loops murni
#      filter_spasial_fast_pad(...)   ← Menggunakan np.pad (lebih cepat)
#      mean_filter_standard(...)      ← Implementasi standar PCDLib
#      median_filter_standard(...)    ← Implementasi standar PCDLib
#      mode_filter_standard(...)      ← Implementasi standar PCDLib
#
#  [3] KONVOLUSI & PENAJAMAN
#      convolution_manual(...)        ← Konvolusi manual (sliding window)
#      convolution_standard(...)      ← Konvolusi standar PCDLib
#      smoothing_standard(...)        ← Kernel smoothing PCDLib
#      sharpening_standard(...)       ← Kernel sharpening PCDLib
#
#  [4] EDGE DETECTION
#      edge_detection_manual(...)     ← Sobel/Prewitt/Roberts manual + Normalize
#      sobel_standard(...)            ← Sobel standar PCDLib
#      prewitt_standard(...)          ← Prewitt standar PCDLib
#      roberts_standard(...)          ← Roberts standar PCDLib
#      laplacian_manual(img)          ← Turunan kedua (sangat sensitif detail)
#      zero_crossing_manual(img)      ← Deteksi tepi via perubahan tanda Laplacian
#
#  [5] ADVANCED ENHANCEMENT
#      gamma_correction_manual(...)   ← Power Law (Perbaiki gelap/terang)
#      log_transformation_manual(...) ← Ekspansi nilai gelap
#      unsharp_masking_manual(...)    ← Penajaman tingkat tinggi (High-Boost)
#      brightness_contrast_manual(...)← Koreksi linear (alpha/beta)
#
#  [6] NOISE GENERATION
#      add_gaussian_noise_manual(...) ← Tambah noise distribusi normal
#      add_salt_pepper_manual(...)    ← Tambah noise titik hitam putih
#
#  [7] COMPOSITION & SLICING
#      crop_image_manual(img, x,y,w,h) ← Slicing ROI (Region of Interest)
#      merge_images_manual(img1, img2) ← Blending/Alpha Merging dua gambar
#      split_channels_manual(img)     ← Slicing [:,:,0] -> R (Api)
#      merge_channels_manual(R,G,B)   ← Assignment manual ke np.zeros
#
#  [8] UTILITAS LAIN
#      thresholding_manual(img, th)   ← Isolasi siluet (Ultra Instinct)
#      grayscale_manual(img)          ← Rumus 0.299R + 0.587G + 0.114B
#      normalize_manual(img)          ← Stretch ke 0-255
#  ─────────────────────────────────────────────────────────────────────────────


# -------------------------------------------------------------------------
# 1. PADDING
# -------------------------------------------------------------------------

def padding_manual(img, pad_size):
    """
    Menambahkan border hitam di sekeliling citra secara manual.
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        padded = np.zeros((h + 2*pad_size, w + 2*pad_size, img.shape[2]), dtype=img.dtype)
        padded[pad_size:pad_size+h, pad_size:pad_size+w, :] = img
    else:
        padded = np.zeros((h + 2*pad_size, w + 2*pad_size), dtype=img.dtype)
        padded[pad_size:pad_size+h, pad_size:pad_size+w] = img
    return padded


def padding_builtin(img, pad_size, mode='constant'):
    """
    Padding menggunakan fungsi bawaan NumPy.
    """
    if len(img.shape) == 3:
        return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode=mode)
    return np.pad(img, pad_size, mode=mode)


# -------------------------------------------------------------------------
# 2. SPATIAL FILTERING
# -------------------------------------------------------------------------

def filter_spasial_manual(img, size, mode='mean'):
    """
    Implementasi manual sliding window untuk Mean, Median, dan Modus filter.
    """
    pad = size // 2
    padded = padding_manual(img, pad)
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    
    if mode == 'mean':
        area = size * size
        for i in range(h):
            for j in range(w):
                region = padded[i:i+size, j:j+size]
                hasil[i, j] = np.sum(region) / area
                
    elif mode == 'median':
        for i in range(h):
            for j in range(w):
                region = padded[i:i+size, j:j+size]
                values = region.flatten()
                values.sort()
                hasil[i, j] = values[len(values)//2]

    elif mode == 'modus':
        for i in range(h):
            for j in range(w):
                region = padded[i:i+size, j:j+size]
                values = region.flatten()
                counts = {}
                for val in values:
                    counts[val] = counts.get(val, 0) + 1
                modus_val = max(counts, key=counts.get)
                hasil[i, j] = modus_val
    
    return hasil.astype(np.uint8)


def mean_filter_standard(img, size=3):
    """Implementasi standar PCDLib: Mean Filter."""
    pad = size // 2
    padded = np.pad(img, pad, mode='edge')
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    area = size * size

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            hasil[i, j] = np.sum(region) / area
    return hasil.astype(np.uint8)


def median_filter_standard(img, size=3):
    """Implementasi standar PCDLib: Median Filter."""
    pad = size // 2
    padded = np.pad(img, pad, mode='edge')
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            hasil[i, j] = np.median(region)
    return hasil.astype(np.uint8)


def mode_filter_standard(img, size=3):
    """Implementasi standar PCDLib: Mode (Modus) Filter."""
    pad = size // 2
    padded = np.pad(img, pad, mode='edge')
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            values = region.ravel()
            count = {}
            for val in values:
                count[val] = count.get(val, 0) + 1
            mode_val = max(count, key=count.get)
            hasil[i, j] = mode_val
    return hasil.astype(np.uint8)


def filter_spasial_builtin(img, size, mode='mean'):
    """
    Versi cepat menggunakan OpenCV.
    """
    if mode == 'mean':
        return cv2.blur(img, (size, size))
    elif mode == 'median':
        return cv2.medianBlur(img, size)
    return img


def filter_spasial_fast_pad(img, size, mode='mean', pad_type='constant'):
    """
    Filter spasial yang menggunakan np.pad (NumPy) untuk handle berbagai jenis border.
    """
    pad = size // 2
    if len(img.shape) == 3:
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode=pad_type)
    else:
        padded = np.pad(img, pad, mode=pad_type)
        
    h, w = img.shape[:2]
    hasil = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            if mode == 'mean':
                hasil[i, j] = np.mean(region)
            elif mode == 'median':
                hasil[i, j] = np.median(region)
            elif mode == 'modus':
                values = region.flatten()
                hasil[i, j] = np.argmax(np.bincount(values.astype(int)))
    return hasil.astype(np.uint8)


# -------------------------------------------------------------------------
# 3. KONVOLUSI & PENAJAMAN
# -------------------------------------------------------------------------

def convolution_manual(img, kernel):
    """
    Operasi konvolusi manual 2D.
    """
    size = kernel.shape[0]
    pad = size // 2
    padded = padding_manual(img, pad).astype(np.float32)

    h, w = img.shape[:2]
    hasil = np.zeros_like(img).astype(np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            hasil[i, j] = np.sum(region * kernel)

    return hasil


def convolution_standard(img, kernel):
    """Implementasi standar PCDLib: Konvolusi."""
    size = kernel.shape[0]
    pad = size // 2
    padded = np.pad(img, pad, mode='constant')
    h, w = img.shape[:2]
    hasil = np.zeros_like(img).astype(np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+size, j:j+size]
            hasil[i, j] = np.sum(region * kernel)
    return hasil


def convolution_builtin(img, kernel):
    """
    Konvolusi menggunakan OpenCV cv2.filter2D.
    """
    return cv2.filter2D(img, -1, kernel)


def smoothing_standard(img):
    """Implementasi standar PCDLib: Smoothing."""
    kernel = np.array([[1/10, 1/10, 1/10], [1/10, 1/5, 1/10], [1/10, 1/10, 1/10]])
    hasil = convolution_standard(img, kernel)
    return np.clip(hasil, 0, 255).astype(np.uint8)


def sharpening_standard(img):
    """Implementasi standar PCDLib: Sharpening."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    hasil = convolution_standard(img, kernel)
    return np.clip(hasil, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# 4. EDGE DETECTION
# -------------------------------------------------------------------------

def edge_detection_manual(img, kernel_x, kernel_y):
    """
    Deteksi tepi manual dengan menggabungkan magnitudo gradien X dan Y.
    """
    gx = convolution_manual(img, kernel_x)
    gy = convolution_manual(img, kernel_y)
    hasil = np.abs(gx) + np.abs(gy)
    return normalize_manual(hasil)


def sobel_standard(img):
    """Implementasi standar PCDLib: Sobel."""
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = convolution_standard(img, sx)
    gy = convolution_standard(img, sy)
    hasil = np.abs(gx) + np.abs(gy)
    return normalize_manual(hasil)


def prewitt_standard(img):
    """Implementasi standar PCDLib: Prewitt."""
    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gx = convolution_standard(img, px)
    gy = convolution_standard(img, py)
    hasil = np.abs(gx) + np.abs(gy)
    return normalize_manual(hasil)


def roberts_standard(img):
    """Implementasi standar PCDLib: Roberts."""
    rx = np.array([[1, 0], [0, -1]])
    ry = np.array([[0, 1], [-1, 0]])
    gx = convolution_standard(img, rx)
    gy = convolution_standard(img, ry)
    hasil = np.abs(gx) + np.abs(gy)
    return normalize_manual(hasil)


def laplacian_manual(img):
    """
    Deteksi tepi turunan kedua (Laplacian).
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    hasil = convolution_manual(img, kernel)
    return normalize_manual(hasil)


def zero_crossing_manual(img, threshold=0):
    """
    Mendeteksi tepi melalui perubahan tanda pada hasil Laplacian (Zero Crossing).
    """
    lap = convolution_manual(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    h, w = lap.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = lap[y-1:y+2, x-1:x+2]
            if patch.min() < -threshold and patch.max() > threshold:
                out[y, x] = 255
    return out


# -------------------------------------------------------------------------
# 5. ADVANCED ENHANCEMENT
# -------------------------------------------------------------------------

def brightness_contrast_manual(img, alpha=1.0, beta=0):
    """
    Koreksi kecerahan dan kontras linear: s = alpha * r + beta
    """
    hasil = img.astype(np.float32) * alpha + beta
    return np.clip(hasil, 0, 255).astype(np.uint8)


def gamma_correction_manual(img, gamma=1.0):
    """
    Power Law Transformation: s = c * r^gamma
    """
    img_norm = img / 255.0
    hasil = np.power(img_norm, gamma)
    return (hasil * 255.0).astype(np.uint8)


def log_transformation_manual(img):
    """
    Log Transformation: s = c * log(1 + r)
    """
    c = 255 / np.log(1 + np.max(img))
    hasil = c * (np.log(1 + img))
    return hasil.astype(np.uint8)


def unsharp_masking_manual(img, size=3, k=1.0):
    """
    Unsharp Masking: sharp = original + k * (original - blurred)
    """
    blur = filter_spasial_manual(img, size, mode='mean')
    mask = img.astype(np.float32) - blur.astype(np.float32)
    hasil = img.astype(np.float32) + k * mask
    return np.clip(hasil, 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# 6. NOISE GENERATION
# -------------------------------------------------------------------------

def add_gaussian_noise_manual(img, mean=0, sigma=25):
    """
    Menambahkan noise distribusi Normal (Gaussian) ke citra.
    """
    noise = np.random.normal(mean, sigma, img.shape)
    hasil = img.astype(np.float32) + noise
    return np.clip(hasil, 0, 255).astype(np.uint8)


def add_salt_pepper_manual(img, salt_prob=0.01, pepper_prob=0.01):
    """
    Menambahkan noise Salt (putih) dan Pepper (hitam) secara acak.
    """
    hasil = np.copy(img)
    salt_mask = np.random.rand(*img.shape[:2]) < salt_prob
    hasil[salt_mask] = 255
    pepper_mask = np.random.rand(*img.shape[:2]) < pepper_prob
    hasil[pepper_mask] = 0
    return hasil


# -------------------------------------------------------------------------
# 7. COMPOSITION & SLICING
# -------------------------------------------------------------------------

def crop_image_manual(img, x, y, w, h):
    """
    Slicing manual untuk mengambil area tertentu (ROI).
    """
    return img[y:y+h, x:x+w]


def merge_images_manual(img1, img2, alpha=0.5):
    """
    Blending dua gambar dengan bobot transparansi.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    hasil = (img1.astype(np.float32) * alpha) + (img2.astype(np.float32) * (1 - alpha))
    return hasil.astype(np.uint8)


def split_channels_manual(img):
    """
    Memecah citra RGB menjadi channel R, G, dan B secara manual.
    """
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def merge_channels_manual(R, G, B):
    """
    Menggabungkan kembali channel R, G, B menjadi satu citra.
    """
    h, w = R.shape
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    merged[:, :, 0] = R
    merged[:, :, 1] = G
    merged[:, :, 2] = B
    return merged


# -------------------------------------------------------------------------
# 8. UTILITAS
# -------------------------------------------------------------------------

def thresholding_manual(img, thresh_val=127):
    """
    Manual binary thresholding untuk membuat mask.
    """
    return np.where(img > thresh_val, 255, 0).astype(np.uint8)


def grayscale_manual(img):
    """
    Mengubah RGB ke Grayscale menggunakan rumus luminositas.
    """
    if len(img.shape) == 3:
        R, G, B = split_channels_manual(img)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray.astype(np.uint8)
    return img


def normalize_manual(img):
    """
    Normalisasi nilai piksel ke rentang 0-255.
    """
    img = np.abs(img)
    if np.max(img) > 0:
        img = (img / np.max(img)) * 255
    return img.astype(np.uint8)


def read(path, gray=False):
    """
    Membaca gambar (JPG, PNG, dll) dan mengubahnya ke format NumPy array.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Gambar tidak ditemukan di {path}")
        return None
    
    if gray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
        
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]] # BGR to RGB
    return img


def save(path, img, rgb=True):
    """Menyimpan gambar (otomatis konversi ke BGR untuk OpenCV)."""
    if rgb and len(img.shape) == 3:
        img = img[:, :, [2, 1, 0]] # RGB to BGR
    cv2.imwrite(path, img)


# -------------------------------------------------------------------------
# 9. SAFETY & FIXER UTILITIES (PENCEGAH ERROR)
# -------------------------------------------------------------------------

def ensure_rgb(img):
    """Memastikan citra memiliki 3 channel (RGB). Jika Gray, maka diconvert."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def ensure_grayscale(img):
    """Memastikan citra adalah 1 channel (Grayscale). Jika RGB, maka diconvert."""
    if len(img.shape) == 3:
        return grayscale_manual(img)
    return img


def match_size_manual(img_target, img_source):
    """Menyamakan ukuran img_source agar sama dengan img_target (Manual Nearest Neighbor)."""
    h_target, w_target = img_target.shape[:2]
    h_src, w_src = img_source.shape[:2]
    
    # Rasio skala
    y_ratio = h_src / h_target
    x_ratio = w_src / w_target
    
    # Buat hasil kosong
    if len(img_source.shape) == 3:
        hasil = np.zeros((h_target, w_target, img_source.shape[2]), dtype=img_source.dtype)
    else:
        hasil = np.zeros((h_target, w_target), dtype=img_source.dtype)
        
    for i in range(h_target):
        for j in range(w_target):
            # Ambil tetangga terdekat secara manual
            orig_i = int(i * y_ratio)
            orig_j = int(j * x_ratio)
            # Pastikan tidak out of bounds
            orig_i = min(orig_i, h_src - 1)
            orig_j = min(orig_j, w_src - 1)
            hasil[i, j] = img_source[orig_i, orig_j]
    return hasil


def fix_dtype_manual(img):
    """Memastikan nilai 0-255 dan tipe uint8 secara manual (tanpa np.clip)."""
    hasil = np.copy(img)
    hasil[hasil < 0] = 0
    hasil[hasil > 255] = 255
    return hasil.astype(np.uint8)


# =============================================================================
#  QUICK REFERENCE GUIDE (CHEATSHEET)
# =============================================================================
#
#  [1] OPENCV (cv2) - I/O & Dasar
#  ─────────────────────────────────────────────────────────────────────────────
#  cv2.imread(path, flag)  -> Baca gambar.
#      Flag: cv2.IMREAD_COLOR (1)      : 3-channel BGR (default)
#            cv2.IMREAD_GRAYSCALE (0)  : 1-channel Gray
#            cv2.IMREAD_UNCHANGED (-1) : Termasuk Alpha channel (PNG)
#
#  cv2.imshow(win, img)    -> Tampilkan jendela gambar (khusus GUI desktop).
#  cv2.waitKey(0)          -> Menunggu input keyboard (0 = selamanya).
#  cv2.destroyAllWindows() -> Menutup semua jendela OpenCV.
#
#  cv2.cvtColor(img, code) -> Konversi ruang warna.
#      Code: cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2GRAY
#
#  [2] MATPLOTLIB (plt) - Visualisasi
#  ─────────────────────────────────────────────────────────────────────────────
#  plt.imshow(img, cmap='gray') -> Plot array sebagai gambar.
#  plt.title("Judul")           -> Memberi judul.
#  plt.axis('off')              -> Menghilangkan angka koordinat x & y.
#  plt.show()                   -> Menampilkan plot ke layar.
#
#  plt.subplot(rows, cols, idx) -> Membuat banyak gambar dalam satu figure.
#
#  [3] NUMPY (np) - Manipulasi Array
#  ─────────────────────────────────────────────────────────────────────────────
#  img.shape -> (Tinggi, Lebar, Channel).
#  img.dtype -> Tipe data (biasanya uint8 atau float32).
#  np.zeros((h, w), dtype=np.uint8) -> Gambar hitam polos.
#  np.ones((h, w)) * 255            -> Gambar putih polos.
# ─────────────────────────────────────────────────────────────────────────────
#
#  [4] TROUBLESHOOTING & COMMON ERRORS (PENTING!)
#  ─────────────────────────────────────────────────────────────────────────────
#  1. ERROR: "not enough values to unpack (expected 3, got 100)"
#     Penyebab: Mencoba split R, G, B pada gambar Grayscale.
#     Solusi  : Gunakan ensure_rgb(img) sebelum melakukan splitting.
#
#  2. ERROR: "operands could not be broadcast together with shapes..."
#     Penyebab: Menjumlahkan dua gambar yang ukurannya (h, w) berbeda.
#     Solusi  : Gunakan match_size(img1, img2) sebelum menjumlahkan.
#
#  3. ERROR: "AttributeError: 'NoneType' object has no attribute 'shape'"
#     Penyebab: Gambar tidak ditemukan saat read('path').
#     Solusi  : Cek lokasi file atau typo pada nama file.
#
#  4. ERROR: "TypeError: 'module' object is not callable"
#     Penyebab: Biasanya karena nama variabel sama dengan nama library (ex: cv2 = read('..')).
#     Solusi  : Jangan gunakan nama 'cv2', 'np', atau 'plt' sebagai nama variabel gambar.
#
#  5. HASIL GAMBAR ANEH (Dominan Biru/Merah):
#     Penyebab: Tertukar antara format RGB (Matplotlib) dan BGR (OpenCV).
#     Solusi  : Gunakan fungsi read() dan save() yang sudah saya buat (otomatis).
#
#  6. HASIL GAMBAR GELAP/HILANG:
#     Penyebab: Lupa melakukan np.clip() atau normalisasi setelah konvolusi.
#     Solusi  : Gunakan fix_dtype(img) sebelum menampilkan atau menyimpan.
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
#  CONTOH PENGGUNAAN (MAIN BLOCK)
# =============================================================================
if __name__ == "__main__":
    # Dummy image untuk demonstrasi (100x100 pixel RGB)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 1. Contoh Standard PCDLib (Mean Filter)
    img_gray = grayscale_manual(img)
    hasil_pcd = mean_filter_standard(img_gray, size=3)
    
    # 2. Contoh Standard Sobel
    tepi_sobel = sobel_standard(img_gray)
    
    # 3. Contoh Padding
    img_padded = padding_manual(img, pad_size=10)
    
    print("Selesai! Implementasi Standar PCDLib telah ditambahkan.")
    print(f"Tipe data: {type(img)}") 
    print(f"Ukuran: {img.shape}")