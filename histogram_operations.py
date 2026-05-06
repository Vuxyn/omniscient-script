import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  HISTOGRAM OPERATIONS CHEATSHEET (MODUL 2) — Class-Based & Manual NumPy
#  Semua operasi dibuat secara manual menggunakan indexing dan loop.
#  Boleh menggunakan fungsi MATEMATIKA numpy: zeros, cumsum, argmin, dll.
# =============================================================================
#
#  DAFTAR FUNGSI (Dalam Class HistogramProcessor):
#  ─────────────────────────────────────────────────────────────────────────────
#  [1] HISTOGRAM DASAR
#      calculate_histogram(img)      ← Menghitung frekuensi tiap nilai piksel
#      calculate_cdf(hist)           ← Akumulasi histogram
#      calculate_cdf_normalized(cdf) ← CDF dinormalisasi 0-1
#      normalize_min_max(img)        ← Normalisasi linear 0-255
#
#  [2] EQUALIZATION & SPECIFICATION
#      equalize(img)                 ← Ekualisasi histogram (soal 2)
#      histogram_specification(src, target)
#                                    ← Matching ke target (soal 3 & 5)
#      selective_specification(img, target, mask)
#                                    ← Spesifikasi berbeda per bagian (soal 5)
#
#  [3] GEOMETRI & DIAGONAL
#      diagonal_split(img)           ← Memecah gambar jadi 2 segitiga (diagonal)
#      diagonal_combine(bawah, atas) ← Menggabungkan kembali 2 segitiga diagonal
#      resize_manual(img, h, w)      ← Resize manual (nearest neighbor)
#
#  [4] ADVANCED ENHANCEMENT
#      contrast_stretching(img, low, high)
#                                    ← Penarikan kontras linear
#      otsu_threshold(img)           ← Thresholding otomatis berbasis variansi
#      local_enhancement(img, ...)   ← Statistik histogram lokal (soal sulit)
#
#  [5] METRICS & VISUALISASI
#      calculate_metrics(orig, proc) ← MSE & PSNR (Kualitas citra)
#      plot_histogram(img, title)    ← Plot histogram menggunakan Matplotlib
#      show_comparison(img1, img2)   ← Menampilkan perbandingan gambar
#  ─────────────────────────────────────────────────────────────────────────────

class HistogramProcessor:
    """
    Class untuk menangani operasi pengolahan citra berbasis histogram.
    Mendukung pemanggilan statis maupun melalui instansiasi objek.
    """

    def __init__(self, image=None):
        """
        Inisialisasi objek dengan citra (opsional).
        
        Args:
            image: np.array citra (grayscale atau RGB)
        """
        self.image = image

    # -------------------------------------------------------------------------
    # 1. HISTOGRAM DASAR
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_histogram(img):
        """
        Menghitung frekuensi tiap nilai piksel (0-255).
        """
        h, w = img.shape[:2]
        hist = np.zeros(256, dtype=float)
        
        for i in range(h):
            for j in range(w):
                pixel_val = int(img[i, j])
                hist[pixel_val] += 1
        return hist

    @staticmethod
    def calculate_histogram_normalized(img):
        """
        Menghitung histogram yang sudah dinormalisasi (Probabilitas).
        """
        bins = [i for i in range(256)]
        hist = HistogramProcessor.calculate_histogram(img)
        hist_norm = hist / (np.sum(hist) + 1e-8)
        return bins, hist_norm

    @staticmethod
    def calculate_cdf(hist):
        """
        Menghitung Cumulative Distribution Function (CDF) dari histogram.
        
        Args:
            hist: np.array histogram
        Returns:
            np.array CDF
        """
        cdf = np.zeros(256, dtype=float)
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + hist[i]
        return cdf

    @staticmethod
    def calculate_cdf_normalized(cdf):
        """
        Normalisasi CDF ke rentang [0, 1].
        """
        total_pixels = cdf[-1]
        return cdf / (total_pixels + 1e-8)

    @staticmethod
    def normalize_min_max(image):
        """
        Normalisasi citra menggunakan Min-Max ke rentang 0-255.
        
        Args:
            image: np.array citra
        Returns:
            np.array citra ternormalisasi
        """
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8) 
        norm = (image - min_val) / (max_val - min_val)
        return (norm * 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 2. EQUALIZATION
    # -------------------------------------------------------------------------

    @staticmethod
    def equalize(img):
        """
        Ekualisasi Histogram (Manual).
        Menggunakan rumus: round(cdf(i) * 255 / (H*W))
        """
        h, w = img.shape[:2]
        
        # 1. Hitung Histogram
        hist = HistogramProcessor.calculate_histogram(img)
        
        # 2. Hitung CDF
        cdf = np.zeros(256, dtype=int)
        for i in range(256):
            cdf[i] = np.sum(hist[0:i+1])
        
        # 3. Transformasi Nilai (LUT)
        result_lut = np.round((cdf * 255) / (h * w)).astype(np.uint8)
        
        # 4. Terapkan LUT
        hasil = np.zeros_like(img, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = result_lut[img[i, j]]
                
        return hasil

    @staticmethod
    def histogram_specification(source_img, target_img):
        """
        Histogram Specification / Matching (Manual).
        Mencari mapping nilai source yang paling dekat dengan target.
        """
        h, w = source_img.shape[:2]
        
        # Step A: CDF Source (Scaled 0-255)
        hist_s = HistogramProcessor.calculate_histogram(source_img)
        cdf_s = np.cumsum(hist_s)
        cdf_s = np.round(cdf_s * 255 / cdf_s[-1]).astype(np.uint8)
        
        # Step B: CDF Target (Scaled 0-255)
        hist_t = HistogramProcessor.calculate_histogram(target_img)
        cdf_t = np.cumsum(hist_t)
        cdf_t = np.round(cdf_t * 255 / cdf_t[-1]).astype(np.uint8)
        
        # Step C: Buat Mapping LUT (Nearest Match)
        mapping = np.zeros(256, dtype=np.uint8)
        for s_val in range(256):
            diff = np.abs(cdf_s[s_val] - cdf_t)
            mapping[s_val] = np.argmin(diff)
            
        # Step D: Terapkan Mapping
        hasil = np.zeros_like(source_img)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = mapping[source_img[i, j]]
                
        return hasil

    @staticmethod
    def selective_specification(img, target_img, mask):
        """
        Penerapan Histogram Matching hanya pada area tertentu berdasarkan mask.
        """
        # 1. Lakukan matching full
        matched_full = HistogramProcessor.histogram_specification(img, target_img)
        
        # 2. Gabungkan menggunakan mask
        h, w = img.shape[:2]
        hasil = np.zeros_like(img)
        
        # Normalisasi mask ke boolean
        mask_bool = mask > 127 if np.max(mask) > 1 else mask > 0.5
        
        for i in range(h):
            for j in range(w):
                if mask_bool[i, j]:
                    hasil[i, j] = matched_full[i, j]
                else:
                    hasil[i, j] = img[i, j]
                    
        return hasil

    # -------------------------------------------------------------------------
    # 4. GEOMETRI & DIAGONAL
    # -------------------------------------------------------------------------

    @staticmethod
    def diagonal_split(image):
        """
        Memecah citra menjadi bagian bawah diagonal dan atas diagonal.
        """
        h, w = image.shape[:2]
        bawah = np.zeros_like(image)
        atas = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                if i >= j:
                    bawah[i, j] = image[i, j]
                else:
                    atas[i, j] = image[i, j]
        return bawah, atas

    @staticmethod
    def diagonal_combine(bawah, atas):
        """
        Menggabungkan dua citra berdasarkan pembatas diagonal.
        """
        h, w = bawah.shape[:2]
        hasil = np.zeros_like(bawah, dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i > j:
                    hasil[i, j] = bawah[i, j] 
                else:
                    hasil[i, j] = atas[i, j]   
        return hasil

    @staticmethod
    def resize_manual(image, target_h, target_w):
        """
        Resize citra secara manual menggunakan nearest neighbor.
        """
        h_orig, w_orig = image.shape[:2]
        hasil = np.zeros((target_h, target_w), dtype=np.uint8)

        for i in range(target_h):
            for j in range(target_w):
                src_y = int(i * h_orig / target_h)
                src_x = int(j * w_orig / target_w)

                src_y = min(src_y, h_orig - 1)
                src_x = min(src_x, w_orig - 1)

                hasil[i, j] = image[src_y, src_x]

        return hasil

    # -------------------------------------------------------------------------
    # 5. ADVANCED ENHANCEMENT
    # -------------------------------------------------------------------------

    @staticmethod
    def contrast_stretching(img, r_min, r_max):
        """
        Linear Contrast Stretching.
        Menarik rentang [r_min, r_max] menjadi [0, 255].
        """
        h, w = img.shape[:2]
        hasil = np.zeros_like(img, dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                val = img[i, j]
                if val <= r_min:
                    hasil[i, j] = 0
                elif val >= r_max:
                    hasil[i, j] = 255
                else:
                    # Rumus: (val - r_min) * (255 / (r_max - r_min))
                    hasil[i, j] = np.round((val - r_min) * (255 / (r_max - r_min)))
        return hasil

    @staticmethod
    def otsu_threshold(img):
        """
        Otsu's Thresholding Manual.
        Mencari threshold optimal dengan memaksimalkan between-class variance.
        """
        hist = HistogramProcessor.calculate_histogram(img)
        total_pixels = img.size
        
        current_max = -1
        threshold = 0
        
        sum_total = np.sum(np.arange(256) * hist)
        weight_bg = 0
        sum_bg = 0
        
        for i in range(256):
            weight_bg += hist[i]
            if weight_bg == 0: continue
            
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0: break
            
            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            
            # Between-class variance
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg)**2
            
            if var_between > current_max:
                current_max = var_between
                threshold = i
                
        # Terapkan threshold
        h, w = img.shape[:2]
        hasil = np.zeros_like(img, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = 255 if img[i, j] > threshold else 0
                
        return threshold, hasil

    @staticmethod
    def local_enhancement(img, E=4.0, k0=0.4, k1=0.02, k2=0.4, window_size=3):
        """
        Local Histogram Statistics Enhancement.
        Mencerahkan area yang gelap (mean rendah) dan memiliki variansi rendah.
        """
        h, w = img.shape[:2]
        hasil = np.copy(img).astype(float)
        
        # Statistik Global
        m_global = np.mean(img)
        sigma_global = np.std(img)
        
        pad = window_size // 2
        img_pad = np.pad(img, pad, mode='reflect')
        
        for i in range(h):
            for j in range(w):
                # Ambil window lokal
                window = img_pad[i:i+window_size, j:j+window_size]
                m_local = np.mean(window)
                sigma_local = np.std(window)
                
                # Syarat penguatan:
                # 1. Mean lokal <= k0 * Mean Global (Area Gelap)
                # 2. k1 * Std Global <= Std Lokal <= k2 * Std Global (Kontras Rendah tapi ada detail)
                if m_local <= k0 * m_global and (k1 * sigma_global <= sigma_local <= k2 * sigma_global):
                    hasil[i, j] = img[i, j] * E
                    
        return np.clip(hasil, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # 6. METRICS & VISUALISASI
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(original, processed):
        """
        Menghitung MSE dan PSNR untuk membandingkan kualitas citra.
        """
        mse = np.mean((original.astype(float) - processed.astype(float))**2)
        if mse == 0:
            return 0, 100 # Identik
        
        psnr = 10 * np.log10((255**2) / mse)
        return mse, psnr

    @staticmethod
    def plot_histogram(img, title="Histogram"):
        """Plot histogram menggunakan matplotlib."""
        hist = HistogramProcessor.calculate_histogram(img)
        plt.figure(figsize=(10, 4))
        plt.bar(range(256), hist, color='gray', alpha=0.7)
        plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def show_comparison(img1, img2, title1="Original", title2="Processed"):
        """Tampilkan dua gambar bersebelahan."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title(title1)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(title2)
        plt.axis('off')
        plt.show()

# =============================================================================
#  CONTOH PENGGUNAAN (MAIN BLOCK)
# =============================================================================
if __name__ == "__main__":
    # Buat citra dummy untuk testing
    dummy_source = np.random.randint(0, 100, (100, 100), dtype=np.uint8)
    dummy_target = np.random.randint(150, 255, (100, 100), dtype=np.uint8)
    
    # 1. Instance usage
    proc = HistogramProcessor(dummy_source)
    hist = proc.calculate_histogram(proc.image)
    print(f"Histogram sum: {np.sum(hist)} (Expected: {dummy_source.size})")
    
    # 2. Equalization
    equalized = HistogramProcessor.equalize(dummy_source)
    
    # 3. Specification
    matched = HistogramProcessor.histogram_specification(dummy_source, dummy_target)
    
    # 4. Selective
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1 # Area kotak di tengah
    selective = HistogramProcessor.selective_specification(dummy_source, dummy_target, mask)
    
    print("Operasi berhasil dijalankan!")
