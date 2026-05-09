import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  HISTOGRAM OPERATIONS CHEATSHEET (MODUL 2) — Class-Based & Manual Loops
#  Semua operasi dibuat secara manual menggunakan nested loops.
#  Sesuai request: HANYA BOLEH menggunakan np.zeros dan np.round.
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
#      equalize(img)                 ← Ekualisasi histogram 
#      histogram_specification(src, target)
#                                    ← Matching ke target 
#      selective_specification(img, target, mask)
#                                    ← Spesifikasi berbeda per bagian 
#
#  [3] COMPOSITION & BLENDING
#      merge_image(img1, img2, alpha) ← Blending dua citra (transparansi)
#      replace_background(fg, bg, ...) ← Ganti background manual (masking)
#      apply_mask(img, mask)         ← Menerapkan mask (area di luar mask jadi hitam)
#      create_rectangle_mask(h, w, ...) ← Membuat mask persegi panjang manual
#      create_circle_mask(h, w, ...)    ← Membuat mask lingkaran manual
#
#  [4] GEOMETRI & DIAGONAL
#      diagonal_split(img)           ← Memecah gambar jadi 2 segitiga (diagonal)
#      diagonal_combine(bawah, atas) ← Menggabungkan kembali 2 segitiga diagonal
#      resize_manual(img, h, w)      ← Resize manual (nearest neighbor)
#
#  [5] ADVANCED ENHANCEMENT
#      contrast_stretching(img, low, high)
#                                    ← Penarikan kontras linear
#      otsu_threshold(img)           ← Thresholding otomatis berbasis variansi
#      local_enhancement(img, ...)   ← Statistik histogram lokal 
#
#  [6] METRICS & VISUALISASI
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
        
        # Manual sum
        total = 0
        for val in hist:
            total += val
            
        hist_norm = np.zeros(256, dtype=float)
        for i in range(256):
            hist_norm[i] = hist[i] / (total + 1e-8)
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
        """
        h, w = image.shape[:2]
        is_rgb = len(image.shape) == 3
        
        # Manual min max
        if is_rgb:
            min_val = image[0, 0, 0]
            max_val = image[0, 0, 0]
            for i in range(h):
                for j in range(w):
                    for c in range(3):
                        if image[i, j, c] < min_val: min_val = image[i, j, c]
                        if image[i, j, c] > max_val: max_val = image[i, j, c]
        else:
            min_val = image[0, 0]
            max_val = image[0, 0]
            for i in range(h):
                for j in range(w):
                    if image[i, j] < min_val: min_val = image[i, j]
                    if image[i, j] > max_val: max_val = image[i, j]

        if max_val == min_val:
            return np.zeros(image.shape, dtype=np.uint8)
            
        hasil = np.zeros(image.shape, dtype=np.uint8)
        diff = max_val - min_val
        for i in range(h):
            for j in range(w):
                if is_rgb:
                    for c in range(3):
                        norm = (image[i, j, c] - min_val) / diff
                        hasil[i, j, c] = int(np.round(norm * 255))
                else:
                    norm = (image[i, j] - min_val) / diff
                    hasil[i, j] = int(np.round(norm * 255))
        return hasil

    # -------------------------------------------------------------------------
    # 2. EQUALIZATION
    # -------------------------------------------------------------------------

    @staticmethod
    def equalize(img):
        """
        Ekualisasi Histogram (Manual).
        """
        h, w = img.shape[:2]
        hist = HistogramProcessor.calculate_histogram(img)
        
        # Manual CDF
        cdf = np.zeros(256, dtype=float)
        running_sum = 0
        for i in range(256):
            running_sum += hist[i]
            cdf[i] = running_sum
        
        # Manual LUT with np.round
        result_lut = np.zeros(256, dtype=np.uint8)
        total_pixels = h * w
        for i in range(256):
            result_lut[i] = int(np.round((cdf[i] * 255) / total_pixels))
        
        hasil = np.zeros(img.shape, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = result_lut[int(img[i, j])]
                
        return hasil

    @staticmethod
    def histogram_specification(source_img, target_img):
        """
        Histogram Specification / Matching (Manual).
        """
        h, w = source_img.shape[:2]
        
        # Step A: CDF Source (Scaled 0-255)
        hist_s = HistogramProcessor.calculate_histogram(source_img)
        cdf_s = np.zeros(256, dtype=float)
        run_s = 0
        for i in range(256):
            run_s += hist_s[i]
            cdf_s[i] = run_s
        
        last_cdf_s = cdf_s[255] if cdf_s[255] > 0 else 1
        lut_s = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut_s[i] = int(np.round(cdf_s[i] * 255 / last_cdf_s))
        
        # Step B: CDF Target (Scaled 0-255)
        hist_t = HistogramProcessor.calculate_histogram(target_img)
        cdf_t = np.zeros(256, dtype=float)
        run_t = 0
        for i in range(256):
            run_t += hist_t[i]
            cdf_t[i] = run_t
            
        last_cdf_t = cdf_t[255] if cdf_t[255] > 0 else 1
        lut_t = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut_t[i] = int(np.round(cdf_t[i] * 255 / last_cdf_t))
        
        # Step C: Buat Mapping LUT (Nearest Match)
        mapping = np.zeros(256, dtype=np.uint8)
        for s_val in range(256):
            # Manual argmin with manual abs
            min_diff = 1e9
            best_t = 0
            s_cdf = lut_s[s_val]
            for t_val in range(256):
                t_cdf = lut_t[t_val]
                diff = s_cdf - t_cdf
                if diff < 0: diff = -diff # Manual abs
                
                if diff < min_diff:
                    min_diff = diff
                    best_t = t_val
            mapping[s_val] = best_t
            
        # Step D: Terapkan Mapping
        hasil = np.zeros(source_img.shape, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = mapping[int(source_img[i, j])]
                
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
        hasil = np.zeros(img.shape, dtype=np.uint8)
        
        # Normalisasi mask ke boolean (Manual max check)
        max_mask = 0
        for row in mask:
            for val in row:
                if val > max_mask: max_mask = val
        
        threshold = 127 if max_mask > 1 else 0.5
        
        for i in range(h):
            for j in range(w):
                if mask[i, j] > threshold:
                    hasil[i, j] = matched_full[i, j]
                else:
                    hasil[i, j] = img[i, j]
                    
        return hasil

    # -------------------------------------------------------------------------
    # 3. COMPOSITION & BLENDING
    # -------------------------------------------------------------------------

    @staticmethod
    def merge_image(img1, img2, alpha=0.5):
        """
        Menggabungkan dua citra dengan teknik blending (transparansi) manual.
        """
        h, w = img1.shape[:2]
        hasil = np.zeros(img1.shape, dtype=np.uint8)
        is_rgb = len(img1.shape) == 3
        
        for i in range(h):
            for j in range(w):
                if is_rgb:
                    for c in range(3):
                        val = (img1[i, j, c] * alpha) + (img2[i, j, c] * (1 - alpha))
                        val = np.round(val)
                        if val > 255: val = 255
                        if val < 0: val = 0
                        hasil[i, j, c] = int(val)
                else:
                    val = (img1[i, j] * alpha) + (img2[i, j] * (1 - alpha))
                    val = np.round(val)
                    if val > 255: val = 255
                    if val < 0: val = 0
                    hasil[i, j] = int(val)
                    
        return hasil
        # Contoh: res = HistogramProcessor.merge_image(img1, img2, alpha=0.7)

    @staticmethod
    def replace_background(foreground, background, threshold=240, mode='bright'):
        """
        Mengganti background pada citra foreground dengan citra background lain.
        """
        h, w = foreground.shape[:2]
        hasil = np.zeros(foreground.shape, dtype=np.uint8)
        is_rgb = len(foreground.shape) == 3
        
        for i in range(h):
            for j in range(w):
                # Manual intensity calculation
                if is_rgb:
                    # Manual average for intensity
                    total_p = 0
                    for c in range(3): total_p += foreground[i, j, c]
                    intensity = total_p / 3.0
                else:
                    intensity = foreground[i, j]
                
                # Check threshold
                is_bg = False
                if mode == 'bright':
                    if intensity >= threshold: is_bg = True
                else:
                    if intensity <= threshold: is_bg = True
                
                # Assign pixel
                if is_bg:
                    if is_rgb:
                        for c in range(3): hasil[i, j, c] = background[i, j, c]
                    else:
                        hasil[i, j] = background[i, j]
                else:
                    if is_rgb:
                        for c in range(3): hasil[i, j, c] = foreground[i, j, c]
                    else:
                        hasil[i, j] = foreground[i, j]
                        
        return hasil

    @staticmethod
    def merge_masked(foreground, background, mask):
        """
        Menggabungkan foreground dan background menggunakan mask (0-255).
        """
        h, w = foreground.shape[:2]
        hasil = np.zeros(foreground.shape, dtype=np.uint8)
        is_rgb = len(foreground.shape) == 3
        
        for i in range(h):
            for j in range(w):
                # Normalize mask
                alpha = mask[i, j] / 255.0
                
                if is_rgb:
                    for c in range(3):
                        val = (foreground[i, j, c] * alpha) + (background[i, j, c] * (1.0 - alpha))
                        val = np.round(val)
                        if val > 255: val = 255
                        if val < 0: val = 0
                        hasil[i, j, c] = int(val)
                else:
                    val = (foreground[i, j] * alpha) + (background[i, j] * (1.0 - alpha))
                    val = np.round(val)
                    if val > 255: val = 255
                    if val < 0: val = 0
                    hasil[i, j] = int(val)
                    
        return hasil

    @staticmethod
    def apply_mask(image, mask):
        """
        Menerapkan mask pada citra. Piksel yang tertutup mask (hitam) akan menjadi 0.
        """
        h, w = image.shape[:2]
        hasil = np.zeros(image.shape, dtype=np.uint8)
        is_rgb = len(image.shape) == 3
        
        # Manual max for mask normalization
        max_m = 0
        for row in mask:
            for val in row:
                if val > max_m: max_m = val
        
        threshold = 127 if max_m > 1 else 0.5
        
        for i in range(h):
            for j in range(w):
                if mask[i, j] > threshold:
                    if is_rgb:
                        for c in range(3): hasil[i, j, c] = image[i, j, c]
                    else:
                        hasil[i, j] = image[i, j]
                # else stays 0 from np.zeros
                        
        return hasil
                    # Jika RGB, set semua channel ke 0. Jika Gray, set piksel ke 0.
                    if is_rgb:
                        hasil[i, j] = [0, 0, 0]
                    else:
                        hasil[i, j] = 0
                        
        return hasil
        # Contoh: masked = HistogramProcessor.apply_mask(img, mask)

    @staticmethod
    def create_rectangle_mask(h, w, x, y, lebar, tinggi):
        """
        Membuat mask persegi panjang secara manual (tanpa CV2).
        
        Args:
            h, w: Tinggi dan lebar citra output
            x, y: Titik awal (kolom, baris)
            lebar, tinggi: Ukuran persegi
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(y, min(y + tinggi, h)):
            for j in range(x, min(x + lebar, w)):
                mask[i, j] = 255
        return mask

    @staticmethod
    def create_circle_mask(h, w, cx, cy, radius):
        """
        Membuat mask lingkaran secara manual (tanpa CV2).
        Menggunakan rumus Euclidean Distance: (x-cx)^2 + (y-cy)^2 <= r^2
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                # Hitung jarak piksel ke pusat
                dist_sq = (j - cx)**2 + (i - cy)**2
                if dist_sq <= radius**2:
                    mask[i, j] = 255
        return mask

    # -------------------------------------------------------------------------
    # 4. GEOMETRI & DIAGONAL
    # -------------------------------------------------------------------------

    @staticmethod
    def diagonal_split(image):
        """
        Memecah citra menjadi bagian bawah diagonal dan atas diagonal.
        """
        h, w = image.shape[:2]
        bawah = np.zeros(image.shape, dtype=np.uint8)
        atas = np.zeros(image.shape, dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i >= j:
                    if len(image.shape) == 3:
                        for c in range(3): bawah[i, j, c] = image[i, j, c]
                    else:
                        bawah[i, j] = image[i, j]
                else:
                    if len(image.shape) == 3:
                        for c in range(3): atas[i, j, c] = image[i, j, c]
                    else:
                        atas[i, j] = image[i, j]
        return bawah, atas

    @staticmethod
    def diagonal_combine(bawah, atas):
        """
        Menggabungkan dua citra berdasarkan pembatas diagonal.
        """
        h, w = bawah.shape[:2]
        hasil = np.zeros(bawah.shape, dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if i > j:
                    if len(bawah.shape) == 3:
                        for c in range(3): hasil[i, j, c] = bawah[i, j, c]
                    else:
                        hasil[i, j] = bawah[i, j] 
                else:
                    if len(atas.shape) == 3:
                        for c in range(3): hasil[i, j, c] = atas[i, j, c]
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
        """
        h, w = img.shape[:2]
        hasil = np.zeros(img.shape, dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                val = img[i, j]
                if val <= r_min:
                    hasil[i, j] = 0
                elif val >= r_max:
                    hasil[i, j] = 255
                else:
                    hasil[i, j] = int(np.round((val - r_min) * (255 / (r_max - r_min))))
        return hasil

    @staticmethod
    def otsu_threshold(img):
        """
        Otsu's Thresholding Manual.
        """
        hist = HistogramProcessor.calculate_histogram(img)
        h, w = img.shape[:2]
        total_pixels = h * w
        
        current_max = -1
        threshold = 0
        
        # Manual total sum for intensity
        sum_total = 0
        for i in range(256):
            sum_total += i * hist[i]
            
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
            diff_mean = mean_bg - mean_fg
            var_between = weight_bg * weight_fg * (diff_mean * diff_mean)
            
            if var_between > current_max:
                current_max = var_between
                threshold = i
                
        # Terapkan threshold
        hasil = np.zeros(img.shape, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hasil[i, j] = 255 if img[i, j] > threshold else 0
                
        return threshold, hasil

    @staticmethod
    def local_enhancement(img, E=4.0, k0=0.4, k1=0.02, k2=0.4, window_size=3):
        """
        Local Histogram Statistics Enhancement.
        """
        h, w = img.shape[:2]
        hasil = np.zeros(img.shape, dtype=np.uint8)
        
        # 1. Statistik Global (Manual)
        sum_g = 0
        sum_sq_g = 0
        total_p = h * w
        for i in range(h):
            for j in range(w):
                v = float(img[i, j])
                sum_g += v
                sum_sq_g += v * v
        
        m_global = sum_g / total_p
        # Variance = E[X^2] - E[X]^2
        var_global = (sum_sq_g / total_p) - (m_global * m_global)
        sigma_global = var_global**0.5 # Square root
        
        pad = window_size // 2
        
        for i in range(h):
            for j in range(w):
                # 2. Ambil window lokal (Manual with boundary check)
                sum_l = 0
                sum_sq_l = 0
                count_l = 0
                for r in range(i - pad, i + pad + 1):
                    for c in range(j - pad, j + pad + 1):
                        # Reflect boundary or clamp
                        rr = r
                        if rr < 0: rr = -r
                        if rr >= h: rr = 2*h - r - 1
                        cc = c
                        if cc < 0: cc = -c
                        if cc >= w: cc = 2*w - c - 1
                        
                        v = float(img[rr, cc])
                        sum_l += v
                        sum_sq_l += v * v
                        count_l += 1
                
                m_local = sum_l / count_l
                var_local = (sum_sq_l / count_l) - (m_local * m_local)
                if var_local < 0: var_local = 0
                sigma_local = var_local**0.5
                
                # Syarat penguatan
                val = float(img[i, j])
                if m_local <= k0 * m_global and (k1 * sigma_global <= sigma_local <= k2 * sigma_global):
                    val = val * E
                
                # Manual clip
                if val > 255: val = 255
                if val < 0: val = 0
                hasil[i, j] = int(np.round(val))
                    
        return hasil

    # -------------------------------------------------------------------------
    # 6. METRICS & VISUALISASI
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(original, processed):
        """
        Menghitung MSE dan PSNR untuk membandingkan kualitas citra.
        """
        h, w = original.shape[:2]
        total_err = 0
        for i in range(h):
            for j in range(w):
                if len(original.shape) == 3:
                    for c in range(3):
                        diff = float(original[i, j, c]) - float(processed[i, j, c])
                        total_err += diff * diff
                else:
                    diff = float(original[i, j]) - float(processed[i, j])
                    total_err += diff * diff
        
        # Mean squared error
        count = h * w * (3 if len(original.shape) == 3 else 1)
        mse = total_err / count
        
        if mse == 0:
            return 0, 100 # Identik
        
        # PSNR using manual log approximation or math.log10
        import math
        psnr = 10 * math.log10((255*255) / mse)
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
    # Buat citra dummy manual (Hanya np.zeros)
    dummy_source = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            dummy_source[i, j] = (i + j) % 256
            
    dummy_target = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            dummy_target[i, j] = (i * j) % 256
    
    # 1. Instance usage
    proc = HistogramProcessor(dummy_source)
    hist = proc.calculate_histogram(proc.image)
    
    # Manual sum for print
    s = 0
    for v in hist: s += v
    print(f"Histogram sum: {s} (Expected: {100*100})")
    
    # 2. Equalization
    equalized = HistogramProcessor.equalize(dummy_source)
    
    # 3. Specification
    matched = HistogramProcessor.histogram_specification(dummy_source, dummy_target)
    
    # 4. Selective
    mask = np.zeros((100, 100), dtype=np.uint8)
    for i in range(25, 75):
        for j in range(25, 75):
            mask[i, j] = 255
    selective = HistogramProcessor.selective_specification(dummy_source, dummy_target, mask)
    
    print("Operasi berhasil dijalankan dengan batasan strict NumPy!")

    [ignoring loop detection]
# ... (bagian awal file tetap sama)

    @staticmethod
    def merge_image(img1, img2, alpha=0.5):
        """
        Menggabungkan dua citra dengan teknik blending (transparansi) manual.
        Rumus: hasil = (img1 * alpha) + (img2 * (1 - alpha))
        
        PENTING: Ukuran img1 dan img2 harus sama. Jika berbeda, gunakan:
        >>> h, w = img1.shape[:2]
        >>> img2 = HistogramProcessor.resize_manual(img2, h, w)
        
        Args:
            img1: np.array citra pertama (source A)
            img2: np.array citra kedua (source B).
            alpha: float (0.0 - 1.0), bobot transparansi citra pertama. 
        Returns:
            np.array citra hasil blending (uint8)
        """
        # ... (logika merge tetap sama)

# ... (bagian tengah file)

# =============================================================================
#  CONTOH PENGGUNAAN LENGKAP (TUTORIAL IMPLEMENTASI)
# =============================================================================
# if __name__ == "__main__":
#     # 1. Simulasi dua gambar dengan ukuran berbeda
#     img_asli = np.random.randint(0, 255, (300, 400), dtype=np.uint8) # 300x400
#     img_logo = np.random.randint(0, 255, (100, 100), dtype=np.uint8) # 100x100 (Beda ukuran!)

#     print(f"Ukuran Asli: {img_asli.shape}")
#     print(f"Ukuran Logo: {img_logo.shape}")

#     # 2. CARA MENYAMAKAN UKURAN (Resize Logo ke Ukuran Asli)
#     h, w = img_asli.shape[:2]
#     logo_resized = HistogramProcessor.resize_manual(img_logo, h, w)
#     print(f"Ukuran Logo setelah Resize: {logo_resized.shape}")

#     # 3. Lakukan Merging
#     hasil_merge = HistogramProcessor.merge_image(img_asli, logo_resized, alpha=0.8)
    
#     # 4. Contoh Ganti Background
#     # Misal kita punya foto objek dengan background putih (255)
#     objek = np.full((100, 100), 255, dtype=np.uint8) # Full Putih
#     objek[30:70, 30:70] = 50 # Ada kotak abu-abu di tengah
    
#     background_pemandangan = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
    
#     # Ganti background putih (threshold 240) dengan pemandangan
#     hasil_komposisi = HistogramProcessor.replace_background(
#         objek, 
#         background_pemandangan, 
#         threshold=240, 
#         mode='bright'
#     )

#     print("Implementasi Berhasil!")

# Contoh Taruh Logo di Pojok Kiri Atas (0,0)
# h_logo, w_logo = 50, 50
# logo_kecil = hp.resize_manual(img_logo, h_logo, w_logo)

# # Ambil potongan (ROI) dari gambar asli seukuran logo
# roi = img_asli[0:h_logo, 0:w_logo]

# # Blend hanya di area ROI tersebut
# blended_roi = hp.merge_image(roi, logo_kecil, alpha=0.5)

# # Masukkan kembali ke gambar asli
# img_asli[0:h_logo, 0:w_logo] = blended_roi

# [ignoring loop detection]
# import cv2
# import numpy as np
# from histogram_operations import HistogramProcessor as hp

# # 1. Load Gambar
# # Gunakan gambar asli kamu atau dummy jika tidak ada
# img_asli = cv2.imread('input.jpg', 0) 
# if img_asli is None:
#     img_asli = np.random.randint(50, 150, (256, 256), dtype=np.uint8) # Gambar gelap

# # 2. PLOT HISTOGRAM
# print("Menampilkan Histogram Awal...")
# hp.plot_histogram(img_asli, "Histogram Original")

# # 3. HISTOGRAM EQUALIZATION (Pemerataan)
# # Mengubah gambar yang terlalu gelap/terang menjadi lebih seimbang kontrasnya
# img_equalized = hp.equalize(img_asli)
# hp.show_comparison(img_asli, img_equalized, "Sebelum EQ", "Setelah EQ")
# hp.plot_histogram(img_equalized, "Histogram Setelah Equalization")

# # 4. HISTOGRAM SPECIFICATION (Matching)
# # Menyamakan distribusi warna img_asli agar mirip dengan target_style
# target_style = np.random.randint(200, 255, (256, 256), dtype=np.uint8) # Target sangat terang
# img_matched = hp.histogram_specification(img_asli, target_style)

# hp.show_comparison(img_asli, img_matched, "Original", "Setelah Matching ke Target")

# # 5. HITUNG METRIK KUALITAS (MSE & PSNR)
# mse, psnr = hp.calculate_metrics(img_asli, img_equalized)
# print(f"Hasil Perbaikan - MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")



