
from numba import njit, prange, get_num_threads
from numba import np as numbanp
import numpy as np
import time


# ОСНОВНАЯ ФУНКЦИЯ **************************************************************************
@njit(fastmath=True)
def jit_common_func(array: np.ndarray, offset: int, cpu_count: int):
    
    SIZE = len(array)
    NUM_CORES = cpu_count
    
    # Параметры разбивки
    if SIZE >= 1024 * 1024:
        NUM_SEGMENTS = max(NUM_CORES * 4, 4)
        SEGMENT_SIZE = (SIZE + NUM_SEGMENTS - 1) // NUM_SEGMENTS
    else:
        NUM_SEGMENTS = 1
        SEGMENT_SIZE = SIZE
    
    # Lookup таблица
    LOOKUP = np.zeros(256, dtype=np.uint8)
    LOOKUP[65] = 1  # 'A'
    LOOKUP[97] = 1  # 'a'
    LOOKUP[67] = 2  # 'C'
    LOOKUP[99] = 2  # 'c'
    LOOKUP[71] = 3  # 'G'
    LOOKUP[103] = 3  # 'g'
    LOOKUP[84] = 4  # 'T'
    LOOKUP[116] = 4  # 't'
    LOOKUP[78] = 5  # 'N'
    LOOKUP[110] = 5  # 'n'
    
    # ========== ФУНКЦИЯ 1: ПОДСЧЁТ РИДОВ ==========
    cr = jit_count_reads(array, SIZE, NUM_SEGMENTS, SEGMENT_SIZE)
    
    # ========== ФУНКЦИЯ 2: ПОЗИЦИИ РИДОВ ==========
    POSITIONS_READS = jit_fill_positions(array, SIZE, NUM_SEGMENTS, SEGMENT_SIZE, cr)
    COUNT_READS = len(POSITIONS_READS)
    
    # ========== ФУНКЦИЯ 3: МАКСИМАЛЬНАЯ ДЛИНА ==========
    MAX_LEN_READ = jit_find_max_read_length(array, POSITIONS_READS)
    
    # ========== ПОДГОТОВКА МАССИВОВ ДЛЯ СТАТИСТИКИ ==========
    n_threads = get_num_threads()
    
    # Массивы для каждого рида
    a_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    c_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    g_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    t_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    n_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    gc_arr = np.zeros(COUNT_READS, dtype=np.uint8)
    len_r_arr = np.zeros(COUNT_READS, dtype=np.uint16)
    qual_mean_r_arr = np.zeros(COUNT_READS, dtype=np.uint8)
    
    # Матрицы распределений для каждого потока
    tid_nuc_distrub_arr = np.zeros((n_threads, MAX_LEN_READ, 5), dtype=np.uint32)
    tid_qual_distrub_arr = np.zeros((n_threads, MAX_LEN_READ, 100), dtype=np.uint32)
    tid_qual_max = np.zeros(n_threads, dtype=np.uint8)
    
    # Глобальные массивы распределений
    gc_distrub_arr = np.zeros(100, dtype=np.uint64)
    len_distrub_arr = np.zeros(MAX_LEN_READ, dtype=np.int64)
    nuc_distrub_arr = np.zeros((MAX_LEN_READ, 5), dtype=np.uint32)
    qual_distrub_arr = np.zeros((MAX_LEN_READ, 100), dtype=np.uint32)
    
    # ========== ФУНКЦИЯ 4: СТАТИСТИКА ==========
    result = jit_stat_reads(
        array=array,
        positions_reads=POSITIONS_READS,
        max_len_read=MAX_LEN_READ,
        lookup=LOOKUP,
        a_arr=a_arr,
        c_arr=c_arr,
        g_arr=g_arr,
        t_arr=t_arr,
        n_arr=n_arr,
        gc_arr=gc_arr,
        len_r_arr=len_r_arr,
        qual_mean_r_arr=qual_mean_r_arr,
        tid_nuc_distrub_arr=tid_nuc_distrub_arr,
        tid_qual_distrub_arr=tid_qual_distrub_arr,
        tid_qual_max=tid_qual_max,
        gc_distrub_arr=gc_distrub_arr,
        len_distrub_arr=len_distrub_arr,
        nuc_distrub_arr=nuc_distrub_arr,
        qual_distrub_arr=qual_distrub_arr
    )
    
    # Распаковка результатов
    (a_arr, c_arr, g_arr, t_arr, n_arr, 
     nuc_distrub_arr, qual_distrub_arr, gc_distrub_arr, 
     len_distrub_arr, len_r_arr, qual_distrub_count_r_arr) = result
    
    SUM_A = a_arr.sum()
    SUM_C = c_arr.sum()
    SUM_G = g_arr.sum()
    SUM_T = t_arr.sum()
    SUM_N = n_arr.sum()
    NUC_SUM = SUM_A + SUM_C + SUM_G + SUM_T + SUM_N
    MIN_LEN_READ = len_r_arr.min()
    
    BYTES_RAM = (a_arr.nbytes + c_arr.nbytes + g_arr.nbytes + t_arr.nbytes + n_arr.nbytes +
                 nuc_distrub_arr.nbytes + qual_distrub_arr.nbytes + gc_distrub_arr.nbytes +
                 len_distrub_arr.nbytes + len_r_arr.nbytes + POSITIONS_READS.nbytes)
 
    return (SUM_A, SUM_C, SUM_G, SUM_T, SUM_N, COUNT_READS, POSITIONS_READS, NUC_SUM, MIN_LEN_READ, MAX_LEN_READ,
            nuc_distrub_arr, qual_distrub_arr, gc_distrub_arr, len_distrub_arr, len_r_arr, 
            qual_distrub_count_r_arr, BYTES_RAM)
#---------------------------------------------------------------------------------------------------------------------------------

# ТОЛЬКО ПОДСЧЁТ РИДОВ *******************************************************************
@njit(parallel=True, fastmath=True, error_model='numpy')
def jit_count_reads(array: np.ndarray, size: int, num_segments: int, seg_size: int) -> np.ndarray:
    """
    Подсчёт количества ридов в каждом сегменте.
    
    Returns
    -------
    cr : np.ndarray
        Массив с количеством ридов в каждом сегменте
    """
    cr = np.zeros(num_segments, dtype=np.uint32)
    
    for seg in prange(num_segments):
        start = seg * seg_size
        end = min(start + seg_size, size)
        count = 0
        
        # Обрабатываем первую позицию сегмента
        if start == 0:
            if array[0] == 64:
                count += 1
        else:
            if array[start] == 64 and array[start-1] == 10:
                count += 1
        
        # Считаем остальные '@'
        for i in range(start + 1, end):
            if array[i] == 64 and array[i-1] == 10:
                count += 1
        
        cr[seg] = count
    
    return cr

# ЗАПОЛНЕНИЕ МАССИВА ПОЗИЦИЙ *******************************************************************
@njit(parallel=True, fastmath=True, error_model='numpy')
def jit_fill_positions(array: np.ndarray, size: int, num_segments: int, seg_size: int, cr: np.ndarray) -> np.ndarray:
    """
    Заполнение массива позиций ридов на основе сегментов.
    
    Returns
    -------
    position_reads : np.ndarray
        Массив позиций всех ридов
    """
    # Вычисляем смещения для каждого сегмента
    offsets = np.zeros(num_segments + 1, dtype=np.int64)
    for i in range(num_segments):
        offsets[i + 1] = offsets[i] + cr[i]
    
    count_reads = offsets[-1]
    position_reads = np.zeros(count_reads, dtype=np.int64)
    
    for seg_idx in prange(num_segments):
        start = seg_idx * seg_size
        end = min(start + seg_size, size)
        local_offset = offsets[seg_idx]
        idx = 0
        
        # Первая позиция сегмента
        if start == 0 and array[0] == 64:
            position_reads[local_offset + idx] = start
            idx += 1
        else:
            if array[start] == 64 and array[start-1] == 10:
                position_reads[local_offset + idx] = start
                idx += 1
        
        # Остальные позиции
        for i in range(start + 1, end):
            if array[i] == 64 and array[i-1] == 10:
                position_reads[local_offset + idx] = i
                idx += 1
    
    return position_reads

# ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ РИДА *******************************************************************
@njit(fastmath=True, error_model='numpy')
def jit_find_max_read_length(array: np.ndarray, position_reads: np.ndarray) -> int:
    """
    Поиск максимальной длины рида по массиву позиций.
    
    Returns
    -------
    max_len_read : int
        Максимальная длина последовательности среди всех ридов
    """
    count_reads = len(position_reads)
    if count_reads == 0:
        return 0
    
    # Находим кандидата с максимальным расстоянием между началами ридов
    max_dist = 0
    idx_max = 0
    prev = position_reads[0]
    
    for i in range(1, count_reads):
        dist = position_reads[i] - prev
        if dist > max_dist:
            max_dist = dist
            idx_max = i - 1
        prev = position_reads[i]
    
    # Проверяем последний рид
    last_dist = len(array) - position_reads[-1]
    if last_dist > max_dist:
        idx_max = count_reads - 1
    
    # Определяем границы кандидата
    if idx_max < count_reads - 1:
        start_pos = position_reads[idx_max]
        end_pos = position_reads[idx_max + 1]
    else:
        start_pos = position_reads[-1]
        end_pos = len(array)
    
    # Точный подсчёт длины последовательности
    max_len = 0
    local_state = 0
    
    for i in range(start_pos, end_pos):
        base = array[i]
        if base == 10:  # '\n'
            local_state += 1
        elif local_state == 1:
            max_len += 1
        elif base == 43 and i > 0 and array[i-1] == 10:  # '+' после '\n'
            break
    
    return max_len

#---------------------------------------------------------------------------------------------------------------------------------

# ОБРАБОТКА ДАННЫХ В РИДЕ *******************************************************************
@njit(parallel=True, fastmath=True, error_model='numpy')
def jit_stat_reads(
    array: np.ndarray,
    positions_reads: np.ndarray,
    max_len_read: int,
    lookup: np.ndarray,
    a_arr: np.ndarray,
    c_arr: np.ndarray,
    g_arr: np.ndarray,
    t_arr: np.ndarray,
    n_arr: np.ndarray,
    gc_arr: np.ndarray,
    len_r_arr: np.ndarray,
    qual_mean_r_arr: np.ndarray,
    tid_nuc_distrub_arr: np.ndarray,
    tid_qual_distrub_arr: np.ndarray,
    tid_qual_max: np.ndarray,
    gc_distrub_arr: np.ndarray,
    len_distrub_arr: np.ndarray,
    nuc_distrub_arr: np.ndarray,
    qual_distrub_arr: np.ndarray
):
    """
    Обработка данных в ридах. Все выходные массивы передаются как параметры.
    """
    count_reads = positions_reads.size
    end_array = array.size
    n_threads = tid_nuc_distrub_arr.shape[0]
    
    # ПРОХОД КАЖДОГО РИДА В МАССИВЕ ПОЗИЦИЙ РИДА ДЛЯ ПОДСЧЁТА
    for pos_idx in prange(count_reads):
        # Получение id процесса
        tid = numbanp.ufunc.parallel._get_thread_id()
        
        pos = positions_reads[pos_idx]
        
        if pos_idx + 1 < count_reads:
            end = positions_reads[pos_idx + 1]
        else:
            end = end_array
        
        local_state = 0
        pos_in_line = 0
        
        la = np.uint32(0)
        lc = np.uint32(0)
        lg = np.uint32(0)
        lt = np.uint32(0)
        ln = np.uint32(0)
        lx = np.uint32(0)
        total_q_r = 0
        
        i = pos
        while i < end:
            base = array[i]
            
            if base == 10:
                local_state += 1
                pos_in_line = -1
            
            if local_state == 1 and base != 10:
                la += (base == 65)
                lc += (base == 67)
                lg += (base == 71)
                lt += (base == 84)
                ln += (base == 78)
                if pos_in_line >= 0 and pos_in_line < max_len_read:
                    if lookup[base] > 0:
                        tid_nuc_distrub_arr[tid, pos_in_line, lookup[base] - 1] += 1
                    else:
                        lx += 1
            
            if base == 43 and i > 0 and array[i-1] == 10:
                local_state = 2
            
            if local_state == 3 and base != 10:
                q = base - 33
                total_q_r += q
                if pos_in_line >= 0 and pos_in_line < max_len_read and 0 <= q < 100:
                    tid_qual_distrub_arr[tid, pos_in_line, q] += 1
            
            i += 1
            pos_in_line += 1
        
        len_read = la + lc + lg + lt + ln
        len_r_arr[pos_idx] = len_read
        
        gc_val = (lc + lg) * 100 // len_read if len_read > 0 else 0
        gc_arr[pos_idx] = gc_val if gc_val < 100 else 99
        
        mean_qual_r = total_q_r // len_read if len_read > 0 else 0
        qual_mean_r_arr[pos_idx] = mean_qual_r
        
        if mean_qual_r > tid_qual_max[tid]:
            tid_qual_max[tid] = mean_qual_r
        
        a_arr[pos_idx] = la
        c_arr[pos_idx] = lc
        g_arr[pos_idx] = lg
        t_arr[pos_idx] = lt
        n_arr[pos_idx] = ln
    
    # ЗАПИСЬ ДАННЫХ В ГЛОБАЛЬНЫЕ МАССИВЫ
    for tid in range(n_threads):
        for pos_in_line in range(max_len_read):
            for base in range(5):
                nuc_distrub_arr[pos_in_line, base] += tid_nuc_distrub_arr[tid, pos_in_line, base]
            for qual in range(100):
                qual_distrub_arr[pos_in_line, qual] += tid_qual_distrub_arr[tid, pos_in_line, qual]
    
    # Подсчёт распределений
    max_quality = np.max(tid_qual_max)
    qual_distrub_count_r_arr = np.zeros(max_quality + 1, dtype=np.uint32)
    
    for idx in range(count_reads):
        gc_val = gc_arr[idx]
        if gc_val < 100:
            gc_distrub_arr[gc_val] += 1
        len_val = len_r_arr[idx]
        if len_val > 0 and len_val <= max_len_read:
            len_distrub_arr[len_val] += 1
        qual_distrub_count_r_arr[qual_mean_r_arr[idx]] += 1
    
    return (a_arr, c_arr, g_arr, t_arr, n_arr, nuc_distrub_arr, qual_distrub_arr,
            gc_distrub_arr, len_distrub_arr, len_r_arr, qual_distrub_count_r_arr)

# --------------- ТЕСТИРОВАНИЕ --------------
def probe():
    start_time = time.perf_counter()
    print('Прогрев njit НАЧАТ')
    seq2 = b'@NHGHEJEHJ\nTAGCGATGAGAGCGAGGCAGGCAGGCGACGGAGG\n+NFHHFJDHJD\n$^@*&%*@*(&(@&^(&(@&($@)(*(*(((*(&\n' \
            b'@NHGHEJEHJ\nTAGCGATGAGAGCGAGGCAGGCAGGCGACGGAGG\n+ATAYTYAAYAA\n$^@*&%*@*(&(@&^(&(@&($@)(*(*(((*(&\n' \
            b'@52342342234\nGTGCATGAGATATAGCGAGCTATATTAGCATCGAGCAATTATATATACGAGC\n+64728288781749.12\n&$**#&$&$&$&&$&$**$****$**$(*$&^&%^%^%&&*&&&????&*&)\n'
    warmup_data2 = np.frombuffer(seq2, dtype=np.uint8)
    jit_common_func(warmup_data2, 0, 8)
    elapsed = time.perf_counter() - start_time
    print(f'Прогрев njit ЗАКОНЧЕН {elapsed:.2f}')
