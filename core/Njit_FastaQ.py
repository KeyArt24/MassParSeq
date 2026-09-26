
import os, sys

# Определяем путь к папке core, в которой лежит этот файл
_core_dir = os.path.dirname(os.path.abspath(__file__))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import time
import mmap
import numpy as np
import gc
from isal import igzip, igzip_threaded
from njit_main_opt import jit_common_func
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class Chunk_Result:
    count_a: int = 0
    count_c: int = 0
    count_g: int = 0
    count_t: int = 0
    count_n: int = 0
    count_reads: int = 0
    positions_reads: Optional[np.ndarray] = None
    count_nuc: int = 0
    min_len_read: int = 0
    max_len_read: int = 0
    distrub_nuc: Optional[np.ndarray] = None
    distrub_qual: Optional[np.ndarray] = None
    distrub_gc: Optional[np.ndarray] = None
    distrub_len: Optional[np.ndarray] = None
    len_r_arr: Optional[np.ndarray] = None
    distrub_qual_c: Optional[np.ndarray] = None
    ram: int = 0

    @classmethod
    def from_jit(cls, jit_result: tuple):
        """Создание из кортежа, возвращаемого JIT функцией"""
        return cls(*jit_result)

@dataclass
class Jit:

    @classmethod
    def warmup(cls):
        try:
            start_time = time.perf_counter()
            gc.collect()
            gc.freeze()
            gc.collect()
            print('Прогрев njit НАЧАТ')
            seq2 = b'@NHGHEJEHJ\nTAGCGATGAGAGCGAGGCAGGCAGGCGACGGAGG\n+$^@*&%*@*(&(@&^(&(@&($@)(*(*(((*(&*&*$&%(#%\n' \
                    b'@NHGHEJEHJ\nTAGCGATGAGAGCGAGGCAGGCAGGCGACGGAGG\n+$^@*&%*@*(&(@&^(&(@&($@)(*(*(((*(&*&*$&%(#%\n' \
                    b'@52342342234\nGTGCATGAGATATAGCGAGCTATATTATTATATATACGAGC\n+64728288781749.12\n&$**#&$&$&$&&$&$**$****$**$(*$&^&%^%^%&&*&*&)\n'
            warmup_data2 = np.frombuffer(seq2, dtype=np.uint8)
            jit_common_func(warmup_data2, 0, os.cpu_count())
            elapsed = time.perf_counter() - start_time
            print('Прогрев njit ЗАКОНЧЕН')
            print(f"Время            : {elapsed:.2f} секунд")
            print()
        except Exception as e:
            print(e)

class PYJITFASTQ():
    def __init__(self):
        
        self.OFFSETS: list = []
        self.DATA: dict = {}
        self.FILEPATH: str = ''
        self.FILESIZE: int = 0
        self.REPORT = Chunk_Result()

        self.status_callback = None

    def load_fastq(self, filepath):
        self.FILEPATH = filepath

    def _status(self, msg: str):
        cb = self.status_callback
        if cb is not None:
            cb(msg)

    def run(self, cancel = None):

        if self.FILEPATH.split('.')[-1] == 'fastq':
            self.OFFSETS = [] 
            self.offsets_positions()
            self.read_fastq(cancel)
        elif self.FILEPATH.split('.')[-1] == 'gz':
            self.read_fastq_gz(cancel)
        else:
            self._status('Не найден путь к файлу')

    # ПОЛУЧЕНИЕ ЧАНКОВ И СМЕЩЕНИЙ ПО ФАЙЛУ ------------------------------------------------------
    def offsets_positions(self):
        """
        Упрощенная версия с сохранением той же логики,
        но более надежная
        """
        # !!!!!!!!!!!!!!!!!!!!!!!!!! НУЖНА ПРОВЕРКА ПУСТЫХ БАЙТОВ И СТРУКТУРЫ ФАЙЛА !!!!!!!!!!!!!!!!!!!!!
        # Получени размера файла и размер чанка
        file_size = os.path.getsize(self.FILEPATH)
        chunk = 1024 * 1024 * 512  # 1GB
        # Создания списка смещений по файлу
        offsets_list = [0]

        with open(self.FILEPATH, 'rb') as f:
            for offset in range(0, file_size, chunk):
                current_chunk = min(chunk, file_size - offset)

                if current_chunk == 0:
                    continue

                with mmap.mmap(f.fileno(), current_chunk, access=mmap.ACCESS_READ, offset=offset) as mm:
                    # Ищем последний символ '@' в чанке
                    last_at_pos = -1

                    # Проверяем последние 4096 байт на наличие '@'
                    search_start = max(0, len(mm) - 4096)
                    for i in range(len(mm) - 1, search_start - 1, -1):
                        if mm[i] == 64:  # '@'
                            last_at_pos = i
                            break

                    if last_at_pos != -1:
                        position = offset + last_at_pos
                    else:
                        position = offset + current_chunk

                    # Выравнивание по гранулярности
                    granularity = mmap.ALLOCATIONGRANULARITY
                    position = (position // granularity) * granularity

                    if position > offset and position < file_size:
                        offsets_list.append(position)

            # Добавляем конец файла
            if offsets_list[-1] != file_size:
                offsets_list.append(file_size)

            self.OFFSETS = sorted(set(offsets_list))

    # ЧТЕНИЕ ФАЙЛА ------------------------------------------------------------------------------
    def read_fastq(self, cancel):
        file_size = os.path.getsize(self.FILEPATH)
        

        try:
            with open(self.FILEPATH, 'rb') as f:
                shift_offset = 0
                start_position = 0

                for idx in range(len(self.OFFSETS)-1):
                    if cancel and cancel():
                        self._status(f'Анализ отменён')
                        return

                    self._status(f'Процесс... Обработанно {self.REPORT.count_reads} ридов, {self.REPORT.ram/1024**3:.2f} ГБ памяти RAM на массивы')
                    start_offset = self.OFFSETS[idx] - shift_offset
                    end_offset = self.OFFSETS[idx+1]
                    current_chunk = end_offset - start_offset
                    with mmap.mmap(f.fileno(), current_chunk, access=mmap.ACCESS_READ, offset = start_offset) as mm:

                        # Поиск последнего рида с конца чанка
                        if end_offset == file_size:
                            end_position = current_chunk
                        else:
                            end_position = mm.rfind(b'\n@')

                        # Смещение offset в лево кратно 65536 для сохранения целостности рида
                        shift_offset = mmap.ALLOCATIONGRANULARITY

                   
                        # Получение данных и обработка
                        try:
                            array = np.frombuffer(mm, dtype = np.uint8)[start_position:end_position]
                            chunk_data = jit_common_func(array, start_offset, os.cpu_count())
                        except Exception as e:
                            self._status('Ошибка выполнения njit функции')
                            #print(e)
                        finally:
                            # Удаление memoryview для закрытия mmap
                            del array
                    
                        # Получение стартовой позиции с последнего рида предыдущего чанка
                        start_position = (end_position + start_offset) - (end_offset - shift_offset - 1)

                        result = Chunk_Result(*chunk_data)
                        self.accumulation_data(result)

                        # Вывод данных
                        #print(f'Обработанно {self.REPORT.count_reads} ридов, {self.REPORT.ram/1024**3:.2f} ГБ памяти RAM на массивы', flush = True)

            #print(f'\nНуклеотидов {self.RESULT['Count Nucleotides']}')
            #print(f'Максимальная длина рида {self.RESULT['Max Length Read']}')
            #print(f'Количество чанков {len(self.RESULT['Positions Reads'])}')

        except Exception as ex:
            print(ex)

    def read_fastq_gz(self, cancel):
        chunk = 1024 * 1024 * 102
        buffer = np.zeros(chunk, dtype=np.uint8) # Запас 1МБ для длинных ридов
        offset = 0
        with igzip.open(self.FILEPATH, "rb") as f:
            while True:
                if cancel and cancel():
                    self._status(f'Анализ отменён')
                    return
                self._status(f'Процесс... Обработанно {self.REPORT.count_reads} ридов, {self.REPORT.ram/1024**3:.2f} ГБ памяти RAM на массивы')
                chunk_data = None 
                # Читаем данные в буфер после остатка с прошлого раза
                bytes_read = f.readinto(buffer[offset:])
                if bytes_read == 0 and offset == 0:
                    break
            
                total_bytes = offset + bytes_read
            
                # Ищем границу последнего полного рида (с конца буфера)
                # В FastQ рид заканчивается перед новой строкой с '@'
                # Нужно найти последний '\n', после которого идет '@' (ASCII 64)
                # или просто последний '\n', если вы парсите по состояниям
                last_newline = -1
                for i in range(total_bytes - 1, total_bytes - 2000, -1): # Ищем в хвосте
                    if buffer[i-1] == 10 and buffer[i] == 64: # \n
                        last_newline = i-1
                        break
                        
            
                if bytes_read == 0: # Конец файла
                    end_pos = total_bytes
                else:
                    end_pos = last_newline + 1

                # --- ВАША NJIT ФУНКЦИЯ ---
                active_view = buffer[:end_pos]

                # positions_reads здесь тоже должны учитывать смещение внутри этого вью
                # Получение данных и обработка
                try:
                    chunk_data = jit_common_func(active_view, 0, os.cpu_count())
                except Exception as e:
                    self._status('Ошибка выполнения njit функции')
                    #print(e)   
                    
                if chunk_data is not None:
                    result = Chunk_Result(*chunk_data)
                    self.accumulation_data(result)

                # Вывод данных
                #print(f'Обработанно {self.REPORT.count_reads} ридов, {self.REPORT.ram/1024**3:.2f} ГБ памяти RAM на массивы', end = '\r')
                # -------------------------

                # Переносим остаток в начало для следующей итерации
                remainder = total_bytes - end_pos
                if remainder > 0:
                    buffer[:remainder] = buffer[end_pos:total_bytes]
                    offset = remainder
                else:
                    offset = 0
            
                if bytes_read == 0: break

        #print(f'\nНуклеотидов {self.RESULT['Count Nucleotides']}')
        #print(f'Максимальная длина рида {self.RESULT['Max Length Read']}')
        #print(f'Количество чанков {len(self.RESULT['Positions Reads'])}')

    # СОЗДАНИЕ ОТЧЁТА ---------------------------------------------------------------------------
    def print_report(self, elapsed = 0.0, file_size = 0):
        
        print()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'{self.FILEPATH}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f"  A (Аденин)      : {self.REPORT.count_a:>15,}")
        print(f"  C (Цитозин)     : {self.REPORT.count_c:>15,}")
        print(f"  G (Гуанин)      : {self.REPORT.count_g:>15,}")
        print(f"  T (Тимин)       : {self.REPORT.count_t:>15,}")
        print(f"  N (Неизвестный) : {self.REPORT.count_n:>15,}")
        print(f"  ─────────────────────────────────")
        print(f"  GC% : {(self.REPORT.count_c + self.REPORT.count_g) / self.REPORT.count_nuc * 100:.2f}")
        print(f"  Всего ридов : {self.REPORT.count_reads}")
        print(f"  Минимальная длина рида : {self.REPORT.min_len_read}")
        print(f"  Максимальная длина рида : {self.REPORT.max_len_read}")
        print(f"  Всего оснований : {self.REPORT.count_nuc:>15,}")

        print()

        print(f"  Количество ОЗУ для построения массивов данных numpy : {self.REPORT.ram / (1024 ** 3):.2f} ГБ")
        print(f"  Производительность : {(self.REPORT.count_nuc + self.REPORT.count_n) / elapsed / (100 ** 3):.2f} МБ оснований/с")
        print(f"  Размер файла     : {file_size / (1024 ** 3):.2f} ГБ")
        print(f"  Время            : {elapsed:.2f} секунд")
        print()
        print('*********************************** END REPORT ************************************')
        print()
    
    # АККАМУЛИРОВАНИЕ ДАННЫХ---------------------------------------------------------------------
    def accumulation_data(self, result: Chunk_Result):
   
        # Аккамулирование результатов
        self.REPORT.count_reads += result.count_reads
        self.REPORT.count_nuc += result.count_nuc
        self.REPORT.min_len_read = result.min_len_read if self.REPORT.min_len_read == 0 else min(self.REPORT.min_len_read, result.min_len_read)
        self.REPORT.max_len_read = max(self.REPORT.max_len_read, result.max_len_read)
        self.REPORT.ram += result.ram
        self.REPORT.count_a += result.count_a
        self.REPORT.count_c += result.count_c
        self.REPORT.count_g += result.count_g
        self.REPORT.count_t += result.count_t
        self.REPORT.count_n += result.count_n

        if isinstance(self.REPORT.positions_reads, np.ndarray):
            if result.positions_reads.shape[0] > self.REPORT.positions_reads.shape[0]:
                new_array = np.zeros(result.positions_reads.shape, dtype=np.uint32)
                new_array[:self.REPORT.positions_reads.shape[0]] = self.REPORT.positions_reads
                self.REPORT.positions_reads = new_array + result.positions_reads
            else:
                self.REPORT.positions_reads[:result.positions_reads.shape[0]] += result.positions_reads
        else:
            self.REPORT.positions_reads = result.positions_reads

        if isinstance(self.REPORT.distrub_nuc, np.ndarray):
            if result.distrub_nuc.shape[0] > self.REPORT.distrub_nuc.shape[0]:
                new_array = np.zeros(result.distrub_nuc.shape, dtype=np.uint32)
                new_array[:self.REPORT.distrub_nuc.shape[0], :] = self.REPORT.distrub_nuc
                self.REPORT.distrub_nuc = new_array + result.distrub_nuc
            else:
                self.REPORT.distrub_nuc[:result.distrub_nuc.shape[0], :] += result.distrub_nuc 
        else:
            self.REPORT.distrub_nuc = result.distrub_nuc 

        if isinstance(self.REPORT.distrub_qual, np.ndarray):
            if result.distrub_qual.shape[0] > self.REPORT.distrub_qual.shape[0]:
                new_array = np.zeros(result.distrub_qual.shape, dtype=np.uint32)
                new_array[:self.REPORT.distrub_qual.shape[0], :] = self.REPORT.distrub_qual
                self.REPORT.distrub_qual = new_array + result.distrub_qual
            else:
                self.REPORT.distrub_qual[:result.distrub_qual.shape[0], :] += result.distrub_qual
        else:
            self.REPORT.distrub_qual = result.distrub_qual


        if isinstance(self.REPORT.distrub_gc, np.ndarray):
            if result.distrub_gc.shape[0] > self.REPORT.distrub_gc.shape[0]:
                new_array = np.zeros(result.distrub_gc.shape, dtype=np.uint32)
                new_array[:self.REPORT.distrub_gc.shape[0]] = self.REPORT.distrub_gc
                self.REPORT.distrub_gc = new_array + result.distrub_gc
            else:
                self.REPORT.distrub_gc[:result.distrub_gc.shape[0]] += result.distrub_gc
        else:
            self.REPORT.distrub_gc = result.distrub_gc


        if isinstance(self.REPORT.distrub_len, np.ndarray):
            if result.distrub_len.shape[0] > self.REPORT.distrub_len.shape[0]:
                new_array = np.zeros(result.distrub_len.shape, dtype=np.uint32)
                new_array[:self.REPORT.distrub_len.shape[0]] = self.REPORT.distrub_len
                self.REPORT.distrub_len = new_array + result.distrub_len
            else:
                self.REPORT.distrub_len[:result.distrub_len.shape[0]] += result.distrub_len
        else:
            self.REPORT.distrub_len = result.distrub_len

        if isinstance(self.REPORT.distrub_qual_c, np.ndarray):
            if result.distrub_qual_c.shape[0] > self.REPORT.distrub_qual_c.shape[0]:
                new_array = np.zeros(result.distrub_qual_c.shape, dtype=np.uint32)
                new_array[:self.REPORT.distrub_qual_c.shape[0]] = self.REPORT.distrub_qual_c
                self.REPORT.distrub_qual_c = new_array + result.distrub_qual_c
            else:
                self.REPORT.distrub_qual_c[:result.distrub_qual_c.shape[0]] += result.distrub_qual_c
        else:
            self.REPORT.distrub_qual_c = result.distrub_qual_c

    # СТАТИСТИКА ---------------------------------------------------------------------------    
    def calculate_per_base_stats_fast(self, qual_matrix):
        """
        Оптимизированный расчет статистик с использованием весов.
        """
        n_positions = qual_matrix.shape[0]
        n_qual_levels = qual_matrix.shape[1]

        stats = {
            'mean': np.zeros(n_positions),
            'median': np.zeros(n_positions),
            'q25': np.zeros(n_positions),
            'q75': np.zeros(n_positions),
            'q10': np.zeros(n_positions),
            'q90': np.zeros(n_positions)
        }

        # Предварительно вычисляем cumulative суммы
        quality_levels = np.arange(n_qual_levels)

        for pos in range(n_positions):
            counts = qual_matrix[pos]
            total = np.sum(counts)

            if total == 0:
                continue

            # Среднее (взвешенное)
            stats['mean'][pos] = np.sum(quality_levels * counts) / total

            # Cumulative сумма для поиска процентилей
            cumsum = np.cumsum(counts)

            # Медиана (50-й процентиль)
            median_idx = np.searchsorted(cumsum, total * 0.5)
            stats['median'][pos] = median_idx

            # 25-й процентиль
            q25_idx = np.searchsorted(cumsum, total * 0.25)
            stats['q25'][pos] = q25_idx

            # 75-й процентиль
            q75_idx = np.searchsorted(cumsum, total * 0.75)
            stats['q75'][pos] = q75_idx

            # 10-й процентиль
            q10_idx = np.searchsorted(cumsum, total * 0.10)
            stats['q10'][pos] = q10_idx

            # 90-й процентиль
            q90_idx = np.searchsorted(cumsum, total * 0.90)
            stats['q90'][pos] = q90_idx

        return stats

def main():
    parser = argparse.ArgumentParser(
        description='PYJITFASTQ - High-performance FASTQ quality control analyzer',
        epilog='Example: python Njit_FastaQ.py -i sample.fastq -o report.png -t 8'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input FASTQ file (.fastq or .fastq.gz)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output report filename (default: input_file.report.png)'
    )
    
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=None,
        help='Number of threads (default: CPU count)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Show plots without saving'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='PYJITFASTQ 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    
    # Установка количества потоков
    if args.threads:
        cpu_count = args.threads
    else:
        cpu_count = os.cpu_count()
    
    # Прогрев Numba
    Jit.warmup()
    
    # Запуск анализа
    start_time = time.perf_counter()
    analyzer = PYJITFASTQ()
    analyzer.load_fastq(args.input)
    analyzer.run()
    elapsed = time.perf_counter() - start_time
    
    file_size = os.path.getsize(args.input)
    
    # Вывод отчёта
    analyzer.print_report(result=None, elapsed=elapsed, file_size=file_size)
    
    # Визуализация
    tag = 'save' if not args.no_save else 'show'
    analyzer.visual_dashbord(tag=tag)
    
    print(f"\nInput file: {args.input}")
    print(f"Threads used: {cpu_count}")
    print(f"Time: {elapsed:.2f} seconds")

def testing():
    files = [r"C:\Users\Stupnikova\Downloads\SRR37931586.fastq\SRR18209649.fastq.gz"]

    jit = Jit()
    jit.warmup()

    total_time = 0
    tolat_size = 0
    for file in files:
        start_time = time.perf_counter()
        analyzer = PYJITFASTQ()
        analyzer.load_fastq(file)
        analyzer.run()
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        tolat_size += os.path.getsize(file)
        analyzer.print_report(elapsed = elapsed, file_size = os.path.getsize(file))

    print(f'Всего времени на обработку файлов {total_time / 60} минут')
    print(f"  Размер файлов     : {tolat_size / (1024 ** 3):.2f} ГБ")

if __name__ == "__main__":
    testing()
