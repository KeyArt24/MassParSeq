
import os, sys

# Определяем путь к папке core, в которой лежит этот файл
_core_dir = os.path.dirname(os.path.abspath(__file__))
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

import time
import mmap
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import gc
from isal import igzip
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

class BasePlotter:
    """Базовый класс для настройки общих параметров графиков"""
    
    @staticmethod
    def setup_axes(ax, title, xlabel, ylabel, xlim=None, ylim=None, enable_grid=True):
        """Общая настройка осей"""
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        
        if enable_grid:
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')
            ax.minorticks_on()
    
    @staticmethod
    def add_legend(ax, loc='lower left', bbox_to_anchor=(1, 0)):
        """Добавление легенды с единым стилем"""
        ax.legend(loc=loc, fontsize='small', handlelength=1.5, bbox_to_anchor=bbox_to_anchor)

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

    # ВИЗУАЛИЗАЦИЯ ---------------------------------------------------------------------------
    def visual_distrub_qual(self, ax: plt.axes = None):
        """График качества по позициям"""
        stat = self.calculate_per_base_stats_fast(self.REPORT.distrub_qual)
        max_len = self.REPORT.max_len_read
        positions = range(1, min(len(stat['mean']), max_len) + 1)
    
        # Зоны качества
        ax.axhspan(28, 40, alpha=0.2, color='green', label='Хорошее качество (>28)')
        ax.axhspan(20, 28, alpha=0.2, color='yellow', label='Среднее качество (20-28)')
        ax.axhspan(0, 20, alpha=0.2, color='red', label='Плохое качество (<20)')
    
        # Статистические линии
        ax.fill_between(positions, stat['q25'][:max_len], stat['q75'][:max_len],
                        alpha=0.3, color='blue', label='25-75%')
        ax.plot(positions, stat['q90'][:max_len], 'b--', alpha=0.5, linewidth=0.8, label='10-90%')
        ax.plot(positions, stat['q10'][:max_len], 'b--', alpha=0.5, linewidth=0.8)
        ax.plot(positions, stat['median'][:max_len], 'r-', linewidth=2, label='Медиана')
        ax.plot(positions, stat['mean'][:max_len], 'b-', linewidth=1, alpha=0.7, label='Среднее')
    
        # Настройка осей
        ax.set_xlabel('Позиция в риде (bp)', fontsize=8)
        ax.set_ylabel('Качество (Phred score)', fontsize=8)
        ax.set_title('Per Base Sequence Quality', fontsize=8)
        ax.set_xlim(1, max_len)
        ax.set_ylim(0, 42)
    
        # Настройка сетки ПОЗИЦИЙ (каждый bp)
        ax.set_xticks(range(1, max_len + 1))           # метки на каждой позиции
        ax.set_xticks(range(1, max_len + 1), minor=True)  # minor тики на каждой позиции
        ax.tick_params(axis='x', labelsize=8) 
    
        # Основная сетка (для major тиков, которые можно разрежить)
        major_step = max(1, max_len // 20)  # примерно 20 меток
        major_ticks = range(1, max_len + 1, major_step)
        ax.set_xticks(major_ticks)  # только каждые N позиций
        ax.set_xticks(range(1, max_len + 1), minor=True)  # все позиции как minor
    
        # Включаем сетку
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='lightgray', alpha=0.5)
    
        # Легенда
        ax.legend(loc='lower left', fontsize='small', handlelength=1.5, bbox_to_anchor=(1, 0))

    def visual_distrub_qual_count(self, ax: plt.axes = None):
        """Гистограмма распределения среднего качества ридов"""
        x = range(self.REPORT.distrub_qual_c.size)
        y = self.REPORT.distrub_qual_c
        ax.bar(x, y)
        
        BasePlotter.setup_axes(ax, 'Distribution of Mean Read Quality', 
                               'Среднее качество рида (Phred)', 'Количество ридов')
    
    def visual_distrub_gc(self, ax: plt.axes = None):
        """Гистограмма распределения GC%"""
        x = range(1, 101)
        y = self.REPORT.distrub_gc
        ax.bar(x, y)
        
        BasePlotter.setup_axes(ax, 'Distribution of GC Content', '% GC', 'Количество ридов')
    
    def visual_distrub_len(self, ax: plt.axes = None):
        """Гистограмма распределения длин ридов"""
        x = range(1, self.REPORT.max_len_read + 1)
        y = self.REPORT.distrub_len
        ax.bar(x, y)
        
        BasePlotter.setup_axes(ax, 'Distribution of Read Lengths', 'Длина рида (bp)', 'Количество ридов')
    
    def visual_distrub_nuc(self, ax: plt.axes = None):
        """График нуклеотидного состава по позициям"""
        distrub = self.REPORT.distrub_nuc
        # Нормализация по строкам (в процентах)
        row_sums = distrub.sum(axis=1, keepdims=True)
        distrub_norm = distrub / row_sums * 100
        
        positions = range(1, len(distrub_norm) + 1)
        
        colors = {'A': 'green', 'C': 'blue', 'G': 'black', 'T': 'red', 'N': 'gray'}
        styles = {'A': '-', 'C': '-', 'G': '-', 'T': '-', 'N': '--'}
        
        for i, (base, color) in enumerate(colors.items()):
            ax.plot(positions, distrub_norm[:, i], label=base, color=color, 
                    lw=1, linestyle=styles[base])
        
        BasePlotter.setup_axes(ax, 'Per Base Sequence Content', 'Position in read (bp)', 
                               'Percentage (%)', ylim=(0, 100))
        BasePlotter.add_legend(ax)
    
    def visual_dashbord(self, tag = 'show'):
        """Создание мозаичного отчёта"""


        fig = Figure(figsize = (12, 5))

        axd = fig.subplot_mosaic([['A', 'A'], ['B', 'C'], ['D', 'E']])
        
        self.visual_distrub_qual(axd['A'])
        self.visual_distrub_gc(axd['B'])
        self.visual_distrub_len(axd['C'])
        self.visual_distrub_qual_count(axd['D'])
        self.visual_distrub_nuc(axd['E'])
        
        fig.tight_layout()
        
        if tag == 'show':
            return fig
        elif tag == 'save':
            name = self.FILEPATH.split('\\')[-1] + '.report.png'
            fig.savefig(name, dpi=600, bbox_inches='tight')
            print(f'Результаты сохранены в {name}')

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
        chunk = 1024 * 1024 * 512
        buffer = np.zeros(chunk, dtype=np.uint8) # Запас 1МБ для длинных ридов
        offset = 0
        with igzip.open(self.FILEPATH, "rb") as f:
            while True:
                if cancel and cancel():
                    self._status(f'Анализ отменён')
                    return
                self._status(f'Процесс... Обработанно {self.REPORT.count_reads} ридов, {self.REPORT.ram/1024**3:.2f} ГБ памяти RAM на массивы')
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
    files = []

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
        analyzer.visual_dashbord(tag = 'save')



    print(f'Всего времени на обработку файлов {total_time / 60} минут')
    print(f"  Размер файлов     : {tolat_size / (1024 ** 3):.2f} ГБ")

if __name__ == "__main__":
    testing()
