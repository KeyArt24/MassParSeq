
import sys
import pyqtgraph as pg
import numpy as np
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication



class PlotPyQtGraph(pg.GraphicsLayoutWidget):
    def __init__(self, data=None):
        super().__init__()

        self.REPORT = data

        # 1. Создаем сетку осей (PlotItem) один раз при инициализации виджета
        self.pA = self.addPlot(row=0, col=0, colspan=4)
        self.pB = self.addPlot(row=1, col=0, colspan=1)
        self.pC = self.addPlot(row=1, col=2, colspan=1)
        self.pD = self.addPlot(row=2, col=0, colspan=1)
        self.pE = self.addPlot(row=2, col=2, colspan=1)



        # 2. Инициализируем графические элементы внутри осей и сохраняем на них ссылки
        self._init_graphic_items()

        # Если данные переданы сразу, отображаем их
        if data is not None:
            self.visual_dashbord(data)


        self.setBackground(None)

    def _init_graphic_items(self):
        """Создание неизменяемой структуры графиков (оси, зоны, пустые линии)"""
        # --- График A: Per Base Sequence Quality ---
        self.pA.setTitle('Per Base Sequence Quality', size='8pt')
        self.pA.setLabel('bottom', 'Position in read (bp)', size='8pt')
        self.pA.setLabel('left', 'Quality (Phred score)', size='8pt')
        self.pA.setYRange(0, 42)
        self.pA.showGrid(x=True, y=True, alpha=0.3)

        # Фоновые зоны создаем ОДИН раз (они статичны)
        green_zone = pg.LinearRegionItem([28, 42], orientation='horizontal', movable=False, brush=QColor(0, 255, 0, 30))
        yellow_zone = pg.LinearRegionItem([20, 28], orientation='horizontal', movable=False, brush=QColor(255, 255, 0, 30))
        red_zone = pg.LinearRegionItem([0, 20], orientation='horizontal', movable=False, brush=QColor(255, 0, 0, 30))
        self.pA.addItem(green_zone)
        self.pA.addItem(yellow_zone)
        self.pA.addItem(red_zone)

        # Ссылки на линии графика А
        self.curve_median = self.pA.plot(pen=pg.mkPen('r', width=2), name='Медиана')
        self.curve_mean = self.pA.plot(pen=pg.mkPen('b', width=1), name='Среднее')
        self.curve_q90 = self.pA.plot(pen=pg.mkPen(color=(0, 0, 255, 120), width=0.8, style=Qt.DashLine))
        self.curve_q10 = self.pA.plot(pen=pg.mkPen(color=(0, 0, 255, 120), width=0.8, style=Qt.DashLine))

        self.bars_A = pg.BarGraphItem(x = [], width = 0.9, y0 = [], y1 = [], brush = QColor(255, 165, 0, 150), pen = None)
        self.pA.addItem(self.bars_A)

        # Создание легенды для графика A
        self.legendA = pg.LegendItem(frame=False, colCount=4)
        self.legendA.setColumnCount(1)
        self.legendA.setMaximumHeight(30) 
        self.legendA.setLabelTextColor('black')
        self.addItem(self.legendA, row = 0, col = 4)

        self.legendA.addItem(self.curve_median, 'Медиана')
        self.legendA.addItem(self.curve_mean, 'Среднее')
        self.legendA.addItem(self.bars_A, 'q75/q25')
        self.legendA.addItem(self.curve_q90, 'q90')
        self.legendA.addItem(self.curve_q10, 'q10')


        # --- График B: Distribution of GC Content (Гистограмма) ---
        self.pB.setTitle('Distribution of GC Content', size='8pt')
        self.pB.setLabel('bottom', '% GC', size='8pt')
        self.pB.setLabel('left', 'Количество ридов', size='8pt')
        self.bg_B = pg.BarGraphItem(x=[], height=[], width=0.6, brush='b')
        self.pB.addItem(self.bg_B)

        # --- График C: Distribution of Read Lengths (Гистограмма) ---
        self.pC.setTitle('Distribution of Read Lengths', size='8pt')
        self.pC.setLabel('bottom', 'Длина рида (bp)', size='8pt')
        self.pC.setLabel('left', 'Количество ридов', size='8pt')
        self.bg_C = pg.BarGraphItem(x=[], height=[], width=0.6, brush='b')
        self.pC.addItem(self.bg_C)

        # --- График D: Distribution of Mean Read Quality (Гистограмма) ---
        self.pD.setTitle('Distribution of Mean Read Quality', size='8pt')
        self.pD.setLabel('bottom', 'Среднее качество рида (Phred)', size='8pt')
        self.pD.setLabel('left', 'Количество ридов', size='8pt')
        self.bg_D = pg.BarGraphItem(x=[], height=[], width=0.6, brush='b')
        self.pD.addItem(self.bg_D)

        # --- График E: Per Base Sequence Content ---
        self.pE.setTitle('Per Base Sequence Content', size='8pt')
        self.pE.setLabel('bottom', 'Position in read (bp)', size='8pt')
        self.pE.setLabel('left', 'Percentage (%)', size='8pt')
        self.pE.setYRange(0, 100)

        # Создание легенды для графика E
        self.legendE = pg.LegendItem()
        self.legendE.setColumnCount(1)
        self.legendE.setMaximumHeight(30) 
        self.legendE.setLabelTextColor('black')
        self.addItem(self.legendE, row = 2, col = 4)

        # Создаем 5 линий для нуклеотидов А, С, G, T, N
        colors = {'A': 'g', 'C': 'b', 'G': 'k', 'T': 'r', 'N': 'b'}
        self.nuc_curves = {}
        for base, color in colors.items():
            style = Qt.DashLine if base == 'N' else Qt.SolidLine
            self.nuc_curves[base] = self.pE.plot(pen=pg.mkPen(color, width=1, style=style), name=base)
            self.legendE.addItem(self.nuc_curves[base], base)

    def visual_dashbord(self, data):
        """Главный метод обновления. Сюда мы просто передаем новый объект REPORT"""
        self.REPORT = data
        
        # Запускаем методы обновления данных для каждого PlotItem
        self._update_distrub_qual()
        self._update_distrub_gc()
        self._update_distrub_len()
        self._update_distrub_qual_count()
        self._update_distrub_nuc()

    def _update_distrub_qual(self):
        stat = self.calculate_per_base_stats_fast(self.REPORT.distrub_qual)
        max_len = self.REPORT.max_len_read
        positions = list(range(1, min(len(stat['mean']), max_len) + 1))

        self.pA.setXRange(1, max_len)

        # Вместо .plot() вызываем .setData() на сохраненных кривых
        self.curve_median.setData(positions, stat['median'][:max_len])
        self.curve_mean.setData(positions, stat['mean'][:max_len])
        self.curve_q90.setData(positions, stat['q90'][:max_len])
        self.curve_q10.setData(positions, stat['q10'][:max_len])

        # Для гистограммы .setOpts()
        y1_data = stat['q75'][:max_len] + 0.2
        self.bars_A.setOpts(x=positions, y0=stat['q25'][:max_len], y1=y1_data)

        self.pA.autoRange()

    def _update_distrub_qual_count(self):
        x = list(range(self.REPORT.distrub_qual_c.size))
        y = list(self.REPORT.distrub_qual_c)
        # Для гистограмм pyqtgraph использует метод setOpts()
        self.bg_D.setOpts(x=x, height=y, width=0.6)
        self.pD.autoRange()

    def _update_distrub_gc(self):
        x = list(range(1, 101))
        y = list(self.REPORT.distrub_gc)
        self.bg_B.setOpts(x=x, height=y, width=0.6)
        self.pB.autoRange()

    def _update_distrub_len(self):
        x = list(range(1, self.REPORT.max_len_read + 1))
        y = list(self.REPORT.distrub_len)
        self.bg_C.setOpts(x=x, height=y, width=0.6)
        self.pC.autoRange()

    def _update_distrub_nuc(self):
        distrub = self.REPORT.distrub_nuc
        row_sums = distrub.sum(axis=1, keepdims=True)
        # Защита от деления на ноль, если массив пустой
        row_sums[row_sums == 0] = 1 
        distrub_norm = distrub / row_sums * 100
        
        positions = list(range(1, len(distrub_norm) + 1))
        bases = ['A', 'C', 'G', 'T', 'N']
        
        for i, base in enumerate(bases):
            self.nuc_curves[base].setData(positions, list(distrub_norm[:, i]))
        self.pE.autoRange()

    def calculate_per_base_stats_fast(self, distrub_qual):
        n_positions = distrub_qual.shape[0]
        n_qual_levels = distrub_qual.shape[1]

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
            counts = distrub_qual[pos]
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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    graph = PlotPyQtGraph()
    graph.show()

    sys.exit(app.exec())