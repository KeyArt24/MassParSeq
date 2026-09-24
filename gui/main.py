
import os
import re
import sys
import time
import gc
from pathlib import Path

# Получаем путь к папке 'gui'
current_dir = os.path.dirname(os.path.abspath(__file__))
# Поднимаемся на один уровень вверх — в корень проекта 'Massparseq EDAQ'
project_root = os.path.dirname(current_dir)

# Добавляем корень проекта в список путей, где Python ищет модули
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataclasses import dataclass
from collections import deque

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QToolBar, QFileDialog, QMessageBox, QScrollArea, QTextEdit
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import QThread, Signal
import pyqtgraph as pg

from gui_styles.css_style import css_tab_style
from core.Njit_FastaQ import PYJITFASTQ, Jit
from core.graph_plot import PlotPyQtGraph
from about import about_programm



@dataclass
class Card():

	file_name: str
	file_path: str

	tab_information: QLabel
	tab_report: PlotPyQtGraph


@dataclass
class AppState():

	loaded_files = {}
	analysis_queue = {}
	current_thread = None

class ParseWorkerThread(QThread):
	# Сигнал для передачи результатов (отчет, время, размер файла) в главный поток
	analysis_finished = Signal(object, float, object)
	# Сигнал на случай, если внутри парсера произойдет ошибка (например, MemoryError)
	analysis_failed = Signal(str)
	# Сигнал на статуса анализа
	analysis_status = Signal(str)

	def __init__(self, parse: PYJITFASTQ, file_path, parent=None):
		super().__init__(parent=parent)
		self.parser = parse
		self.parser.status_callback = self.analysis_status.emit
		self.file_path = file_path
		self._cancelled = False


	def cancel(self):
		self._cancelled = True

	def is_cancelled(self):
		return self._cancelled

	def run(self):
		"""Этот метод автоматически выполняется в отдельном системном потоке"""
		try:
			
			self.parser.status_callback('Запуск анализа')
			jit = Jit()
			jit.warmup()
			# Вызываем переданную функцию парсинга в фоне
			start_time = time.perf_counter()
			self.parser.status_callback('Загрузка файла....')
			self.parser.load_fastq(self.file_path)
			self.parser.status_callback('Анализ запущен....')
			self.parser.run(cancel=self.is_cancelled)
			elapsed = time.perf_counter() - start_time
			self.parser.print_report(elapsed = elapsed, file_size = os.path.getsize(self.file_path))
			# Отправляем результаты обратно в главное окно через безопасный сигнал Qt
			self.analysis_finished.emit(self.parser.REPORT, elapsed, os.path.getsize(self.file_path))
		except Exception as e:
			# Если парсер упал (например, из-за нехватки памяти), передаем текст ошибки
			import traceback
			error_msg = f"{str(e)}\n{traceback.format_exc()}"
			self.analysis_failed.emit(error_msg)

class MainWindow(QMainWindow):
	def __init__(self, parent = None):
		super().__init__()

		self._init_ui_()
		self._init_actions_()
		self._init_menu_()
		self._config_()
		self._connect_signals_()

	def _init_ui_(self):
		self.tab_cards = QTabWidget()
		self.main_lay = QVBoxLayout()
		self.main_widget = QWidget()
		self.state = AppState()
		self.about = QTextEdit()
		

		self.main_lay.addWidget(self.tab_cards)
		self.main_widget.setLayout(self.main_lay)
		self.setCentralWidget(self.main_widget)


	def _init_actions_(self):
		self.action_file_analysis = QAction(QIcon(), 'Запустить анализ файла', self)
		self.action_all_files_analysis = QAction(QIcon(), 'Запустить анализ всех файлов', self)
		self.action_stop_analysis = QAction(QIcon(), 'Остановить анализ', self)
		self.action_about = QAction(QIcon(), 'О программе', self)

		self.action_load_file = QAction(QIcon(), 'Загрузить файлы', self)
		#self.action_load_dir = QAction(QIcon(), 'Загрузить папку', self) ---------- MINUS ACTION
		self.action_save_result = QAction(QIcon(), 'Сохранить отчёт', self)

	def _init_menu_(self):
		menu_bar = self.menuBar()
		menu_file = menu_bar.addMenu('Файл')
		menu_analysis = menu_bar.addMenu('Анализ')
		self.menu_information = self.menuBar().addAction(self.action_about)
		
		menu_file.addAction(self.action_load_file)
		#menu_file.addAction(self.action_load_dir) ---------- MINUS ACTION
		menu_file.addAction(self.action_save_result)

		menu_analysis.addAction(self.action_file_analysis)
		menu_analysis.addAction(self.action_all_files_analysis)
		menu_analysis.addAction(self.action_stop_analysis)

	def _init_toolbar_(self):
		self.toolbar = QToolBar('Главная панель')
		self.addToolBar(self.toolbar)

	def _config_(self):
		# Настройки главного окна
		self.resize(800, 600)
		self.setWindowTitle('Масспарсек ЭДАК')

		# Настройки вкладок
		self.tab_cards.setTabsClosable(True)
		self.tab_cards.setStyleSheet(css_tab_style)

	def _style_(self):
		pass

	def _connect_signals_(self):
		self.tab_cards.tabCloseRequested.connect(self.close_tab)
		self.action_file_analysis.triggered.connect(self.start_analysis_file)
		self.action_all_files_analysis.triggered.connect(self.start_analysis_all_files)
		self.action_load_file.triggered.connect(self.open_file)
		self.action_stop_analysis.triggered.connect(self.stop_parsing)
		self.action_about.triggered.connect(self.open_about)

	def add_cards(self):
		tab_widget = QWidget()
		for card in self.state.loaded_files:
			label = QLabel(f'Анализ данных {card} не запущен')
			lay = QVBoxLayout()
			lay.addWidget(label)
			tab_widget.setLayout(lay)
			self.tab_cards.addTab(tab_widget, card)

	def close_tab(self, index):
		card = self.tab_cards.tabBar().tabData(index)
		widget = self.tab_cards.widget(index)
		self.tab_cards.removeTab(index)
		if widget is not None:
			widget.deleteLater()

		if card is not None:
			self.state.loaded_files.pop(card.file_name, None)
			self.state.analysis_queue.pop(card.file_name, None)

			if self.state.current_thread is not None and self.state.current_thread.file_path == card.file_path:
				self.state.current_thread.analysis_finished.disconnect()
				self.state.current_thread.analysis_status.disconnect()
				self.state.current_thread.cancel()

				while self.state.current_thread.isRunning():
					self.state.current_thread.cancel()
					self.state.current_thread.wait(1000)

				self.state.current_thread = None
				self.process_analysis()



	def start_analysis_file(self):
		# Определяем индекс активной вкладки
		current_index = self.tab_cards.currentIndex()
		if current_index < 0:
			print(f'Индекс {current_index} карточки не определён')
			return
		# Получаем карточку файла
		card = self.tab_cards.tabBar().tabData(current_index)
		if card is None:
			print(f'Карточка вкладки не найдена')
			return
		# Проверяем есть ли карточка в очередь на анализ
		if card.file_name in self.state.analysis_queue:
			return
		else:
			# Если нет в очереди на анализ то ставим в очередь и запускаем цикл анализа
			card.tab_information.setText('В очереди на анализ.....')
			self.state.analysis_queue[card.file_name] = card

			self.process_analysis()
	
	def process_analysis(self):
		# Проверяем есть ли в очереди файлы для анализа
		if not self.state.analysis_queue:
			return


		file_name, card = next(iter(self.state.analysis_queue.items()))
		tab_information = card.tab_information
		tab_report = card.tab_report
		tab_report.hide()

		if not (tab_information or tab_report):
			print(f"Виджеты не найдены для {file_name}, пропускаем")
			self.state.analysis_queue.pop(file_name, None)
			self.process_analysis()
			return

		# Внутрення функция для отслеживания процесса окончания
		def on_parsing_done(report, elapsed, file_size):
			information_run = self.create_report(report, elapsed, file_size)
			tab_information.setText(information_run)
			tab_report.show()
			tab_report.visual_dashbord(report)
			tab_report.update()

			self.state.analysis_queue.pop(card.file_name, None)	
			self.state.current_thread = None
			self.process_analysis()
			
		# Внутрення функция для отслеживания статуса процесса 
		def on_parsing_run(msg):
			tab_information.setText(msg)


		thread = self.run_parsing(card.file_path)

		if self.state.current_thread:
			return

		self.state.current_thread = thread


		# Привязываем сигналы потока к нашим локальным функциям
		thread.analysis_finished.connect(on_parsing_done)
		thread.analysis_status.connect(on_parsing_run)
		# Стартуем поток. Управление мгновенно вернется в интерфейс PySide!
		thread.start()
		


	def start_analysis_all_files(self):
		for file, card in self.state.loaded_files.items():
			if file in self.state.analysis_queue:
				continue
			else:
				# Если нет в очереди на анализ то ставим в очередь и запускаем цикл анализа
				card.tab_information.setText('В очереди на анализ.....')
				self.state.analysis_queue[card.file_name] = card
				self.process_analysis()


	def run_parsing(self, file_path):
		if file_path:
			gc.collect()
			self.edaq_engine = PYJITFASTQ()
			thread = ParseWorkerThread(self.edaq_engine, file_path)
			return thread

	def stop_parsing(self):
		if self.state.current_thread is not None:
			while self.state.current_thread.isRunning():
					self.state.current_thread.cancel()
					self.state.current_thread.wait(1000)

			self.state.current_thread = None

		for file, card in self.state.analysis_queue.items():
			card.tab_information.setText((f'Анализ данных {card.file_name} не запущен'))
		
		self.state.analysis_queue.clear()


	

	def open_file(self):
		# Открываем диалог выбора файла
		# Метод возвращает кортеж: (выбранный_путь, выбранный_фильтр)
		files, _ = QFileDialog.getOpenFileNames(
			self,
			"Выберите FastQ файл для анализа",
			"",  # Стартовая папка (пустая строка — откроет текущую или последнюю)
			"FastQ Files (*.fastq *.fq *.gz);;All Files (*)"  # Фильтр расширений
		)
	    
		# Обязательно проверяем, выбрал ли пользователь файл 
		# (если он нажмет «Отмена», file_path будет пустой строкой)
		if files:
			for file_path in files:
				file_name = os.path.basename(file_path)


				if file_name in self.state.loaded_files:
					QMessageBox.warning(
						self,                                
						"Файл уже добавлен",                 
						f"Файл '{file_name}' уже добавлен!"  
					)
					return  # Прерываем выполнение функции, файл не перезаписывается
        
				# Создаём виджет вкладки для отображения результатов анализа
				scroll_area = QScrollArea()
				scroll_area.setWidgetResizable(True)
				tab_widget = QWidget()
			
				tab_information = QLabel(f'Анализ данных {file_name} не запущен')
				tab_information.setObjectName('tab_information')

				tab_report = PlotPyQtGraph()
				tab_report.setObjectName('tab_report')
				tab_report.setMinimumHeight(750)
				tab_report.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
			
				lay = QVBoxLayout()
				lay.addWidget(tab_information)
				lay.addWidget(tab_report)
				tab_widget.setLayout(lay)
				scroll_area.setWidget(tab_widget)

				tab_report.hide()

				# Если файла нет в словаре, добавляем его и создаём карточку
				card = Card(
							file_name=file_name,
							file_path=file_path,
							tab_information=tab_information,
							tab_report=tab_report,
							)

				self.state.loaded_files[file_name] = card

				index = self.tab_cards.addTab(scroll_area, file_name)
				self.tab_cards.tabBar().setTabData(index, card)




	def create_report(self,  report, elapsed = 0.0, file_size = 0):
		result = [
			'********************************** START REPORT ************************************',
			f"  A (Аденин)      : {report.count_a:>15,}",
			f"  C (Цитозин)     : {report.count_c:>15,}",
			f"  G (Гуанин)      : {report.count_g:>15,}",
			f"  T (Тимин)       : {report.count_t:>15,}",
			f"  N (Неизвестный) : {report.count_n:>15,}",
			f"  ────────────────────────────────────────────────────",
			f"  GC% : {(report.count_c + report.count_g) / report.count_nuc * 100:.2f}",
			f"  Всего ридов : {report.count_reads}",
			f"  Минимальная длина рида : {report.min_len_read}",
			f"  Максимальная длина рида : {report.max_len_read}",
			f"  Всего оснований : {report.count_nuc:>15,}",
			f"  Количество ОЗУ для построения массивов данных numpy : {report.ram / (1024 ** 3):.2f} ГБ",
			f"  Производительность : {(report.count_nuc + report.count_n) / elapsed / (100 ** 3):.2f} МБ оснований/с",
			f"  Размер файла     : {file_size / (1024 ** 3):.2f} ГБ",
			f"  Время            : {elapsed:.2f} секунд",
			'*********************************** END REPORT ************************************']
		
		report_text = "\n".join(result)

		return report_text

	def open_about(self):
		self.about.setHtml(about_programm)
		self.about.setWindowTitle('Справка')
		self.about.setMinimumSize(800, 600)
		self.about.show()

		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	main = MainWindow()
	main.show()
	app.exec()
