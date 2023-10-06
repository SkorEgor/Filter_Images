import random

from Bilinear_Interpolation import Bilinear_Interpolation
from gui import Ui_Dialog

from graph import Graph
from drawer import Drawer as drawer
from FastFourierTransform import FFT

from PyQt5 import QtCore
import cv2
import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def uniform_distribution():
    repeat = 12
    val = 0
    for i in range(repeat):
        val += random.random()  # значение от 0.0 до 1.0
    return val / repeat


# Перевод цветной картинки в серую
def black_white_image(color_picture):
    height, width, _ = color_picture.shape

    gray_image = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            pixel = color_picture[i, j]
            gray_image[i, j] = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

    maximum_intensity = np.max(gray_image)
    multiplier = 255 / maximum_intensity
    gray_image = gray_image * multiplier

    return gray_image


def dome_2d(amplitude, x, x_0, sigma_x, y, y_0, sigma_y):
    return amplitude * math.exp(-(
            (
                    ((x - x_0) * (x - x_0)) /
                    (2 * sigma_x * sigma_x)
            ) + (
                    ((y - y_0) * (y - y_0)) /
                    (2 * sigma_y * sigma_y)
            )))


# Находит энергию прямоугольника, расположенного по центру картинки
def energy_central_rectangle(half_width, half_rectangle_width,
                             half_height, half_rectangle_height,
                             picture):
    zone_energy = 0
    for i in range(half_width - half_rectangle_width, half_width + half_rectangle_width):
        for j in range(half_height - half_rectangle_height, half_height + half_rectangle_height):
            zone_energy += picture[j, i] * picture[j, i]
    return zone_energy


# Возвращает энергию картинку
def energy_pictures(pictures):
    energy = 0
    for picture_line in pictures:
        for pixel in picture_line:
            energy += pixel * pixel
    return energy


# Считаем критерий разницы изображений
def criterion_difference_images(images1, images2):
    epsilon = 0
    height, width = images1.shape
    for j in range(height):
        for i in range(width):
            epsilon += \
                (images1[j, i] - images2[j, i]) * \
                (images1[j, i] - images2[j, i])
    return epsilon / energy_pictures(images1)


# Проверка размера на степень двойки
def checking_powers_two(matrix):
    # Находим ширину и высоту
    height, width = matrix.shape
    start_interpolation = False

    # Если высота не степень двойки
    if height & (height - 1) != 0:
        print("height")
        height = 2 ** (int(math.log2(height)) + 1)
        start_interpolation = True

    # Если ширина не степень двойки
    if width & (width - 1) != 0:
        print("width")
        width = 2 ** (int(math.log2(width)) + 1)
        start_interpolation = True

    if start_interpolation:
        print("start")
        matrix = Bilinear_Interpolation.matrix_interpolation(matrix, height, width)

    return matrix


# КЛАСС АЛГОРИТМА ПРИЛОЖЕНИЯ
class GuiProgram(Ui_Dialog):
    # Конструктор
    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        dialog.setWindowFlags(  # Передаем флаги создания окна
            QtCore.Qt.WindowCloseButtonHint |  # Кнопка закрытия
            QtCore.Qt.WindowMaximizeButtonHint |  # Кнопка развернуть
            QtCore.Qt.WindowMinimizeButtonHint  # Кнопка свернуть
        )
        # Устанавливаем пользовательский интерфейс
        self.setupUi(dialog)

        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Исходная картинка
        self.graph_original_picture = Graph(
            layout=self.layout_plot_1,
            widget=self.widget_plot_1,
            name_graphics="Рис. 1. Исходное изображение"
        )
        # Параметры 2 графика - Исходная картинка
        self.graph_picture_with_noise = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="Рис. 2. Изображение с шумом"
        )
        # Параметры 3 графика - Спектр, со сменой по диагонали и зоной процента энергии
        self.graph_spectrum = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="Рис. 3. Спектр и область с заданной энергией"
        )
        # Параметры 4 графика - Восстановленная картинка
        self.graph_restored_picture = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="Рис. 4. Восстановленное изображение"
        )

        # Картинки этапов обработки
        self.color_picture = None
        self.grey_picture = None

        self.original_picture = None
        self.noise_image = None

        self.picture_spectrum = None
        self.module_spectrum_repositioned = None
        self.constant_module_spectrum = None

        self.half_rectangle_width = None
        self.half_rectangle_height = None

        self.restored_picture = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ
        # спрятать область загрузки картинки
        self.widget_loading_pictures.setVisible(False)

        # Смена режима отображения картинки
        self.radioButton_color_picture.clicked.connect(self.change_picture_to_colored)
        self.radioButton_gray_picture.clicked.connect(self.change_picture_to_gray)
        # Выбрана картинка или график
        self.radioButton_generation_domes.clicked.connect(self.creating_dome)
        self.radioButton_loading_pictures.clicked.connect(self.display_picture)

        # Алгоритм обратки
        # Создание куполов
        self.pushButton_display_domes.clicked.connect(self.creating_dome)
        # Загрузка картинки
        self.pushButton_loading_pictures.clicked.connect(self.load_image)
        # Добавление шума
        self.pushButton_display_noise.clicked.connect(self.noise)
        # Добавление шума
        self.pushButton_start_processing.clicked.connect(self.spectrum_numpy)

    # ОБРАБОТКА ИНТЕРФЕЙСА
    # Смена режима отображения картинки
    # Выбрано отображение цветной картинки
    def change_picture_to_colored(self, state):
        if state and self.color_picture is not None:
            drawer.image_color_2d(self.graph_original_picture, self.color_picture)

    # Выбрано отображение серой картинки
    def change_picture_to_gray(self, state):
        if state and self.original_picture is not None:
            drawer.image_gray_2d(self.graph_original_picture, self.original_picture)

    # Отобразить картинку
    def display_picture(self):
        # Картинки нет - не отображаем
        if self.color_picture is None:
            return

        self.original_picture = self.grey_picture

        # Проверяем вид отображаемой картинки
        if self.radioButton_color_picture.isChecked():
            drawer.image_color_2d(self.graph_original_picture, self.color_picture)
        else:
            drawer.image_gray_2d(self.graph_original_picture, self.original_picture)

        self.noise()

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1г) Вычислить график
    def creating_dome(self):
        # Запрашиваем размер области
        width_area = int(self.lineEdit_width_area.text())
        height_area = int(self.lineEdit_height_area.text())

        # Запрашиваем параметры куполов
        amplitude_1 = float(self.lineEdit_amplitude_1.text())
        x0_1 = float(self.lineEdit_x0_1.text())
        sigma_x_1 = float(self.lineEdit_sigma_x_1.text())
        y0_1 = float(self.lineEdit_y0_1.text())
        sigma_y_1 = float(self.lineEdit_sigma_y_1.text())

        amplitude_2 = float(self.lineEdit_amplitude_2.text())
        x0_2 = float(self.lineEdit_x0_2.text())
        sigma_x_2 = float(self.lineEdit_sigma_x_2.text())
        y0_2 = float(self.lineEdit_y0_2.text())
        sigma_y_2 = float(self.lineEdit_sigma_y_2.text())

        amplitude_3 = float(self.lineEdit_amplitude_3.text())
        x0_3 = float(self.lineEdit_x0_3.text())
        sigma_x_3 = float(self.lineEdit_sigma_x_3.text())
        y0_3 = float(self.lineEdit_y0_3.text())
        sigma_y_3 = float(self.lineEdit_sigma_y_3.text())

        # Создаем пустую матрицу пространства
        self.original_picture = np.zeros((height_area, width_area))

        # Для каждой точки матрицы считаем сумму куполов
        for x in range(width_area):
            for y in range(height_area):
                self.original_picture[y, x] = dome_2d(amplitude_1, x, x0_1, sigma_x_1, y, y0_1, sigma_y_1) + \
                                              dome_2d(amplitude_2, x, x0_2, sigma_x_2, y, y0_2, sigma_y_2) + \
                                              dome_2d(amplitude_3, x, x0_3, sigma_x_3, y, y0_3, sigma_y_3)

        # Выводим картинку
        drawer.graph_color_2d(self.graph_original_picture, self.original_picture)
        # Проверка на размерность степени двойки для быстрого фурье
        self.original_picture = checking_powers_two(self.original_picture)

        # Считаем и показываем шум
        self.noise()

    # (1к) Загрузить картинку
    def load_image(self):
        # Вызов окна выбора файла
        # filename, filetype = QFileDialog.getOpenFileName(None,
        #                                                  "Выбрать файл изображения",
        #                                                  ".",
        #                                                  "All Files(*)")
        filename = "image_256_128.png"

        # Загружаем картинку
        self.color_picture = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Конвертируем в серый
        self.grey_picture = black_white_image(self.color_picture)

        self.original_picture = self.grey_picture

        # Отображаем картинку
        self.display_picture()
        # Проверка на размерность степени двойки для быстрого фурье
        self.original_picture = checking_powers_two(self.original_picture)

    # (2) Накладываем шум
    def noise(self):
        # Нет исходны данных - сброс
        if self.original_picture is None:
            return

        self.noise_image = self.original_picture.copy()

        # Считаем энергию изображения
        energy_noise_image = energy_pictures(self.noise_image)

        # Создаем изображение шума
        height, width = self.noise_image.shape
        picture_noise = np.zeros((height, width))
        energy_noise = 0
        for x in range(width):
            for y in range(height):
                val = uniform_distribution()
                # Записываем пиксель шума
                picture_noise[y, x] = val
                # Копим энергию шума
                energy_noise += val * val

        # Запрашиваем процент шума
        noise_percentage = float(self.lineEdit_noise.text()) / 100
        # Считаем коэффициент/множитель шума
        noise_coefficient = math.sqrt(noise_percentage *
                                      (energy_noise_image / energy_noise))

        # К пикселям изображения добавляем пиксель шума
        for x in range(width):
            for y in range(height):
                self.noise_image[y, x] += noise_coefficient * picture_noise[y, x]

        # Отображаем итог
        # выбираем график или картинку
        if self.radioButton_loading_pictures.isChecked():
            drawer.image_gray_2d(self.graph_picture_with_noise, self.noise_image)
        else:
            drawer.graph_color_2d(self.graph_picture_with_noise, self.noise_image)

        # Считаем разницы изображения с шумом
        epsilon = criterion_difference_images(self.original_picture, self.noise_image)
        self.label_deviation_original_and_noise.setText(f'{epsilon:.4f}')

    # (3) Спектр от картины с шумом, спектр с диагональной перестановкой
    def spectrum_numpy(self):
        if self.noise_image is None:
            return

        # Переводи картинку в комплексную
        complex_image = np.array(self.noise_image, dtype=complex)

        # Считаем спектр
        self.picture_spectrum = FFT.matrix_fft(complex_image)

        # Берем модуль, для отображения
        module_picture_spectrum = abs(self.picture_spectrum)
        self.constant_module_spectrum = module_picture_spectrum[0, 0]
        module_picture_spectrum[0, 0] = 0

        # Матрица со спектром посередине
        height, width = module_picture_spectrum.shape
        middle_h = height // 2
        middle_w = width // 2
        self.module_spectrum_repositioned = np.zeros((height, width))

        # Меняем по главной диагонали
        self.module_spectrum_repositioned[0:middle_h, 0:middle_w] = \
            module_picture_spectrum[middle_h:height, middle_w:width]

        self.module_spectrum_repositioned[middle_h:height, middle_w:width] = \
            module_picture_spectrum[0:middle_h, 0:middle_w]

        # Меняем по главной диагонали
        self.module_spectrum_repositioned[middle_h:height, 0:middle_w] = \
            module_picture_spectrum[0:middle_h, middle_w:width]

        self.module_spectrum_repositioned[0:middle_h, middle_w:width] = \
            module_picture_spectrum[middle_h:height, 0:middle_w]

        self.search_zone()

    # (4) поиск ширины и высоты зоны
    def search_zone(self):
        if self.module_spectrum_repositioned is None:
            return

        height, width = self.module_spectrum_repositioned.shape

        # Энергия модуля спектра без константы
        spectrum_module_energy = energy_pictures(self.module_spectrum_repositioned)

        # Запрашиваем процент энергии
        percent_filtering = float(self.lineEdit_percent_energy.text()) / 100

        # Фильтрация
        # 0. Вычисляем половину высоты и ширины
        half_width = width // 2
        half_height = height // 2

        half_rectangle_width = None
        half_rectangle_height = None

        # 1. Находим большую сторону и по ней перебираем по пикселям
        # Если ширина больше
        if width >= height:
            # Начиная с половины ширины до центра перебираем прямоугольники
            for half_rectangle_width in range(half_width, 0, -1):
                # Для данной ширины, находим высоту
                half_rectangle_height = int((half_rectangle_width * half_height) / half_width)

                # Считаем энергию зоны
                zone_energy = energy_central_rectangle(half_width, half_rectangle_width,
                                                       half_height, half_rectangle_height,
                                                       self.module_spectrum_repositioned)
                # Как только энергия в области стала равна или меньше заданного процента - стоп
                if zone_energy / spectrum_module_energy <= percent_filtering:
                    break

        # Если высота больше
        else:
            # Начиная с половины высоты до центра перебираем прямоугольники
            for half_rectangle_height in range(half_height, 0, -1):
                # Для данной ширины, находим высоту
                half_rectangle_width = int((half_rectangle_height * half_width) / half_height)

                # Считаем энергию зоны
                zone_energy = energy_central_rectangle(half_width, half_rectangle_width,
                                                       half_height, half_rectangle_height,
                                                       self.module_spectrum_repositioned)

                # Как только энергия в области стала равна или меньше заданного процента - стоп
                if zone_energy / spectrum_module_energy <= percent_filtering:
                    break

        # Находим координаты прямоугольника
        x1 = half_width - half_rectangle_width - 0.5
        y1 = half_height - half_rectangle_height - 0.5

        x2 = half_width + half_rectangle_width - 0.5
        y2 = half_height - half_rectangle_height - 0.5

        x3 = half_width + half_rectangle_width - 0.5
        y3 = half_height + half_rectangle_height - 0.5

        x4 = half_width - half_rectangle_width - 0.5
        y4 = half_height + half_rectangle_height - 0.5

        # Рисуем модуль спектра
        drawer.graph_color_2d(self.graph_spectrum, self.module_spectrum_repositioned,
                              logarithmic_axis=self.radioButton_logarithmic_axis.isChecked())

        # Строим график прямых от точки к точке. Прямоугольник выбранной области
        self.graph_spectrum.axis.plot(
            [x1, x2, x3, x4, x1],
            [y1, y2, y3, y4, y1])

        # Убеждаемся, что все помещается внутри холста
        self.graph_spectrum.figure.tight_layout()
        # Показываем новую фигуру в интерфейсе
        self.graph_spectrum.canvas.draw()

        # Запоминаем размер области заданного процента энергии
        self.half_rectangle_width = half_rectangle_width
        self.half_rectangle_height = half_rectangle_height

        self.spectrum_nulling()

    # (5) Зануляем комплексный спектр за областью прямоугольника
    def spectrum_nulling(self):
        if (self.half_rectangle_width is None and
                self.half_rectangle_height is None and
                self.picture_spectrum is None):
            return

        height, width = self.picture_spectrum.shape

        for i in range(self.half_rectangle_width, width - self.half_rectangle_width):
            for j in range(0, self.half_rectangle_height):
                self.picture_spectrum[j, i] = 0 + 0j

        for i in range(0, width):
            for j in range(self.half_rectangle_height, height - self.half_rectangle_height):
                self.picture_spectrum[j, i] = 0 + 0j

        for i in range(self.half_rectangle_width, width - self.half_rectangle_width):
            for j in range(height - self.half_rectangle_height, height):
                self.picture_spectrum[j, i] = 0 + 0j

        self.inverse_fourier()

    # (6) Вызываем обратное преобразование фурье для восстановления картинки
    def inverse_fourier(self):
        if self.picture_spectrum is None:
            return

        # Считаем спектр
        spectral_reconstructed_image = FFT.matrix_fft_reverse(self.picture_spectrum)

        # Берем модуль, для отображения
        self.restored_picture = abs(spectral_reconstructed_image)

        drawer.graph_color_2d(self.graph_restored_picture, self.restored_picture)
        self.difference_original_and_restored()

    # (7) Отображает скалярную разницу между исходным и восстановленным изображением
    def difference_original_and_restored(self):
        # Проверяем наличие изображений
        if self.restored_picture is None and self.original_picture is None:
            return

        # Считаем критерий очистки и отображаем
        epsilon = criterion_difference_images(self.original_picture, self.restored_picture)
        self.label_deviation_original_and_received.setText(f'{epsilon:.4f}')
