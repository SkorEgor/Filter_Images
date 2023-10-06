import numpy as np


# Билинейная интерполяция
class Bilinear_Interpolation:
    # Интерполяция для двух пикселей
    @staticmethod
    def interpolation_between_two_pixels(x_1, color_1,
                                         x_2, color_2,
                                         x_3):
        """
        Каноническое уравнение прямой между точками:
        x-x1   y-y1
        ---- = ----
        x2-x1  y2-y1
        Зная координату x, можно найти y. Дробь с x обозначим за k
        y = (y2-y1)*k + y1
        x - будет координатой; y - будет цветом
        """
        # Дробь с x обозначим за k
        k = (x_3 - x_1) / (x_2 - x_1)

        # Возвращаем цвет
        return (color_2 - color_1) * k + color_1

    # Интерполяция для массива
    @staticmethod
    def array_interpolation(original_array, new_size):
        # Находим размер исходного массива
        original_size = original_array.size

        # Если увеличивать не нужно, сброс
        if original_size >= new_size:
            return

        # Находим шаг
        new_step = (original_size - 1) / (new_size - 1)
        # Создаем результирующий массив,
        # Нужного размера, значения индексов от 0 до макс индекса исходного
        new_array = np.arange(new_size) * new_step

        # Перебираем индексы нового массива
        for i in range(new_size):
            # Если значение под индексом целое, берем исходное значение
            if new_array[i] % 1 == 0:
                new_array[i] = original_array[int(new_array[i])]
            # Иначе интерполируем
            else:
                x1 = int(new_array[i])
                x3 = x1 + 1
                new_array[i] = Bilinear_Interpolation.interpolation_between_two_pixels(
                    x1, original_array[x1],
                    x3, original_array[x3],
                    new_array[i]
                )

        return new_array

    @staticmethod
    def matrix_interpolation(matrix, new_height, new_width):
        height, width = matrix.shape

        # Если не расширять - сброс
        if new_height < height and new_width < width:
            return

        final_matrix = np.zeros((new_height, new_width))
        final_matrix[0:height, 0:width] = matrix[0:height, 0:width]

        # Увеличение ширины
        if new_width > width:
            for i in range(height):
                final_matrix[i] = Bilinear_Interpolation.array_interpolation(final_matrix[i][0:width], new_width)
            width = new_width

        # Увеличение высоты
        if new_height > height:
            final_matrix = final_matrix.transpose()

            for i in range(width):
                final_matrix[i] = Bilinear_Interpolation.array_interpolation(final_matrix[i][0:height], new_height)

            final_matrix = final_matrix.transpose()

        print(final_matrix.shape)
        return final_matrix
