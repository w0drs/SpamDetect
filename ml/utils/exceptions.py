class DataLoadingError(Exception):
    """Базовое исключение для ошибок загрузки данных"""
    ...

class CSVFileNotFoundError(DataLoadingError):
    """Ошибка когда CSV файл не найден"""
    def __init__(self, file_path = ""):
        self.file_path = file_path
        super().__init__(f"CSV файл не найден по пути: {file_path}")

class CSVLoadingError(DataLoadingError):
    """Ошибка при чтении CSV файла"""
    def __init__(self, file_path = "", original_error = ""):
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(f"Ошибка загрузки CSV файла {file_path}: {original_error}")

class EmptyDataError(DataLoadingError):
    """Ошибка когда CSV файл пустой"""
    def __init__(self, file_path):
        super().__init__(f"CSV файл {file_path} пустой")