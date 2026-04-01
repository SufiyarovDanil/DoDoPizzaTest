from typing import List, Tuple, Optional
from enum import Enum
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


# Модели данных
class TableEventEnum(Enum):
    """Возможные состояния стола."""

    empty = 'empty'
    approach = 'approach'
    taken = 'taken'


@dataclass(frozen=True)
class Event:
    """События изменения состояния стола."""

    type: TableEventEnum
    timestamp_sec: int


@dataclass(frozen=True)
class Rect:
    """
    Прямоугольная область ROI.
    - x, y: координаты верхнего левого угла прямоугольника.
    """
    x: int
    y: int
    height: int
    width: int

    @property
    def top_left(self) -> Tuple[int, int]:
        """Возвращает координаты верхнего левого угла."""
        return self.x, self.y

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Возвращает координаты нижнего правого угла."""
        return self.x + self.width, self.y + self.height

    def intersects_with(self, other: "Rect") -> bool:
        """Проверяет, пересекается ли текущий прямоугольник с другим."""
        (stlx, stly), (sbrx, sbry) = self.top_left, self.bottom_right
        (otlx, otly), (obrx, obry) = other.top_left, other.bottom_right

        xl = max(sbrx, obrx) - min(stlx, otlx)
        yl = max(sbry, obry) - min(stly, otly)

        return xl <= self.width + other.width and yl <= self.height + other.height

    def fully_overlaps(self, other: "Rect") -> bool:
        """Проверяет, полностью ли текущий прямоугольник перекрывает другой."""
        (stlx, stly), (sbrx, sbry) = self.top_left, self.bottom_right
        (otlx, otly), (obrx, obry) = other.top_left, other.bottom_right

        overlaps_on_x: bool = stlx <= otlx and sbrx >= obrx
        overlaps_on_y: bool = stly <= otly and sbry >= obry

        return overlaps_on_x and overlaps_on_y


# Видео обработчик
class VideoProcessor:
    """Обеспечивает удобные чтение и итерацию по кадрам в видео."""

    def __init__(self, video_path: Path):
        self.video_path: Path = video_path
        self.video_capture = cv2.VideoCapture(self.video_path)

    @property
    def fps(self) -> int:
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))

    @property
    def frame_resolution(self) -> Tuple[int, int]:
        return (
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def __enter__(self):
        if not self.video_capture.isOpened():
            raise Exception(f'Не удалось открыть видеозапись {self.video_path}')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_capture:
            self.video_capture.release()

    @property
    def first_frame(self) -> cv2.Mat:
        pos_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.video_capture.read()

        if not ret:
            raise Exception('Не удалось прочитать первый кадр')

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)

        return frame

    def iter_batches(self, batch_size: int = 32):
        """
        Генератор, возвращающий пакет кадров для множественной обработки.
        batch_size - количество кадров в одном наборе.
        """
        batch: List[cv2.Mat] = []

        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()

            if not ret:
                yield batch
                break

            batch.append(frame)

            if len(batch) >= batch_size:
                yield batch
                batch = []


class VideoWriter:
    """Класс для записи видео с наложениями объектов в кадрах."""

    def __init__(self, output_path: Path, fps: int, resolution: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)

    def write(self, frame: cv2.Mat):
        """Записывает кадр в выходной файл"""
        self.video_writer.write(frame)

    def release(self):
        """Освобождает ресурсы для записывания видео"""
        self.video_writer.release()

    @staticmethod
    def draw_rect(frame: cv2.Mat, rect: Rect, bgr_color: Tuple[int, int, int]):
        """Рисует прямоугольник на кадре (цвет в формате BGR)"""
        cv2.rectangle(frame, rect.top_left, rect.bottom_right, bgr_color, 2)

    @staticmethod
    def put_text(frame: cv2.Mat, text: str, xy: Tuple[int, int], bgr_color: Tuple[int, int, int]) -> cv2.Mat:
        """Накладывает текст на кадр"""
        return cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_COMPLEX, 1, bgr_color, 2)


class RoiSelector:
    """Класс для выбора области в кадре для наблюдения."""

    @staticmethod
    def select(frame: cv2.Mat) -> Rect:
        """
        Открывает окно для ручного выделения области.
        Возвращает выделенную область в виде прямоугольника (Rect)
        """
        window_name: str = 'Выделите нужную область, и нажмите ENTER или SPACE (для отмены нажмите C)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x, y, w, h = map(
            int,
            cv2.selectROI(
                window_name,
                frame,
                fromCenter=False,
                showCrosshair=True,
                printNotice=False
            )
        )
        cv2.destroyWindow(window_name)

        if x == y == w == h == 0:
            raise Exception('Выделение области отменено.')

        return Rect(x=x, y=y, width=w, height=h)


class PersonDetector:
    """Класс для обнаружения людей на кадрах."""

    def __init__(self, model: YOLO):
        self.model = model

    def detect_in_frame(self, frames: List[cv2.Mat], rect: Rect) -> List[TableEventEnum]:
        """
        Анализирует набор кадров и определяет состояние стола для каждого.
        Логика определения:
        - taken: если человек полностью перекрывает выделенную область.
        - approach: если человек пересекается с областью, но не перекрывает полностью.
        - empty: если рядом с областью нет людей.

        frames: список кадров для анализа
        rect: область ROI
        """
        results = self.model(frames, classes=[0], conf=0.5, verbose=False)
        events: List[TableEventEnum] = []

        for f, r in zip(frames, results):
            someone_approaches: bool = False

            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                person_rect = Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

                if rect.fully_overlaps(person_rect):
                    events.append(TableEventEnum.taken)
                    break
                elif rect.intersects_with(person_rect):
                    someone_approaches = True
            else:
                events.append(TableEventEnum.approach if someone_approaches else TableEventEnum.empty)

        return events


class TableAnalytics:
    """Класс для сбора статистики и формирования отчета"""

    def __init__(self):
        self.events: List[Event] = []

    def add_event(self, event: Event) -> None:
        """Добавляет событие в список"""
        self.events.append(event)

    def _calculate_intervals(self) -> pd.DataFrame:
        """
        Вычисляет интервалы между циклами "уход" -> "подход".
        Возвращает датафрейм с событиями и интервалами.
        """
        df = pd.DataFrame(
            ((event.type.value, event.timestamp_sec) for event in self.events),
            columns=['event', 'timestamp']
        )
        # Отфильтровываем события approach (они не нужны для расчета интервалов)
        df = df[df['event'] != TableEventEnum.approach.value]
        # Убираем последовательные одинаковые события (нам нужны моменты смены состояния)
        df = df.loc[df['event'].shift(-1) != df['event']]
        # Вычисление разницы между временными метками событий
        df['interval'] = df['timestamp'].diff()

        return df

    def print_report(self) -> None:
        """Выводит отчет о времени между циклами"""
        df = self._calculate_intervals()
        interval_col = df['interval']
        interval_count = interval_col.count()

        if interval_count > 0:
            avg_interval = interval_col.mean()
            print(f'Среднее время между уходом гостя и подходом следующего человека: {avg_interval:.2f} сек.')
            print(f'Общее количество циклов ухода и подхода к столу: {interval_count}')
            print(f'Интервалы (в секундах): {", ".join(f"{i:.2f}" for i in interval_col.dropna())}')
        else:
            print('Не обнаружено никаких полных циклов ухода и подхода к столу')


def main() -> None:
    ap = ArgumentParser(description='Мониторинг занятости стола на видео', prefix_chars='-', add_help=True)
    ap.add_argument('-v', '--video', dest='video_path',
                    required=True, help='Путь к файлу видеозаписи', type=str)
    args = ap.parse_args()
    video_path = Path(args.video_path)

    if not video_path.exists():
        print(f'Ошибка: "{video_path}" не существует в файловой системе.')
        return

    try:
        with VideoProcessor(video_path) as processor:
            fps: int = processor.fps
            frame = processor.first_frame
            writer = VideoWriter(Path('./output.mp4'), processor.fps, processor.frame_resolution)

            model = YOLO('yolov8n.pt')
            person_detector = PersonDetector(model)

            table_analytics = TableAnalytics()
            rect_area = RoiSelector.select(frame)

            i_frame: int = 0

            try:
                for frames in tqdm(processor.iter_batches(), desc=f'Анализ {video_path}'):
                    results = person_detector.detect_in_frame(frames, rect_area)
                    for frame, event in zip(frames, results):
                        timestamp_sec: int = i_frame // fps

                        if event == TableEventEnum.taken:
                            color = (0, 0, 255)
                            text = 'Стол занят'
                        elif event == TableEventEnum.approach:
                            color = (0, 255, 255)
                            text = 'Подход к столу'
                        else:
                            color = (0, 255, 0)
                            text = 'Стол пустой'

                        table_analytics.add_event(Event(type=event, timestamp_sec=timestamp_sec))
                        writer.draw_rect(frame, rect_area, color)
                        frame = writer.put_text(frame, text, (rect_area.x, rect_area.y + rect_area.height), color)
                        writer.write(frame)
                        i_frame += 1
            except KeyboardInterrupt:
                print('Прерывание анализа видеозаписи')
            finally:
                writer.release()

            table_analytics.print_report()
    except Exception as e:
        print(f'Ошибка: {e}')


if __name__ == '__main__':
    main()
