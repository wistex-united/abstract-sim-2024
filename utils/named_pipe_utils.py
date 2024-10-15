
from io import BufferedReader, BufferedWriter
import os

import numpy as np


def open_pipe_for_reading(pipe_path: str) -> BufferedReader:
    try:
        os.mkfifo(pipe_path)
    except FileExistsError:
        pass
    return open(pipe_path, "rb")

def open_pipe_for_writing(pipe_path: str) -> BufferedWriter:
    try:
        os.mkfifo(pipe_path)
    except FileExistsError:
        pass
    return open(pipe_path, "wb")

def write_vector_to_pipe(pipe: BufferedWriter, vector: np.ndarray) -> None:
    pipe.write(vector.tobytes())
    pipe.flush()  # Ensure data is written immediately

def read_vector_from_pipe(pipe: BufferedReader, vector_size: int) -> np.ndarray:
    raw_data = pipe.read(vector_size * 4)
    try:
        vector = np.frombuffer(raw_data, dtype=np.float32)
    except ValueError:
        print("Failed to parse:", raw_data)
        raise
    if not vector.shape == (vector_size,):
        raise ValueError(f"Expected {vector_size=} but got {vector.shape=}")
    return vector
