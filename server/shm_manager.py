# -*- coding: utf-8 -*-
import cv2
import numpy as np
from multiprocessing import shared_memory
import platform
from typing import Optional

class ShmManager:
    def __init__(self, name="shiroha_frame", shape=(480, 640, 3)):
        self.name = name
        self.shape = shape
        # 确保 size 是原生 int，防止 Windows 报错
        self.size = int(np.prod(shape) * np.dtype(np.uint8).itemsize)
        self.shm: Optional[shared_memory.SharedMemory] = None
        self._buffer: Optional[np.ndarray] = None

    def create(self) -> np.ndarray:
        """采集端(Flask)调用：创建共享内存"""
        # Linux 清理逻辑保留
        if platform.system() != "Windows":
            try:
                old_shm = shared_memory.SharedMemory(name=self.name)
                old_shm.close()
                old_shm.unlink()
            except:
                pass

        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            print(f"共享内存创建成功: {self.name} ({self.size} bytes)")
        except FileExistsError:
            # 如果已存在，直接挂载（防止重复创建）
            self.shm = shared_memory.SharedMemory(name=self.name)
            print(f"共享内存已存在，直接挂载: {self.name}")

        self._buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        return self._buffer

    def attach(self) -> np.ndarray:
        """处理端(rtc_main)调用：挂载内存"""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name)
            self._buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
            return self._buffer
        except FileNotFoundError:
            # 向上抛出异常，由逻辑层决定是否重试
            raise FileNotFoundError(f"未找到共享内存 '{self.name}'。请先启动 Flask 程序。")

    def write(self, frame: np.ndarray):
        """将图像帧写入共享内存"""
        if frame is None or self._buffer is None:
            return

        if frame.shape != self.shape:
             return

        # 零拷贝写入
        self._buffer[:] = frame

    def close(self):
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except:
                pass