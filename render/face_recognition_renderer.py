# -*- coding: utf-8 -*-
import cv2


class FaceRecognitionRenderer:
    def __init__(self,face_solver_instance=None):
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        # self.solver = face_solver_instance
        self.high_confidence_threshold = 0.4

    def draw(self, frame, processed_faces):
        if frame is None or not processed_faces:
            return frame

        h, w, _ = frame.shape

        for face_info in processed_faces.values():
            bbox = face_info.get('bbox')
            if not bbox:
                continue

            x, y, bw, bh = map(int, bbox)
            top_left = (x, y)
            bottom_right = (x + bw, y + bh)

            # 逻辑判断
            name = face_info.get('name', 'Unknown')
            distance = face_info.get('distance', 1.0)

            if name == "Unknown":
                color = (0, 0, 255)  # 红色：陌生人
                label_text = "STRANGER"
            elif distance < self.high_confidence_threshold:
                color = (0, 255, 0)  # 绿色：高匹配
                label_text = f"{name}"
            else:
                color = (0, 255, 255)  # 黄色：低匹配
                label_text = f"{name}?"

            # 绘制矩形框
            cv2.rectangle(frame, top_left, bottom_right, color, self.thickness)

            # 计算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(label_text, self.font, self.font_scale,
                                                                  self.font_thickness)

            # 确定文字位置 (画在框的顶部)
            # 文字背景底部对齐 y，所以背景顶部是 y - text_height - padding
            padding = 5
            label_top_y = y - text_height - padding
            label_bottom_y = y + padding  # 稍微盖住一点框的顶部线，视觉更整体

            # 边界检查，如果人脸在屏幕最顶端，文字画不下，就画在框内部
            if label_top_y < 0:
                label_top_y = y
                label_bottom_y = y + text_height + padding
                text_y = label_top_y + text_height + baseline
            else:
                text_y = label_top_y + text_height + baseline

            # 绘制文字背景
            bg_top_left = (x, label_top_y)
            bg_bottom_right = (x + text_width + padding * 2, label_bottom_y)
            cv2.rectangle(frame, bg_top_left, bg_bottom_right, color, -1)

            # 绘制文字
            cv2.putText(frame, label_text, (x + padding, text_y - baseline),
                        self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        return frame