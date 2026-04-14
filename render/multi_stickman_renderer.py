import cv2
import numpy as np
from .single_stickman_renderer import SingleStickmanRenderer


class MultiStickmanRenderer:
    def __init__(self):
        self.single_renderer = SingleStickmanRenderer()
        self.bone_color = (0, 255, 0)
        self.joint_color = (0, 200, 255)
        self.box_color = (255, 0, 0)
        self.id_bg_color = (0, 0, 0)
        self.id_text_color = (255, 255, 255)
        self.aim_line_color = (255, 0, 0)

    def draw(self, frame, multi_body_data):
        if frame is None or not multi_body_data:
            return frame

        # 创建一个用于绘制半透明背景的图层（只用于画文字背景）
        overlay = frame.copy()

        for i, body_pts_list in enumerate(multi_body_data):
            if not body_pts_list:
                continue

            # 预处理坐标
            # 过滤掉 None 并转为 int
            valid_points = [(int(pt[0]), int(pt[1])) for pt in body_pts_list if pt is not None]

            if len(valid_points) < 2:
                continue

            points_np = np.array(valid_points, dtype=np.int32)

            # 计算最小外接矩形
            rect = cv2.minAreaRect(points_np)
            box = cv2.boxPoints(rect).astype(np.int32)

            # 绘制外接框
            cv2.drawContours(frame, [box], 0, self.box_color, 2)

            # 绘制 ID 标签 (优化版：局部混合，不影响全图性能)
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            id_text = f"User{i + 1}"

            # 获取文字大小
            (text_width, text_height), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 定义背景矩形区域
            bg_top_left = (x_min - 15, y_min - 10)
            bg_bottom_right = (x_min - 15 + text_width + 5, y_min - 10 + text_height + 5)

            # 在 overlay 上画实心矩形
            cv2.rectangle(overlay, bg_top_left, bg_bottom_right, self.id_bg_color, -1)

            # 全图混合，或者使用局部混合
            # 直接画个半透明矩形在 frame 上
            # 直接在 frame 上画矩形会覆盖像素，所以用 overlay 技巧
            pass

        cv2.addWeighted(overlay, 0.6, frame, 1, 0, frame)

        # 再循环绘制文字、骨架和连线，直接画在混合后的 frame 上
        for i, body_pts_list in enumerate(multi_body_data):
            if not body_pts_list: continue
            valid_points = [(int(pt[0]), int(pt[1])) for pt in body_pts_list if pt is not None]
            if len(valid_points) < 2: continue

            points_np = np.array(valid_points, dtype=np.int32)
            rect = cv2.minAreaRect(points_np)
            box = cv2.boxPoints(rect).astype(np.int32)

            # 重新计算位置用于画文字
            x_min = min([p[0] for p in box])
            y_min = min([p[1] for p in box])
            id_text = f"User{i + 1}"
            (text_width, text_height), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_pos = (x_min - 15, y_min - 10)

            # 画文字
            cv2.putText(frame, id_text, (label_pos[0] + 2, label_pos[1] + text_height + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.id_text_color, 1)

            # 画连线 (从文字中心到框)
            text_center = (label_pos[0] + text_width // 2, label_pos[1] + text_height // 2)
            cv2.line(frame, text_center, box[0], self.aim_line_color, 1, cv2.LINE_AA)

            # 画骨架和关节
            for a, b in SingleStickmanRenderer.BONES:
                if a < len(body_pts_list) and b < len(body_pts_list):
                    pt_a, pt_b = body_pts_list[a], body_pts_list[b]
                    if pt_a is not None and pt_b is not None:
                        cv2.line(frame, (int(pt_a[0]), int(pt_a[1])), (int(pt_b[0]), int(pt_b[1])), self.bone_color, 1)

            for p in body_pts_list:
                if p is not None:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, self.joint_color, -1)

        return frame