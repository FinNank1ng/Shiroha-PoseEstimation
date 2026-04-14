import cv2

class SingleStickmanRenderer:
    BONES = [
        (11, 13), (13, 15),  # 左手
        (12, 14), (14, 16),  # 右手
        (11, 12),            # 肩
        (11, 23), (12, 24),  # 躯干
        (23, 24),            # 髋
        (23, 25), (25, 27),  # 左腿
        (24, 26), (26, 28)   # 右腿
    ]

    def __init__(self):
        self.bone_color = (0, 255, 0)
        self.joint_color = (0, 200, 255)
        self.head_color = (255, 0, 0)
        self.line_thickness = 2
        self.joint_radius = 3

    def draw(self, frame, body, sk, visibility_threshold=0.5):
        """
        绘制单个人的骨架。

        Args:
            frame: 输入图像帧
            body: 关键点列表。
                  如果是 MediaPipe 原始数据，可能是 landmark 对象列表；
                  如果是处理后的数据，应该是 [(x,y,vis), ...] 或 [(x,y), ...]。
            sk: 解算后的骨架数据（包含头部等信息）。
            visibility_threshold: 置信度阈值，低于此值不绘制。

        Returns:
            绘制后的图像帧。
        """
        if body is None:
            return frame

        dbg = frame

        # 辅助函数：获取坐标和置信度
        def get_pt(idx):
            if idx >= len(body): return None, 0
            pt = body[idx]
            # 兼容两种格式：(x,y,vis) 或 (x,y)
            vis = pt[2] if len(pt) > 2 else 1.0
            return (int(pt[0]), int(pt[1])), vis

        # 画骨骼 (带置信度检查)
        for a, b in self.BONES:
            pt_a, vis_a = get_pt(a)
            pt_b, vis_b = get_pt(b)

            # 只有当两个点都可见时，才画线
            if (pt_a and pt_b and
                vis_a > visibility_threshold and
                vis_b > visibility_threshold):
                cv2.line(dbg, pt_a, pt_b, self.bone_color, self.line_thickness)

        # 画关节点
        for i in range(len(body)):
            pt, vis = get_pt(i)
            if pt and vis > visibility_threshold:
                cv2.circle(dbg, pt, self.joint_radius, self.joint_color, -1)

        # 头部框
        if sk and sk.get("head"):
            x, y = sk["head"]
            cv2.rectangle(dbg, (x-20, y-20), (x+20, y+20), self.head_color, 2)

        return dbg