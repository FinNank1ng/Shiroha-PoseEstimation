import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math


class BodyPose:
    def __init__(self, model_path="models/pose_landmarker_full.task", num_poses=1):

        try:
            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=num_poses
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            print("AI 模型初始化成功 (CPU模式)")

        except Exception as e:
            print(f"模型初始化失败: {e}")
            # 将异常重新抛出，让调用者知道初始化失败
            raise e

        self.timestamp = 0
        self.is_multi_mode = num_poses > 1

    def _calculate_raw_yaw_from_world_landmarks(self, world_landmarks):
        if len(world_landmarks) < 24:
            return None

        # 索引常量
        LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12
        LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24

        try:
            ls = world_landmarks[LEFT_SHOULDER_IDX]
            rs = world_landmarks[RIGHT_SHOULDER_IDX]
            lh = world_landmarks[LEFT_HIP_IDX]
            rh = world_landmarks[RIGHT_HIP_IDX]

            # 计算躯干中心线
            shoulder_center_x = (ls.x + rs.x) / 2.0
            shoulder_center_z = (ls.z + rs.z) / 2.0
            hip_center_x = (lh.x + rh.x) / 2.0
            hip_center_z = (lh.z + rh.z) / 2.0

            # 向量：髋 -> 肩
            dx = shoulder_center_x - hip_center_x
            dz = shoulder_center_z - hip_center_z

            # 计算角度
            # 注意：MediaPipe Z 轴指向摄像头。
            # 如果 dz > 0，说明肩膀比髋部更靠近摄像头（人向后仰或背对）。
            raw_yaw_rad = math.atan2(dz, dx)
            raw_yaw_deg = math.degrees(raw_yaw_rad)

            while raw_yaw_deg > 180: raw_yaw_deg -= 360
            while raw_yaw_deg <= -180: raw_yaw_deg += 360

            return raw_yaw_deg
        except Exception:
            return None

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 时间戳递增，确保视频模式追踪稳定
        self.timestamp += 10

        result = self.detector.detect_for_video(mp_image, self.timestamp)

        people_data = []
        if result.pose_landmarks:
            for idx, pose_landmarks in enumerate(result.pose_landmarks):
                # 提取 2D+深度 坐标 (x, y, z)
                # z 是相对深度，值越小离摄像头越远（MediaPipe 坐标系）
                pts = [(lm.x * w, lm.y * h, lm.z) for lm in pose_landmarks]

                # 计算 3D Yaw
                raw_body_yaw_deg = None
                if result.pose_world_landmarks and idx < len(result.pose_world_landmarks):
                    raw_body_yaw_deg = self._calculate_raw_yaw_from_world_landmarks(result.pose_world_landmarks[idx])

                people_data.append({
                    'landmark_points': pts,
                    'raw_body_yaw': raw_body_yaw_deg
                })

        if self.is_multi_mode:
            return {'people': people_data} if people_data else None
        else:
            return people_data[0] if people_data else None

    def detect_multi(self, frame):
        # 线程安全提示：如果多线程调用，这里修改类变量会有风险
        original_is_multi = self.is_multi_mode
        self.is_multi_mode = True
        result = self.detect(frame)
        self.is_multi_mode = original_is_multi
        return result

    def detect_single(self, frame):
        original_is_multi = self.is_multi_mode
        self.is_multi_mode = False
        result = self.detect(frame)
        self.is_multi_mode = original_is_multi
        return result