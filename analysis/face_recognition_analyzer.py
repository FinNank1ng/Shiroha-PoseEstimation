import dlib
import numpy as np
import cv2
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)


class FaceRecognitionAnalyzer:
    def __init__(self):
        logger.info("正在初始化人脸识别分析器...")
        # 0 是上采样次数，通常 0 即可，速度快
        self.detector = dlib.get_frontal_face_detector()

        try:
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
            logger.info("Dlib 模型加载成功")
        except RuntimeError as e:
            logger.error(f"错误：无法加载 Dlib 模型文件，请检查路径。详情: {e}")
            return

        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        self.frame_count = 0
        self.recent_results = {}  # 用于暂存上一帧的结果，避免重复计算
        self.recognition_interval = 15  # 每 15 帧进行一次人脸识别

    def load_known_faces(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        known_faces_dir = os.path.join(current_dir, "known_faces")

        if not os.path.exists(known_faces_dir):
            logger.error(f"  - 未找到人脸存放库: {known_faces_dir}")
            return

        logger.info("正在加载已知人脸库...")
        for file_name in os.listdir(known_faces_dir):
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            path = os.path.join(known_faces_dir, file_name)

            try:
                img = dlib.load_rgb_image(path)
                faces = self.detector(img, 0)
                if len(faces) > 0:
                    # 取最大的人脸
                    face = max(faces, key=lambda rect: rect.width() * rect.height())
                    shape = self.predictor(img, face)
                    face_encoding = self.face_rec_model.compute_face_descriptor(img, shape)
                    self.known_face_encodings.append(np.array(face_encoding))
                    self.known_face_names.append(file_name.split('.')[0])
                    logger.debug(f"  - 加载成功: {file_name}")
            except Exception as e:
                logger.error(f"  - 加载失败 {file_name}: {e}")

        logger.info(f"人脸库加载完成，共 {len(self.known_face_names)} 人。")

    def solve(self, frame):
        """
        分析帧，返回处理后的数据
        """
        self.frame_count += 1
        results = {}

        # 转灰度图用于检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 彩色图用于提取特征（如果需要识别）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测人脸位置
        rects = self.detector(gray, 0)
        current_centers = {}

        # 1. 先进行人脸检测，确定框的位置
        for i, rect in enumerate(rects):
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            center_x, center_y = x + w // 2, y + h // 2
            current_centers[i] = (center_x, center_y)

            # 初始化基础结果
            results[i] = {
                'bbox': [x, y, w, h],
                'name': "Unknown",
                'distance': 1.0
            }

        # 识别逻辑
        if self.frame_count % self.recognition_interval == 0:
            # 只有当有人脸且有人脸库时才执行
            if len(rects) > 0 and len(self.known_face_encodings) > 0:
                for i, rect in enumerate(rects):
                    # 获取当前人脸的关键点
                    shape = self.predictor(rgb_frame, rect)
                    face_encoding = self.face_rec_model.compute_face_descriptor(rgb_frame, shape)
                    face_encoding_np = np.array(face_encoding)

                    # 计算当前人脸与库中所有人脸的欧氏距离
                    distances = np.linalg.norm(np.array(self.known_face_encodings) - face_encoding_np, axis=1)
                    min_index = np.argmin(distances)
                    min_distance = distances[min_index]

                    # 设定阈值
                    if min_distance < 0.6:
                        results[i]['name'] = self.known_face_names[min_index]
                        results[i]['distance'] = float(min_distance)
                        logger.debug(f"识别到: {self.known_face_names[min_index]}, 距离: {min_distance:.2f}")
                    else:
                        logger.debug(f"未识别 (距离过远: {min_distance:.2f})")

        # 3. 跟踪逻辑：如果不是识别帧，尝试从上一帧继承结果
        else:
            for curr_id, curr_center in current_centers.items():
                min_move_dist = float('inf')
                best_prev_id = -1

                for prev_id, prev_data in self.recent_results.items():
                    prev_center = (prev_data['bbox'][0] + prev_data['bbox'][2] // 2,
                                   prev_data['bbox'][1] + prev_data['bbox'][3] // 2)
                    dist = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                    if dist < min_move_dist and dist < 100:
                        min_move_dist = dist
                        best_prev_id = prev_id

                if best_prev_id != -1:
                    # 沿用上一帧的识别结果
                    results[curr_id]['name'] = self.recent_results[best_prev_id]['name']
                    results[curr_id]['distance'] = self.recent_results[best_prev_id]['distance']

        # 更新缓存并返回
        self.recent_results = results.copy()
        return results