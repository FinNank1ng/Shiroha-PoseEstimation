# -*- coding: utf-8 -*-
import asyncio
import cv2
import numpy as np
import json
import logging
import fractions
import socket

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from concurrent.futures import ThreadPoolExecutor

from server.shm_manager import ShmManager
from pose.body_pose import BodyPose
from face.head_pose import HeadPose
from rig.skeleton import SkeletonSolver
from render.single_stickman_renderer import SingleStickmanRenderer
from render.multi_stickman_renderer import MultiStickmanRenderer
from render.fall_detector_renderer import FallDetectorRenderer
from render.face_recognition_renderer import FaceRecognitionRenderer
from render.intrusion_detection_renderer import IntrusionDetectionRenderer
from render.loitering_detection_renderer import LoiteringDetectionRenderer
from render.static_detection_renderer import StaticDetectionRenderer
from render.vigorous_activity_renderer import VigorousActivityRenderer
from render.activity_level_renderer import ActivityLevelRenderer
from analysis.fall_detector import FallDetector
from analysis.face_recognition_analyzer import FaceRecognitionAnalyzer
from rig.face_solver import FaceSolver, MODE_SINGLE, MODE_MULTI

# 基础配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtc_main")
# 9个窗口并行，线程池设为 12 保证调度顺滑
executor = ThreadPoolExecutor(max_workers=12)

# 读取配置文件
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 全局初始化组件 (单例模式)
logger.info("正在初始化 AI 模型组件...")

body_single = BodyPose(num_poses=config["pose"]["single"]["num_poses"])
body_multi = BodyPose(num_poses=config["pose"]["multi"]["num_poses"])
head_pose_model = HeadPose()
skeleton_solver = SkeletonSolver(filter_alpha=config["skeleton"]["filter_alpha"])
single_render_node = SingleStickmanRenderer()

fall_analyser = FallDetector(ground_threshold_sec=config["fall_detector"]["ground_threshold_sec"])
fall_render_node = FallDetectorRenderer(fall_analyser)
pose_multi_render_node = MultiStickmanRenderer()

# 基础检测组件
intrusion_render_node = IntrusionDetectionRenderer()
loitering_render_node = LoiteringDetectionRenderer(
    alert_duration=config["detection"]["loitering"]["alert_duration"],
    cycle_length=config["detection"]["loitering"]["cycle_length"],
    alert_threshold=config["detection"]["loitering"]["alert_threshold"]
)
static_render_node = StaticDetectionRenderer(
    history_length=config["detection"]["static"]["history_length"],
    movement_threshold=config["detection"]["static"]["movement_threshold"]
)
vigorous_render_node = VigorousActivityRenderer(
    activity_threshold=config["detection"]["vigorous_activity"]["activity_threshold"]
)
activity_render_node = ActivityLevelRenderer(
    low_threshold=config["detection"]["activity_level"]["low_threshold"],
    high_threshold=config["detection"]["activity_level"]["high_threshold"]
)

face_cfg = config["face"].copy()
face_cfg.pop("mode", None)  # 移除配置中的 mode 防止冲突

face_solver_m = FaceSolver(mode=MODE_MULTI, **face_cfg)
face_render_multi = FaceRecognitionRenderer(face_solver_m)

# 人脸检测器 (OpenCV Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_analyzer = FaceRecognitionAnalyzer()
face_render_multi = FaceRecognitionRenderer(face_solver_instance=None)

class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, mode="type1", shm_name="shiroha_frame"):
        super().__init__()
        self.mode = mode
        self.shm_manager = ShmManager(name=shm_name)
        self.frame_buffer = None
        self._timestamp = 0
        self._fps = 60
        self._clock_rate = 540000

        try:
            self.frame_buffer = self.shm_manager.attach()
        except:
            logger.error(f"Track {mode} 无法挂载共享内存")

    async def recv(self):
        await asyncio.sleep(1 / self._fps)
        pts = self._timestamp
        self._timestamp += int(self._clock_rate / self._fps)

        loop = asyncio.get_event_loop()
        try:
            # 推理逻辑丢进线程池，保证 WebRTC 信令不超时
            final_img = await loop.run_in_executor(executor, self._process_frame)
        except Exception as e:
            logger.error(f"渲染失败: {e}")
            final_img = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = VideoFrame.from_ndarray(final_img, format="bgr24")
        frame.pts = pts
        frame.time_base = fractions.Fraction(1, self._clock_rate)
        return frame

    def _process_frame(self):
        if self.frame_buffer is None:
            return np.zeros((360, 480, 3), dtype=np.uint8)

        # 拷贝原始帧
        raw_frame = self.frame_buffer.copy()
        # 预准备渲染画布
        draw_frame = raw_frame.copy()

        try:
            # 按模式分发算法
            if self.mode == "type1":  # Fall Detector
                res = body_single.detect(raw_frame)
                if res and 'landmark_points' in res:
                    pts = res['landmark_points']
                    skel = skeleton_solver.solve({i: pt for i, pt in enumerate(pts)}, head_pose_model.detect(raw_frame))
                    draw_frame = single_render_node.draw(draw_frame, pts, skel)
                    draw_frame = fall_render_node.draw(draw_frame, skel)

            elif self.mode == "type2":  # Pose Monitoring
                res = body_multi.detect(raw_frame)
                if res and 'people' in res:
                    pts_list = [p['landmark_points'] for p in res['people'] if 'landmark_points' in p]
                    draw_frame = pose_multi_render_node.draw(draw_frame, pts_list)

            elif self.mode == "type3":  # Face Recognition
                faces_data = face_analyzer.solve(raw_frame)
                draw_frame = face_render_multi.draw(draw_frame, faces_data)

            elif self.mode == "type4":  # Intrusion Detection
                res = body_multi.detect(raw_frame)
                if res and 'people' in res:
                    pts_list = [p['landmark_points'] for p in res['people'] if 'landmark_points' in p]
                    draw_frame = intrusion_render_node.draw(draw_frame, pts_list)

            elif self.mode == "type5":  # Loitering Detection
                res = body_multi.detect(raw_frame)
                if res and 'people' in res:
                    pts_list = [p['landmark_points'] for p in res['people'] if 'landmark_points' in p]
                    draw_frame = loitering_render_node.draw(draw_frame, pts_list)

            elif self.mode == "type6":  # Static Detection
                res = body_multi.detect(raw_frame)
                if res and 'people' in res:
                    pts_list = [p['landmark_points'] for p in res['people'] if 'landmark_points' in p]
                    draw_frame = static_render_node.draw(draw_frame, pts_list)

            elif self.mode == "type7":  # Vigorous Activity
                res = body_single.detect(raw_frame)
                pts = res.get('landmark_points') if res else None
                draw_frame = vigorous_render_node.draw(draw_frame, pts)

            elif self.mode == "type8":  # Activity Level
                res = body_single.detect(raw_frame)
                pts = res.get('landmark_points') if res else None
                draw_frame = activity_render_node.draw(draw_frame, pts)

            elif self.mode == "type9":  # RAW / Skeleton
                cv2.putText(draw_frame, "ENGINE RAW STATUS: OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)

        except Exception as e:
            logger.error(f"Mode {self.mode} 运行报错: {e}")

        # 统一窗口分辨率，降低前端 9 屏渲染压力
        return cv2.resize(draw_frame, (640, 480))


# 信令服务器 (Signaling)
pcs = set()


async def offer(request):
    params = await request.json()
    offer_obj = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    track = VideoProcessorTrack(mode=params.get("mode", "type1"), shm_name=params.get("shm_name", "shiroha_frame"))
    pc.addTrack(track)
    await pc.setRemoteDescription(offer_obj)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def options_handler(request):
    return web.Response(status=200)

def get_preferred_ip():

    """
    获取本机首选 IP 地址（即连接互联网时使用的网卡 IP）
    原理：尝试连接一个公网地址，会自动选择路由，获取这个路由的源 IP
    """
    try:
        # 创建一个 UDP 套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 尝试连接但只确定 Google DNS
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 如果完全没网，回退到本地回环
        return "127.0.0.1"


# CORS 中间件，允许跨域
@web.middleware
async def cors_middleware(request, handler):
    # 如果是 OPTIONS 预检请求，直接返回成功
    if request.method == "OPTIONS":
        return web.Response(status=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        })

    # 处理正常请求
    response = await handler(request)
    # 在响应头里加上允许跨域的标记
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == "__main__":
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post("/offer", offer)
    app.router.add_options("/offer", options_handler)
    # 智能 IP 选择逻辑
    server_cfg = config.get("server", {})
    final_host = server_cfg.get("bind_host")
    if not final_host:
        # 如果配置是 null 或不存在，则自动探测
        final_host = get_preferred_ip()
        logger.info(f"🌍 未指定 bind_host，已自动探测当前网络 IP: {final_host}")
    else:
        logger.info(f"🔒 已根据配置强制绑定 IP: {final_host}")

    logger.info(f"GO GO GO!  WebRTC Server 矩阵中心已启动 @ {final_host}:{server_cfg.get('port', 8888)}")
    web.run_app(app, host=final_host, port=server_cfg.get('port', 8888))