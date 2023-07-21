A# Standard modules
import os
import sys
import math
import time

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2

class PoseLandmarks(object):
    def __init__(self, input_video: str, output_video: str, video_gen: str = True) -> None:
        # Input video attributes
        self.input_video_name = input_video
        self.input_video = cv2.VideoCapture(input_video)

        self.length = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)

        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)

        # Output video attributes
        if video_gen:
            self.output_video_name = output_video
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.output_video = cv2.VideoWriter(output_video, fourcc, self.fps, self.size)

        # Media Pipe
        base_options = mp.tasks.BaseOptions(model_asset_path = 'trainedModels/pose_landmarker_full.task')
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = pose_landmarker_options(
            base_options = base_options,
            running_mode = vision_running_mode.VIDEO,
        )

        self.landmarker = pose_landmarker.create_from_options(options)

    def _decorate_landmarks_pose_landmarks(self, frame: np.ndarray, results) -> None:
        # Pose landmarks
        pose_landmarks_list = results.pose_landmarks
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in pose_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = pose_landmarks_proto,
                connections = solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec = solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    def _decorate_landmarks_pose_world_landmarks(self, frame: np.ndarray, results) -> None:
        # Pose world landmarks
        pose_world_landmarks_list = results.pose_world_landmarks
        for idx in range(len(pose_world_landmarks_list)):
            pose_world_landmarks = pose_world_landmarks_list[idx]

            # Draw the pose world landmarks.
            pose_world_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_world_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in pose_world_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = pose_world_landmarks_proto,
                connections = solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec = solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    def decorate_landmarks(self, frame: np.ndarray, results) -> None:
        frame = self._decorate_landmarks_pose_landmarks(frame, results)

    def check_one_frame(self, iframe: int) -> None:
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_video.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.landmarker.detect_for_video(mp_frame, iframe)

            self.decorate_landmarks(frame, results)

        return frame

    def check_given_frames(self, tuple: tuple) -> None:
        for i in range(*tuple):
            frame = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def check_all_frames(self) -> None:
        for i in range(self.length):
            frame = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def release_all(self):
        self.output_video.release()
        self.input_video.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Solokatsu.mp4'
    output_video = '../Outputs/Solokatsu_PoseLandmarker.mp4'

    Landmarker = PoseLandmarks(input_video=input_video, output_video=output_video)
    Landmarker.check_all_frames()
    Landmarker.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))