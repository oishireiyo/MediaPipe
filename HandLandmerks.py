# Standard python modules
import os
import sys
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

class HandLandmarker(object):
    def __init__(self, input_video: str, output_video: str, video_gen: bool = True) -> None:
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
        base_options = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/hand_landmarker.task')
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = hand_landmarker_options(
            base_options = base_options,
            running_mode = vision_running_mode.VIDEO,
        )

        self.landmarker = hand_landmarker.create_from_options(options)

    def decorate_landmarks(self, frame: np.ndarray, results):
        hand_landmarks_list = results.hand_landmarks
        handedness_list = results.handedness
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = hand_landmarks_proto,
                connections = solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec = solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec = solutions.drawing_styles.get_default_hand_connections_style(),
            )

            # Get the top left corner of the detected hand's bounding box
            height, width, _ = frame.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            test_x = int(min(x_coordinates) * width)
            test_y = int(min(y_coordinates) * height)

            # Draw handedness (left ot right hand) on the image.
            cv2.putText(frame, f'{handedness[0].category_name}', (test_x, test_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)

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
    output_video = '../Outputs/Solokatsu_HandLandmarker.mp4'

    Landmarker = HandLandmarker(input_video=input_video, output_video=output_video)
    Landmarker.check_all_frames()
    Landmarker.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))