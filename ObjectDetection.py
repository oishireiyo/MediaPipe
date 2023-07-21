# Standard modules
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
import cv2

class ObjectDetection(object):
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
        base_options = mp.tasks.BaseOptions
        object_detector = mp.tasks.vision.ObjectDetector
        object_detector_options = mp.tasks.vision.ObjectDetectorOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = object_detector_options(
            base_options = base_options(model_asset_path = 'TrainedModels/efficientdet_lite0.tflite'),
            max_results = 5,
            running_mode = vision_running_mode.VIDEO,
        )

        self.detector = object_detector.create_from_options(options)

    def check_one_frame(self, iframe: int) -> np.ndarray:
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_video.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.detector.detect_for_video(mp_frame, iframe)

            for detection in results.detections:
                # Draw bounding-boxes
                bbox = detection.bounding_box
                point_lt = (bbox.origin_x, bbox.origin_y)
                point_rb = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, point_lt, point_rb, (255, 0, 0), 3)

                # Draw 
                category = detection.categories[0]
                category_name = category.category_name
                category_score = category.score
                result_text = category_name + ' (' + str(category_score) + ')'
                text_location = (10 + bbox.origin_x, 20 + 10 + bbox.origin_y)
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        return frame

    def check_given_frames(self, tuple: tuple) -> None:
        for i in tuple:
            frame = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def check_all_frames(self) -> None:
        for i in range(self.length):
            frame = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def release_all(self):
        self.input_video.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Solokatsu.mp4'
    output_video = '../Outputs/Solokatsu_ObjectDetector.mp4'

    Detector = ObjectDetection(input_video=input_video, output_video=output_video)
    Detector.check_all_frames()
    Detector.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))