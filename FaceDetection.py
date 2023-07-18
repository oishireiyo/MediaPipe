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

class FaceDetection(object):
    def __init__(self, inputvideo: str, outputvideo: str, videogen: bool = True) -> None:
        # Input video attributes
        self.inputvideoname = inputvideo
        self.inputvideo = cv2.VideoCapture(inputvideo)

        self.length = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (self.width, self.height)

        fps = self.inputvideo.get(cv2.CAP_PROP_FPS)

        # Output video attributes
        if videogen:
            self.outputvideoname = outputvideo
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.outputvideo = cv2.VideoWriter(outputvideo, fourcc, fps, size)

        # Media Pipe
        BaseOptions = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/blaze_face_short_range.tflite')
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options = BaseOptions,
            running_mode = VisionRunningMode.VIDEO,
        )

        self.detector = FaceDetector.create_from_options(options)

    def ConvertToPixelCoordinates(self, x: float, y: float) -> any:
        def IsValidNormalizedValue(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and \
                (value < 1 or math.isclose(1, value))
        
        if not (IsValidNormalizedValue(x) and IsValidNormalizedValue(y)):
            return None
        
        x_px = min(math.floor(x * self.width), self.width - 1)
        y_px = min(math.floor(y * self.height), self.height - 1)
        return (x_px, y_px)

    def CheckOneFrame(self, iframe: int) -> np.ndarray:
        self.inputvideo.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.inputvideo.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.detector.detect_for_video(mp_frame, iframe)

            for detection in results.detections:
                # Draw bounsing-boxes
                bbox = detection.bounding_box
                point_lt = (bbox.origin_x, bbox.origin_y)
                point_rb = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, point_lt, point_rb, (255, 0, 0), 3)

                # Draw key-points
                for keypoint in detection.keypoints:
                    keypoint_pxs = self.ConvertToPixelCoordinates(keypoint.x, keypoint.y)
                    cv2.circle(frame, keypoint_pxs, 2, (0, 255, 0), 2)

                # Draw label and score
                category = detection.categories[0]
                category_name = '' if category.category_name is None else category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (10 + bbox.origin_x,
                                 20 + 10 + bbox.origin_y)
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 0, 0), 1)

        return frame

    def CheckGivenFrames(self, tuple: tuple) -> None:
        for i in tuple:
            frame = self.CheckOneFrame(iframe=i)
            self.outputvideo.write(frame)
    
    def CheckAllFrames(self) -> None:
        for i in range(self.length):
            frame = self.CheckOneFrame(iframe=i)
            self.outputvideo.write(frame)

    def ReleaseAll(self) -> None:
        self.outputvideo.release()
        self.inputvideo.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    inputvideo = '../Inputs/Solokatsu.mp4'
    outputvideo = '../Outputs/Solokatsu_FaceDetector.mp4'

    Detector = FaceDetection(inputvideo=inputvideo, outputvideo=outputvideo, videogen=True)
    Detector.CheckAllFrames()
    Detector.ReleaseAll()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))