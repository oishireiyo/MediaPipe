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
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

class FaceDirection(object):
    def __init__(self) -> None:
        # Input video attributes
        self.capture = cv2.VideoCapture(1)
        if self.capture.isOpened():
            logger.info('Video is read.')
        else:
            logger.error('Video is not read.')
            sys.exit(1)

        # Media Pipe
        base_options = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/face_landmarker.task')
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        # Create a face landmarker instance with the live stream mode:
        options = face_landmarker_options(
            base_options = base_options,
            running_mode = vision_running_mode.LIVE_STREAM,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            num_faces = 1,
            result_callback = self.result_callback,
        )

        self.landmarker = face_landmarker.create_from_options(options)
        self.landmarks = None
        self.blendshapes = None

    def result_callback(self, result: mp.tasks.vision.FaceLandmarkerResult,
                        utput_image: mp.Image, timestamp_ms: int):
        
        logger.info(result)

        self.landmarks = result.face_landmarks
        self.blendshapes = result.face_blendshapes

    def decorate_frame(self, frame):

        logger.info(self.landmarks)

        return frame

        face_landmarks_list = self.landmarks

        # Loop through the detected faces
        for face_landmarks in face_landmarks_list:

            # Draw the face landmarks
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x = landmark.x, y = landmark.y, z = landmark.z
                ) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks_proto,
                connections = solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks_proto,
                connections = solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )

            solutions.drawing_utils.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks_proto,
                connection = solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None,
                connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return frame

    def check_one_frame(self, iframe):
        ret, frame = self.capture.read()
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            self.landmarker.detect_async(mp_frame, iframe)
            self.decorate_frame(frame=frame)

        return frame

    def check_all_frames(self):
        iframe = 0
        while(True):
            logger.info('Frame No.%d' % (iframe))
            frame = self.check_one_frame(iframe=iframe)
            cv2.imshow('frame', frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')): break
            iframe += 1
            if iframe == 2:
                break

    def release_all(self) -> None:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_time = time.time()

    face_direction = FaceDirection()
    face_direction.check_all_frames()
    face_direction.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))