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

class FaceDirections(object):
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
        base_options = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/face_landmarker.task')
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options = base_options,
            running_mode = vision_running_mode.VIDEO,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            num_faces = 1,
        )

        self.landmarker = face_landmarker.create_from_options(options)

    def decorate_frame(self, frame, results):
        face_landmarks_list = results.face_landmarks

        # Loop through the detected faces
        for face_landmarks in face_landmarks_list:

            # Draw the face landmarks
            face_landmarks_proto = landmark_pb2.NoemalizedLandmarkList()
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

        return frame

    def check_one_frame(self, iframe: int):
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_video.read()
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.landmarker.detect_for_video(mp_frame, iframe)
            frame = self.decorate_frame(frame=frame, results=results)

        return frame
    
    def check_all_frames(self) -> None:
        for i in range(self.length):
            frame = self.check_one_frame(iframe=i)
            self.output_video.write(frame)
            break

    def release_all(self) -> None:
        self.output_video.release()
        self.input_video.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Videos/Solokatsu.mp4'
    output_video = '../Outputs/Videos/Solokatsu_FaceDirectionByDetection.mp4'

    Landmarker = FaceDirections(input_video=input_video, output_video=output_video)
    Landmarker.check_all_frames()
    Landmarker.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))