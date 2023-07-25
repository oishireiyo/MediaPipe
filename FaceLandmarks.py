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

class FaceLandmark(object):
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
        BaseOptions = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/face_landmarker.task')
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options = BaseOptions,
            running_mode = VisionRunningMode.VIDEO,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            num_faces = 1,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def DrawLandmarks(self, frame, results):
        face_landmarks_list = results.face_landmarks

        # Loop through the detected faces
        for face_landmarks in face_landmarks_list:
            
            # Draw the face landmarks
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z) for landmark in face_landmarks
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
                connections = solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = None,
                connection_drawing_spec = solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        
        return frame

    def CheckOneFrame(self, iframe: int):
        self.inputvideo.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.inputvideo.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.landmarker.detect_for_video(mp_frame, iframe)
            frame = self.DrawLandmarks(frame, results=results)

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

    inputvideo = '../Inputs/Videos/Solokatsu.mp4'
    outputvideo = '../Outputs/Videos/Solokatsu_FaceLandmarker.mp4'

    Landmarker = FaceLandmark(inputvideo=inputvideo, outputvideo=outputvideo)
    Landmarker.CheckAllFrames()
    Landmarker.ReleaseAll()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))