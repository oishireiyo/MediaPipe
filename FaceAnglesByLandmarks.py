# Standard python modules
import os
import sys
import time

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

# PnP
sys.path.append(os.pardir)
from OpenCV.PerspectiveNPoints import PerspectiveNPoints as PnP

COLOR_BLACK  = (0,   0,   0  )
COLOR_BLUE   = (255, 0,   0  )
COLOR_GREEN  = (0,   255, 0  )
COLOR_RED    = (0,   0,   255)
COLOR_CYAN   = (255, 255, 0  )
COLOR_PINK   = (255, 0,   255)
COLOR_YELLOW = (0,   255, 255)
COLOR_WHITE  = (255, 255, 255)

class FaceLandmarkForVideo(object):
    def __init__(self, input_video: str, output_video: str):
        # Input video attributes
        self.input_video_name = input_video
        self.input_video = cv2.VideoCapture(input_video)

        self.length = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (self.width, self.height)

        fps = self.input_video.get(cv2.CAP_PROP_FPS)

        # Output video attributes
        self.output_video_name = output_video
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.output_video = cv2.VideoWriter(output_video, fourcc, fps, size)

        # Media Pipe
        base_options = mp.tasks.BaseOptions(
            model_asset_path = 'TrainedModels/face_landmarker.task')
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

        # Perspective-N-Points
        self.PnP = PnP(width = self.width, height = self.height)

    def decorate_frame(self, frame, landmarks):
        self.PnP.parse_detected_facial_points_2d(landmarks = landmarks)
        points_2d_cal, projected_points_2d_cal = self.PnP.project_points_used_in_pnp()

        # Drawing calculation points and projected points
        for p in points_2d_cal:
            cv2.circle(
                frame, (int(p[0]), int(p[1])), 3, COLOR_WHITE, 2)

        for p in projected_points_2d_cal:
            cv2.circle(
                frame, (int(p[0][0]), int(p[0][1])), 3, COLOR_PINK, 2)

        # Drawing X,Y,Z-axis in a frame from the nose point
        given_points_3d = np.array([
            (10,  0,  0),
            ( 0, 10,  0),
            ( 0,  0, 10),
            ( 0,  0,  0),
        ], dtype = np.float32)
        projected_points_2d_cal = self.PnP.project_given_points(points_3d = given_points_3d)
        diff_x = points_2d_cal[0][0] - projected_points_2d_cal[3][0][0]
        diff_y = points_2d_cal[0][1] - projected_points_2d_cal[3][0][1]

        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[0][0][0] + diff_x), int(projected_points_2d_cal[0][0][1] + diff_y)),
            COLOR_BLUE, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[1][0][0] + diff_x), int(projected_points_2d_cal[1][0][1] + diff_y)),
            COLOR_GREEN, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[2][0][0] + diff_x), int(projected_points_2d_cal[2][0][1] + diff_y)),
            COLOR_RED, 4
        )
        cv2.circle(frame, (int(projected_points_2d_cal[3][0][0]), int(projected_points_2d_cal[3][0][1])), 3, COLOR_YELLOW, 2)

        pitch, yaw, roll = self.PnP.get_roll_pitch_yaw()
        cv2.putText(frame, 'Pitch : %.4f' % (pitch), (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)
        cv2.putText(frame, 'Yaw   : %.4f' % (yaw),   (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)
        cv2.putText(frame, 'Roll  : %.4f' % (roll),  (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)

    def check_one_frame(self, iframe: int):
        self.input_video.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        ret, frame = self.input_video.read()
        logger.info('Frame: %4d / %d, read frame: %s' % (iframe, self.length, ret))
        if ret:
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            results = self.landmarker.detect_for_video(mp_frame, iframe)

            if len(results.face_landmarks) > 0:
                self.decorate_frame(frame, results.face_landmarks[0])

        return frame, results

    def check_given_frames(self, tuple: tuple):
        for i in range(*tuple):
            frame, _ = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def check_all_frames(self):
        for i in range(self.length):
            frame, _ = self.check_one_frame(iframe=i)
            self.output_video.write(frame)

    def release_all(self):
        self.output_video.release()
        self.input_video.release()
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

class FaceLandmarkForImage(object):
    def __init__(self, input_image: str, output_image: str) -> None:
        # Input image and its attributes
        self.input_image_name = input_image
        self.input_image = cv2.imread(input_image)
        self.height, self.width, _ = self.input_image.shape

        # Output image
        self.output_image_name = output_image

        # Media pipe
        base_options = mp.tasks.BaseOptions(model_asset_path = 'TrainedModels/face_landmarker.task')
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options = base_options,
            running_mode = vision_running_mode.IMAGE,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
        )

        self.landmarker = face_landmarker.create_from_options(options)

        # Perspective-N-Points
        self.PnP = PnP(width = self.width, height = self.height)

    def decorate_frame(self, frame, landmarks):
        self.PnP.parse_detected_facial_points_2d(landmarks = landmarks)
        points_2d_cal, projected_points_2d_cal = self.PnP.project_points_used_in_pnp()

        # Drawing calculation points and projected points
        for p in points_2d_cal:
            cv2.circle(
                frame, (int(p[0]), int(p[1])), 2, COLOR_WHITE, 2)

        for p in projected_points_2d_cal:
            cv2.circle(
                frame, (int(p[0][0]), int(p[0][1])), 3, COLOR_PINK, 2)

        # Drawing X,Y,Z-axis in a frame from the nose point
        given_points_3d = np.array([
            (10,  0,  0),
            ( 0, 10,  0),
            ( 0,  0, 10),
            ( 0,  0,  0),
        ], dtype = np.float32)
        projected_points_2d_cal = self.PnP.project_given_points(points_3d = given_points_3d)
        diff_x = points_2d_cal[0][0] - projected_points_2d_cal[3][0][0]
        diff_y = points_2d_cal[0][1] - projected_points_2d_cal[3][0][1]

        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[0][0][0] + diff_x), int(projected_points_2d_cal[0][0][1] + diff_y)),
            COLOR_BLUE, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[1][0][0] + diff_x), int(projected_points_2d_cal[1][0][1] + diff_y)),
            COLOR_GREEN, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d_cal[3][0][0] + diff_x), int(projected_points_2d_cal[3][0][1] + diff_y)),
            (int(projected_points_2d_cal[2][0][0] + diff_x), int(projected_points_2d_cal[2][0][1] + diff_y)),
            COLOR_RED, 4
        )
        cv2.circle(frame, (int(projected_points_2d_cal[3][0][0]), int(projected_points_2d_cal[3][0][1])), 3, COLOR_YELLOW, 2)

        pitch, yaw, roll = self.PnP.get_roll_pitch_yaw()
        cv2.putText(frame, 'Pitch : %.4f' % (pitch), (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)
        cv2.putText(frame, 'Yaw   : %.4f' % (yaw),   (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)
        cv2.putText(frame, 'Roll  : %.4f' % (roll),  (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLACK, thickness=2, lineType=2)

        cv2.imshow('Final', frame)
        cv2.waitKey(0)

    def check_image(self):
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = self.input_image)
        results = self.landmarker.detect(mp_frame)

        if len(results.face_landmarks) > 0:
            self.decorate_frame(self.input_image, results.face_landmarks[0])

    def plot_3d_landmarks(self):
        face_landmarks_list = self.check_image().face_landmarks
        Xs = np.array([landmark.x * self.width for landmark in face_landmarks_list[0]])
        Ys = np.array([-landmark.y * self.height for landmark in face_landmarks_list[0]])
        Zs = np.array([-landmark.z * self.width for landmark in face_landmarks_list[0]])

        fig = plt.figure(figsize = plt.figaspect(1.0,))
        fig.suptitle('Face landmarks')

        ax = fig.add_subplot(2, 2, 1, projection = '3d')
        ax.scatter(Zs, Xs, Ys, color = 'blue')

        ax = fig.add_subplot(2, 2, 2)
        ax.scatter(Zs, Xs, color = 'green')

        ax = fig.add_subplot(2, 2, 3)
        ax.scatter(Zs, Ys, color = 'red')

        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(Xs, Ys, color = 'gray')

        plt.savefig(self.output_image_name)

    def release_all(self):
        cv2.destroyAllWindows()
        logger.info('All jobs done.')

if __name__ == '__main__':
    start_time = time.time()

    input_video = '../Inputs/Videos/Solokatsu.mp4'
    output_video = '../Outputs/Videos/Solokatsu_FaceAnglesByLandmarks.mp4'

    landmarker_video = FaceLandmarkForVideo(input_video = input_video,
                                            output_video = output_video)
    landmarker_video.check_all_frames()
    landmarker_video.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))

    # sys.exit(1)

    start_time = time.time()

    # Get face landmarks
    base_name = 'Kanna_Hashimoto'
    # base_name = 'Haruka_Ayase'
    # base_name = 'Gaki'
    input_image = '../Inputs/Images/%s.jpg' % (base_name)
    output_image = '../Outputs/Images/%s_landmark3D.jpg' % (base_name)

    #landmarker = FaceLandmarkForImage(input_image=input_image, output_image=output_image)
    #landmarker.check_image()
    # landmarker.plot_3d_landmarks()
    #landmarker.release_all()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))