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
import matplotlib.pyplot as plt

COLOR_BLACK  = (0,   0,   0  )
COLOR_BLUE   = (255, 0,   0  )
COLOR_GREEN  = (0,   255, 0  )
COLOR_RED    = (0,   0,   255)
COLOR_CYAN   = (255, 255, 0  )
COLOR_PINK   = (255, 0,   255)
COLOR_YELLOW = (0,   255, 255)
COLOR_WHITE  = (255, 255, 255)

class MediaPipePnPLight(object):
    '''
    PnP = MediaPipePnPLight(width, height)
    PnP.parse_detected_facial_points_2d() # Reset face landmarks
    X = PnP.project_points_used_in_pnp() # For validation
    X = PnP.project_given_points() # For any given points
    X = PnP.get_roll_pitch_yaw() # Get facial angles (roll, pitch, yaw)
    '''
    def __init__(self, width: int, height: int,
                 landmark_indices: list = [
                     1, # Nose
                     33, 263, # Left, right eye
                     61, 291, # Left, right mouth
                     199, # Chin
                     168, # between the eyebrows
                     17, # bottom lip
                     101, 330, # Left, right cheek
                     234, 454, # Left, right ear
                 ]):
        # Features of input image
        self.width = width
        self.height = height

        # Face landmarks (canonical points)
        self.facial_point_file = \
            'canonical_face_model/canonical_face_model.obj'
        self.facial_points_3d = {}
        self._parse_canonical_facial_points_3d()
        self.facial_points_2d = {}

        # Indices for PnP
        self.landmarks_indices = landmark_indices
        self.cam_matrix = np.array([
            [width,     0,  width / 2],
            [    0, width, height / 2],
            [    0,     0,          1],
        ])
        self.distortion_matrix = np.zeros((4, 1))

    def _parse_canonical_facial_points_3d(self):
        logger.info('Parse canonical facial 3D points.')
        with open(self.facial_point_file, mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split()
                if elements[0] == 'v':
                    self.facial_points_3d[len(self.facial_points_3d)] = \
                        (float(elements[1]),
                         float(elements[2]),
                         float(elements[3].replace('\n', '')))

    def parse_detected_facial_points_2d(self, landmarks):
        logger.info('Parse detected facial 2D points.')
        self.facial_points_2d = {}
        for i_landmark, landmark in enumerate(landmarks):
            self.facial_points_2d[i_landmark] = \
                (landmark.x * self.width, landmark.y * self.height)

    def perspective_n_points(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        success, vector_rotation, vector_translation = \
            cv2.solvePnP(points_3d_cal, points_2d_cal, self.cam_matrix, self.distortion_matrix,
                         flags = cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_EPNP or SOLVEPNP_ITERATIVE

        return (success, vector_rotation, vector_translation)

    def project_points_used_in_pnp(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        _, rotation, translation = self.perspective_n_points()
        projected_points_2d_cal, _ = \
            cv2.projectPoints(points_3d_cal, rotation, translation, self.cam_matrix, self.distortion_matrix)

        return (points_2d_cal, projected_points_2d_cal)

    def project_given_points(self, points_3d):
        _, rotation, translation = self.perspective_n_points()
        projected_points_2d_cal, _ = \
            cv2.projectPoints(points_3d, rotation, translation, self.cam_matrix, self.distortion_matrix)

        return projected_points_2d_cal

    def get_roll_pitch_yaw(self):
        from math import pi, atan2, asin
        _, rotation, _ = self.perspective_n_points()
        R = cv2.Rodrigues(rotation)[0]
        roll = 188 * atan2(-R[2][1], R[2][2]) / pi
        pitch = 180 * asin(R[2][0]) / pi
        yaw = 180 * atan2(-R[1][0], R[0][0]) / pi

        return (roll, pitch, yaw)

class MediaPipePnP(object):
    def __init__(self, image_width: int, image_height: int, landmarks,
                 landmark_indices: list = [
                     1, # Nose
                     33, 263, # Left, right eye
                     61, 291, # Left, right mouth
                     199, # Chin
                     168, # between the eyebrows
                     17, # bottom lip
                     101, 330, # Left, right cheek
                     234, 454, # Left, right ear
                ]):
        # Features of input image
        self.width = image_width
        self.height = image_height

        # Face landmarks (canonical points & detected points)
        self.landmarks = landmarks
        self.facial_point_file = 'canonical_face_model/canonical_face_model.obj'
        self.facial_points_3d = {}
        self._parse_canonical_facial_points_3d()
        self.facial_points_2d = {}
        self._parse_detected_facial_points_2d()

        # Indices for PnP
        self.landmarks_indices = landmark_indices
        self.cam_matrix = np.array([
            [image_width,           0,  image_width / 2],
            [          0, image_width, image_height / 2],
            [          0,           0,                1],
        ])
        self.distortion_matrix = np.zeros((4, 1))

    def _parse_canonical_facial_points_3d(self):
        logger.info('Parse canonical facial 3d points')
        with open(self.facial_point_file, mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split()
                if elements[0] == 'v':
                    self.facial_points_3d[len(self.facial_points_3d)] = \
                        (float(elements[1]), float(elements[2]), float(elements[3].replace('\n', '')))

    def _parse_detected_facial_points_2d(self):
        logger.debug('Parse detected facial 2d points')
        for i_landmark, landmark in enumerate(self.landmarks):
            self.facial_points_2d[i_landmark] = (landmark.x * self.width, landmark.y * self.height)

    def perspective_n_points(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        success, vector_rotation, vector_translation = \
            cv2.solvePnP(points_3d_cal, points_2d_cal, self.cam_matrix, self.distortion_matrix,
                         flags = cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_EPNP or SOLVEPNP_ITERATIVE
        
        return (success, vector_rotation, vector_translation)
    
    def project_points_used_in_pnp(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        _, rotation, translation = self.perspective_n_points()
        facial_end_points_2d, _ = \
            cv2.projectPoints(points_3d_cal, rotation, translation, self.cam_matrix, self.distortion_matrix)
        
        return (points_2d_cal, facial_end_points_2d)

    def project_given_points(self, points_3d):
        _, rotation, translation = self.perspective_n_points()
        facial_end_points_2d, _ = \
            cv2.projectPoints(points_3d, rotation, translation, self.cam_matrix, self.distortion_matrix)
        return facial_end_points_2d

    def rot_params_rv(self):
        from math import pi, atan2, asin
        _, rotation, _ = self.perspective_n_points()
        R = cv2.Rodrigues(rotation)[0]
        roll = 188 * atan2(-R[2][1], R[2][2]) / pi
        pitch = 180 * asin(R[2][0]) / pi
        yaw = 180 * atan2(-R[1][0], R[0][0]) / pi
        return (roll, pitch, yaw)

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
        self.PnP = MediaPipePnPLight(width = self.width, height = self.height)

    def decorate_frame(self, frame, landmarks):
        self.PnP.parse_detected_facial_points_2d(landmarks = landmarks)
        points_2d_cal, projected_points_2d = self.PnP.project_points_used_in_pnp()

        # Drawing calculation points and projected points
        for p in points_2d_cal:
            cv2.circle(
                frame,
                (int(p[0]), int(p[1])),
                3, COLOR_WHITE, 2
            )

        for p in projected_points_2d:
            cv2.circle(
                frame,
                (int(p[0][0]), int(p[0][1])),
                3, COLOR_PINK, 2
            )

        # Drawing X,Y,Z-axis in a frame from the nose point
        given_points_3d = np.array([
            (10,  0,  0),
            ( 0, 10,  0),
            ( 0,  0, 10),
            ( 0,  0,  0),
        ], dtype = np.float32)
        projected_points_2d = self.PnP.project_given_points(points_3d = given_points_3d)
        diff_x = points_2d_cal[0][0] - projected_points_2d[3][0][0]
        diff_y = points_2d_cal[0][1] - projected_points_2d[3][0][1]

        cv2.line(
            frame,
            (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
            (int(projected_points_2d[0][0][0] + diff_x), int(projected_points_2d[0][0][1] + diff_y)),
            COLOR_BLUE, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
            (int(projected_points_2d[1][0][0] + diff_x), int(projected_points_2d[1][0][1] + diff_y)),
            COLOR_GREEN, 4
        )
        cv2.line(
            frame,
            (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
            (int(projected_points_2d[2][0][0] + diff_x), int(projected_points_2d[2][0][1] + diff_y)),
            COLOR_RED, 4
        )

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

    def check_image(self):
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = self.input_image)
        results = self.landmarker.detect(mp_frame)
        return results

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

    sys.exit(1)

    start_time = time.time()

    # Get face landmarks
    base_name = 'Kanna_Hashimoto'
    # base_name = 'Haruka_Ayase'
    # base_name = 'Gaki'
    input_image = '../Inputs/Images/%s.jpg' % (base_name)
    output_image = '../Outputs/Images/%s_landmark3D.jpg' % (base_name)
    landmarker = FaceLandmarkForImage(input_image=input_image, output_image=output_image)
    results = landmarker.check_image()

    # Perspective-N-points
    mp_pnp = MediaPipePnP(image_width=landmarker.width, image_height=landmarker.height,
                          landmarks=results.face_landmarks[0])
    points_2d_cal, end_points_2d = mp_pnp.project_points_used_in_pnp()

    for p in points_2d_cal:
        cv2.circle(landmarker.input_image,
                   (int(p[0]), int(p[1])),
                   3, COLOR_WHITE, 2)

    for p in end_points_2d:
        cv2.circle(landmarker.input_image,
                   (int(p[0][0]), int(p[0][1])),
                   3, COLOR_PINK, 2)

    given_points_3d = np.array([
        (10, 0,  0),
        (0, 10,  0),
        (0,  0, 10),
        (0,  0,  0),
    ], dtype = np.float32)
    projected_points_2d = mp_pnp.project_given_points(points_3d = given_points_3d)
    diff_x = points_2d_cal[0][0] - projected_points_2d[3][0][0]
    diff_y = points_2d_cal[0][1] - projected_points_2d[3][0][1]

    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
             (int(projected_points_2d[0][0][0] + diff_x), int(projected_points_2d[0][0][1] + diff_y)),
             COLOR_BLUE, 4)
    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
             (int(projected_points_2d[1][0][0] + diff_x), int(projected_points_2d[1][0][1] + diff_y)),
             COLOR_GREEN, 4)
    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0] + diff_x), int(projected_points_2d[3][0][1] + diff_y)),
             (int(projected_points_2d[2][0][0] + diff_x), int(projected_points_2d[2][0][1] + diff_y)),
             COLOR_RED, 4)

    roll, pitch, yaw = mp_pnp.rot_params_rv()
    cv2.putText(landmarker.input_image,
                text = 'Roll: %.4f, Pitch: %.4f, Yaw: %.4f' % (roll, pitch, yaw),
                org = (int(landmarker.width * 0.05), int(landmarker.height * 0.05)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=COLOR_BLACK,
                thickness=3,
                lineType=cv2.LINE_4)

    cv2.imshow('Final', landmarker.input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))