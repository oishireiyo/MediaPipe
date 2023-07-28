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

class MediaPipePnP(object):
    def __init__(self, image_width: int, image_height: int, landmarks,
                 landmark_indices: list = [
                     1, # Nose
                     33, 263, # Left, right eye
                     61, 291, # Left, right mouth
                     199, # Chin
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
        logger.info('Parse detected facial 2d points')
        for i_landmark, landmark in enumerate(self.landmarks):
            self.facial_points_2d[i_landmark] = (landmark.x * self.width, landmark.y * self.height)

    def perspective_n_points(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        success, vector_rotation, vector_translation = \
            cv2.solvePnP(points_3d_cal, points_2d_cal, self.cam_matrix, self.distortion_matrix,
                         flags = cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_EPNP or SOLVEPNP_ITERATIVE
        
        return success, vector_rotation, vector_translation
    
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
        _, rotation, _ = self.perspective_n_points()
        from math import pi, atan2, asin
        R = cv2.Rodrigues(rotation)[0]
        roll = 188 * atan2(-R[2][1], R[2][2]) / pi
        pitch = 180 * asin(R[2][0]) / pi
        yaw = 180 * atan2(-R[1][0], R[0][0]) / pi
        return (roll, pitch, yaw)

class FaceLandmark(object):
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
        results = self.check_image()
        face_landmarks_list = results.face_landmarks

        Xs, Ys, Zs = [], [], []
        # for face_landmarks in face_landmarks_list:
        for landmark in face_landmarks_list[0]:
            Xs.append(landmark.x * self.width)
            Ys.append(- landmark.y * self.height)
            Zs.append(- landmark.z * self.width)
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Zs = np.array(Zs)

        fig, axes = plt.subplots(2, 2)

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

    base_name = 'Kanna_Hashimoto'
    base_name = 'Haruka_Ayase'
    input_image = '../Inputs/Images/%s.jpeg' % (base_name)
    output_image = '../Outputs/Images/%s_landmark3D.jpeg' % (base_name)
    landmarker = FaceLandmark(input_image=input_image, output_image=output_image)
    results = landmarker.check_image()

    mp_pnp = MediaPipePnP(image_width=landmarker.width, image_height=landmarker.height,
                          landmarks=results.face_landmarks[0])
    points_2d_cal, end_points_2d = mp_pnp.project_points_used_in_pnp()

    for p in points_2d_cal:
        cv2.circle(landmarker.input_image, (int(p[0]), int(p[1])), 3, (0, 0, 255), 2)

    for p in end_points_2d:
        cv2.circle(landmarker.input_image, (int(p[0][0]), int(p[0][1])), 3, (255, 0, 0), 2)

    given_points_3d = np.array([
        (10,   0,  0),
        (  0, 10,  0),
        (  0,  0, 10),
        (  0,  0,  0),
    ], dtype = np.float32)
    projected_points_2d = mp_pnp.project_given_points(points_3d = given_points_3d)
    roll, pitch, yaw = mp_pnp.rot_params_rv()

    print(projected_points_2d)
    print(projected_points_2d[0])
    print(projected_points_2d[0][0])

    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0]), int(projected_points_2d[3][0][1])),
             (int(projected_points_2d[0][0][0]), int(projected_points_2d[0][0][1])),
             (255, 0, 0), 2)
    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0]), int(projected_points_2d[3][0][1])),
             (int(projected_points_2d[1][0][0]), int(projected_points_2d[1][0][1])),
             (0, 255, 0), 2)
    cv2.line(landmarker.input_image,
             (int(projected_points_2d[3][0][0]), int(projected_points_2d[3][0][1])),
             (int(projected_points_2d[2][0][0]), int(projected_points_2d[2][0][1])),
             (0, 0, 255), 2)

    print(roll, pitch, yaw)

    cv2.imshow('Final', landmarker.input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))