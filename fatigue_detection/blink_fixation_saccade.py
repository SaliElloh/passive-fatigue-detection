import numpy as np
import cv2
import mediapipe as mp
import math
import pandas as pd

FPS = 30
TIME_PER_FRAME = 1/FPS
default_pose_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

# Important for gaze fixation dtection
SCREEN_RESOLUTION_WIDTH = 1920
SCREEN_RESOLUTION_HEIGHT = 1080
VIEWING_DISTANCE = 15
VISUAL_ANGLE = 2 * math.atan((SCREEN_RESOLUTION_WIDTH/2)/VIEWING_DISTANCE)


# pixel to degree conversion 
PIXEL_PER_DEGREE = SCREEN_RESOLUTION_WIDTH/VISUAL_ANGLE

# for fixation detection
# convert 100 deg/s to pix/s using the conversion PIXEL_PER_DEGREE
PIXEL_PER_SECONDS_FIX = 100*PIXEL_PER_DEGREE

# in 1 frame, how many pixels
PIXELS_FIX = int(TIME_PER_FRAME*PIXEL_PER_SECONDS_FIX)

# for saccadic detection
# CONVERT 300 deg/s to pix/sec using the conversion PIXEL_PER_DEGREE
PIXEL_PER_SECONDS_SAC = 30*PIXEL_PER_DEGREE
PIXELS_SAC = int(TIME_PER_FRAME*PIXEL_PER_SECONDS_SAC)


SEC_TO_MS = 1/1000

# fixation :
FIX_DEG_PER_SEC = 100
FIX_DEG_PER_30_MS = FIX_DEG_PER_SEC * SEC_TO_MS * FPS

# SACCADE
SAC_DEG_PER_SEC = 300
SAC_DEG_PER_30_MS = SAC_DEG_PER_SEC  * SEC_TO_MS * FPS


# For Blink detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

COUNTER = 0
TOTAL = 0
BLINK_LENGTH = 0
blink = False

def event_counter(df):
    """
    Analyzes consecutive occurrences of events within a DataFrame.

    This function calculates two things for each event type in the DataFrame:
    1. The length of consecutive sequences of each event type.
    2. The number of times each event type appears consecutively in these sequences.

    Args:
        df (pd.DataFrame): DataFrame containing at least a column named 'event_type'.

    Returns:
        pd.Series: A Series with the count of consecutive occurrences of each event type.
    """
    result = df.groupby(df['event_type'].ne(df['event_type'].shift()).cumsum())['event_type'].value_counts() 
    

    return result


def calculate_saccade_metrics(df):
  """
  Calculates the distance and velocity for each saccade in a DataFrame.

  Args:
    df: A pandas DataFrame containing saccade data.

  Returns:
    A new DataFrame with distance and velocity columns added.
  """

  df_new = df.copy()
  df["time_dif"] =  df['timeinsec'].diff()
  df['x'] = pd.to_numeric(df['x'], errors='coerce')
  df['y'] = pd.to_numeric(df['y'], errors='coerce')
  df['distance'] = np.sqrt((df['x'].diff().fillna(0) ** 2) + (df['y'].diff().fillna(0) ** 2))
  # Replace zero or NaN time differences with 1/30
  df['time_dif'] = df['time_dif'].replace({0: 1/30, np.nan: 1/30})
  # Calculate velocity
  df['velocity'] = df['distance'] / df['time_dif']

  return df


def blink_fixation_saccade_metrics(df): 
    event_counts = event_counter(df)
    # calculate the average saccades, blink, and fixation duration
    blink_rows = event_counts[event_counts.index.isin(['blink'], level=1)]

    blink_rows = blink_rows.to_frame()
    blink_rows = blink_rows.reset_index(names=['row_number', 'event_type', 'event_number'])

    saccade_rows = event_counts[event_counts.index.isin(['saccade'], level=1)]

    saccade_rows = saccade_rows.to_frame()
    saccade_rows = saccade_rows.reset_index(names=['row_number', 'event_type', 'event_number'])

    fixation_rows = event_counts[event_counts.index.isin(['fixation'], level=1)]

    fixation_rows = fixation_rows.to_frame()
    fixation_rows = fixation_rows.reset_index(names=['row_number', 'event_type', 'event_number'])


    blink_average_counts = blink_rows['count'].mean()
    saccade_average_counts = saccade_rows['count'].mean()

    fixation_average_counts = fixation_rows['count'].mean()

    # finding saccade data
    saccade_data = df[df['event_type'] == 'saccade']

    saccade_data_with_metrics = calculate_saccade_metrics(saccade_data)

    avg_saccadic_distance = saccade_data_with_metrics['distance'].mean()
    avg_saccadic_velocity = saccade_data_with_metrics['velocity'].mean()


    return blink_average_counts, saccade_average_counts, fixation_average_counts, avg_saccadic_distance, avg_saccadic_velocity



def calculate_angle(pointA, pointB):
    A = np.array(pointA)
    B = np.array(pointB)

    dot_product = np.dot(A, B)

    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)

    cos_theta = dot_product / (magnitude_A * magnitude_B)

    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def relative(landmark, shape): return (
    int(landmark.x * shape[1]), int(landmark.y * shape[0]))

def relativeT(landmark, shape): return (
    int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

# def event_detection()
def calculate_gaze_distance(previous_gaze, current_gaze):
    P1 = np.array(previous_gaze)
    P2 = np.array(current_gaze)

    temp = P1 - P2

    euclid_dist = np.sqrt(np.dot(temp.T, temp))

    return euclid_dist
previous_gaze = None

def blink_fixation_saccade(ear, frame, points):

    global COUNTER, TOTAL, BLINK_LENGTH, blink
    
    blink = False
    saccade = False
    fixation = False

    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    # the center of the left eyeball as a vector.
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])
    '''
    camera matrix estimation
    '''
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point
        pupil_world_cord = transformation @ np.array(
            [[left_pupil[0], left_pupil[1], 0, 1]]).T
        
        # 3D gaze point (10 is arbitrary value denoting gaze distance)
        S = Eye_ball_center_left + \
            (pupil_world_cord - Eye_ball_center_left) * 15


        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        # Draw head pose line into screen                                  
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 300.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 1, ((0,255,255)), -1)
        hp1 = ( int(image_points[0][0]), int(image_points[0][1]))
        hp2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, hp1, hp2, (255,0,0), 2)  

          
        gaze = left_pupil + \
            (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

        global previous_gaze

        # Draw gaze line into screen
        ep1 = (int(left_pupil[0]), int(left_pupil[1]))

        ep2 = (int(gaze[0]), int(gaze[1]))

        # Calculate cosine theta for distance change:
    

        # to find whether the eye blinked or not
        if ear < EYE_AR_THRESH:
            BLINK_LENGTH += 1
            COUNTER += 1
            blink = True

        else:
            if COUNTER > EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                print(BLINK_LENGTH)
                print('blink_detected')
                COUNTER = 0
                BLINK_LENGTH = 0

            else:
                COUNTER = 0
                BLINK_LENGTH = 0 
                blink = False


        # calculate gaze distance change
        if previous_gaze is not None:
            dist = calculate_gaze_distance(previous_gaze, ep2)
            time = 1/FPS
            speed = dist/time

            if -50 < dist < 50:
                # print("fixation")
                fixation = True
            elif fixation == False:
                saccade = True

            previous_gaze = ep2

        cv2.line(frame, ep1, ep2, (0,0,255), 2)

        previous_gaze = ep2

        cv2.line(frame, ep1, ep2, (0,0,255), 2)

        pose_data = []
        for ip in image_points:
            pose_data.append(ip[0])
            pose_data.append(ip[1])
        pose_data.append(hp1[0])
        pose_data.append(hp1[1])
        pose_data.append(hp2[0])
        pose_data.append(hp2[1])
        pose_data.append(ep1[0])
        pose_data.append(ep1[1])
        pose_data.append(ep2[0])
        pose_data.append(ep2[1])
        if len(pose_data)>0:
            return (saccade, fixation, blink, ep2, BLINK_LENGTH, frame, pose_data) 
        return (saccade, fixation, blink, (0,0), BLINK_LENGTH, frame, default_pose_data)
    return (saccade, fixation, blink, (0,0), BLINK_LENGTH, frame, default_pose_data)

