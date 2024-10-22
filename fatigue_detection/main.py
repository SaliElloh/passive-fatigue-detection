import mediapipe as mp
import cv2
from blink_fixation_saccade import blink_fixation_saccade, blink_fixation_saccade_metrics
import numpy as np
mp_face_mesh = mp.solutions.face_mesh # initialize the face mesh model
cap = cv2.VideoCapture
import time
import pprint
import csv
import pandas as pd

# Import necessary module
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from preprocess_data import predict_label, preprocess_weather_label


# for environement prediction
from weather_detector import preprocess_frame, predict_frame

# to alert the driver
from alert_notifier import AlertNotifier


# Constants and configuration
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
FPS = 30
TIME_INCREMENT = 1 / FPS
PROCESS_EVERY_N_FRAMES = 10
WEATHER_PREDICTION_THRESHOLD = 0.5
WEATHER_CLASSES = ['Cloudy', 'Countryside', 'Downtown', 'Evening', 'Highway', 'Morning', 'Night', 'Rainy', 'Sunny']

# Weather frame count
WEATHER_FRAME_COUNT = 0
LAST_PREDICTED_WEATHER_CLASSES = []

# Initialize the alert notifier
notifier = AlertNotifier()

# videos to test:
evening_sunny_highway = 'fatigue_detection/test_data/eveny_sunny_highway.avi'
morning_cloudy_highway = 'fatigue_detection/test_data/morning_cloudy_highway.avi'
night_cloudy_downtown =  'fatigue_detection/test_data/night_cloudy_downtown.avi'

#v

# initialize timestamp
timestamp = 0
fps = 30
time_increment = 1/fps


FPS_START_TIME = time.time()
FRAME_COUNT = 0
PROCESS_EVERY_N_FRAMES = 10
LAST_PREDICTED_WEATHER_CLASSES = []
WEATHER_PREDICTION_THRESHOLD = 0.5


def main():
    global timestamp
    global WEATHER_FRAME_COUNT

    
    all_data = []
    args = get_args()
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

        
    """instantiation of mediapipe face mesh model. This model give back 478 landmarks
    if the rifine_landmarks parameter is set to True. 468 landmarks for the face and
    the last 10 landmarks for the irises
    """

    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # instantiation of the Eye Detector and Head Pose estimator objects
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs

    )

    # timing variables
    prev_time = time.perf_counter()
    fps = 0.0  # Initial FPS value
    t_now = time.perf_counter()

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores

    Scorer = AttScorer(
            t_now=t_now,
            ear_thresh=args.ear_thresh,
            gaze_time_thresh=args.gaze_time_thresh,
            roll_thresh=args.roll_thresh,
            pitch_thresh=args.pitch_thresh,
            yaw_thresh=args.yaw_thresh,
            ear_time_thresh=args.ear_time_thresh,
            gaze_thresh=args.gaze_thresh,
            pose_time_thresh=args.pose_time_thresh,
            verbose=args.verbose,
        )

    cap = cv2.VideoCapture(args.camera)
    cap2 = cv2.VideoCapture(night_cloudy_downtown)


    if not cap.isOpened():  
        print("Cannot open camera")
        exit()

    if not cap2.isOpened():
        print('cannot read video file')

        exit()

    FRAME_NUM = 0
    COUNTER = 0
    TOTAL = 0
    BLINK_LENGTH = 0

    while True:

        t_now = time.perf_counter()

        elapsed_time = t_now - prev_time

        if elapsed_time > 0:
            fps = np.round(1/elapsed_time, 3)

        ret, frame = cap.read()
        
        ret2, frame2 = cap2.read()

        if not ret:  #
            print("Can't receive frame from camera/stream end")
            
        WEATHER_FRAME_COUNT += 1

        if ret2:
            
            cv2.imshow("Driving Video", frame2)
        
            
            weather_predictions = predict_frame(frame2)
            weather_predictions = (weather_predictions > WEATHER_PREDICTION_THRESHOLD).astype(int)
            weather_predictions = weather_predictions.flatten()

            LAST_PREDICTED_WEATHER_CLASSES = [WEATHER_CLASSES[i] for i, val in enumerate(weather_predictions) if val==1]
            
      
        # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        # start the tick counter for computing the processing time for each frame
        e1 = cv2.getTickCount()

        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        # gray = cv2.bilateralFilter(gray, 5, 10, 10)
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)


        results = Detector.process(gray).multi_face_landmarks

        if results:

            landmarks = get_landmarks(results)

            # shows the eye keypoints (can be commented)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )
            # compute the EAR score of the eyes
            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

            saccade, fixation, blink, gaze_coordinates, blink_length, frame, pose_data = blink_fixation_saccade(ear, frame, results[0])

            if (saccade and blink) or (fixation and blink): 
                saccade, fixation, blink = False, False, True

            event_types = (saccade, fixation, blink)
            print(LAST_PREDICTED_WEATHER_CLASSES)
            # Create a list of active event types
            active_event_types = [event_type for event_type in ['saccade', 'fixation', 'blink'] if eval(event_type)]

            # timestamp += time_increment
            # timestamp = np.around(timestamp, decimals=2)
            
            preprocess_weather = preprocess_weather_label(LAST_PREDICTED_WEATHER_CLASSES)

            # for now, lets test with 10
            if active_event_types:
                row = [gaze_coordinates[0], gaze_coordinates[1]] + [elapsed_time] + active_event_types  + preprocess_weather

                if elapsed_time <= 50:
                    all_data.append(row)
                    print(elapsed_time)
                    # process_data

                if elapsed_time > 50:


                    gaze_data = pd.DataFrame(all_data)
                    gaze_data.columns = ['x', 'y', 'timeinsec', 'event_type',                                              
                                            'time_Evening' ,
                                                 'time_Morning' ,
                                                 'time_Night',
                                                 'weather_Cloudy',
                                                 'weather_Rainy',
                                                   'weather_Sunny', 
                                                   'location_Countryside', 
                                                   'location_Downtown', 
                                                   'location_Highway']
                    
                    gaze_data.to_csv('fatigue_detection/test_data/distracted_night_cloudy_downtown_gaze_data.csv', mode='w')


                    blink_average_counts, saccade_average_counts,fixation_average_counts, avg_saccadic_distance, avg_saccadic_velocity = blink_fixation_saccade_metrics(gaze_data)

                    metrics = { 'blink_average_counts': blink_average_counts,
                               'saccade_average_counts': saccade_average_counts,
                               'fixation_average_counts': fixation_average_counts,
                               'avg_saccadic_distance': avg_saccadic_distance, 
                               'avg_saccadic_velocity': avg_saccadic_velocity , 

                    }


                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['metric', 'value'])

                    metrics_df.to_csv('fatigue_detection/test_data/distracted_night_cloudy_downtown_metrics.csv', index=False)
                    
                    predictions_df = predict_label(all_data) 

                    print('saving data from the last 5 minutes')
                    
                    predictions_df.to_csv('fatigue_detection/test_data/distracted_night_cloudy_downtown_prediction_results.csv', index=False)
                    notifier.read_predictions('fatigue_detection/test_data/distracted_night_cloudy_downtown_prediction_results.csv')

                    prev_time = t_now
                    timestamp = 0

                    all_data = []

                    break
                
        
            # compute the Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )

            # compute the head pose
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # evaluate the scores for EAR, GAZE and HEAD POSE
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

        
            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det

            # show the real-time EAR score
            if ear is not None:
                cv2.putText(
                    frame,
                    "EAR:" + str(round(ear, 3)),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # show the real-time Gaze Score
            if gaze is not None:
                cv2.putText(
                    frame,
                    "Gaze Score:" + str(round(gaze, 3)),
                    (10, 80),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if roll is not None:
                cv2.putText(
                    frame,
                    "roll:" + str(roll.round(1)[0]),
                    (450, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if pitch is not None:
                cv2.putText(
                    frame,
                    "pitch:" + str(pitch.round(1)[0]),
                    (450, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if yaw is not None:
                cv2.putText(
                    frame,
                    "yaw:" + str(yaw.round(1)[0]),
                    (450, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if saccade:
                cv2.putText(
                    frame,
                    'saccade',
                    (20, 280),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            
            if blink:
                cv2.putText(
                    frame,
                    "Blink",
                    (10, 300),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if fixation:
                cv2.putText(
                    frame,
                    "fixation",
                    (10, 320),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # if the state of attention of the driver is not normal, show an alert on screen
            if asleep:
                cv2.putText(
                    frame,
                    "ASLEEP!",
                    (10, 300),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if looking_away:
                cv2.putText(
                    frame,
                    "LOOKING AWAY!",
                    (10, 320),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if distracted:
                cv2.putText(
                    frame,
                    "DISTRACTED!",
                    (10, 340),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            FRAME_NUM += 1

                # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()




