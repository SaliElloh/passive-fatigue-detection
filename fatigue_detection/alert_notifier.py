import cv2
import pandas as pd
import pyttsx3


class AlertNotifier:

    def __init__(self):
        self.engine = pyttsx3.init()

    def notify_driver(self, message):
        # Send a voice notification
        self.engine.say(message)
        self.engine.runAndWait()


    

    def read_predictions(self, file_path):
        try:
            df = pd.read_csv(file_path)


        except FileNotFoundError:
            print(f'file {file_path} not found')

            return
        
        if 'predictions' not in df.columns:
            print("The 'predictions' column is missing in the file")
            return
        
        
        risk_count = (df['predictions'] == 'inattentive').sum()
        predictions_len = (len(df['predictions']))

        fatigue_percentage = (risk_count/predictions_len) * 100
        if risk_count >= 10:  
            self.notify_driver("You seem drowsy. Please stay alert!")

        else:
            print('Driver is alert, no notification necessary')