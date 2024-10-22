


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

COUNTER = 0
TOTAL = 0
BLINK_LENGTH = 0
blink = False



def blink_function(ear, points):

            # detection of blinks
    global COUNTER, TOTAL, BLINK_LENGTH, blink


    if ear < EYE_AR_THRESH:
        BLINK_LENGTH += 1
        COUNTER += 1
    else:
        if COUNTER > EYE_AR_CONSEC_FRAMES:
            TOTAL += 1
            print(BLINK_LENGTH)
            print('blink_detected')
            blink = True

            COUNTER = 0
            BLINK_LENGTH = 0

        else:
            COUNTER = 0
            BLINK_LENGTH = 0 
            blink = False


    return blink, BLINK_LENGTH


