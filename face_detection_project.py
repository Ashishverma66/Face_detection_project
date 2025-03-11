import cv2
import mediapipe as mp
import time


# cap = cv2.VideoCapture(r"D:\ComputerVision\Face_Detection_Project\Videos\5.mp4")
cap=cv2.VideoCapture(0)
p_time=0
mp_face_detection=mp.solutions.face_detection
mp_draw=mp.solutions.drawing_utils
face_detection=mp_face_detection.FaceDetection(0.75)
while True:
    success, img = cap.read()
    img_RGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=face_detection.process(img_RGB)
    print(results)
    
    if results.detections:
        for  id, detection in enumerate(results.detections):
            # mp_draw.draw_detection(img, detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            b_boxc=detection.location_data.relative_bounding_box
            print(b_boxc)
            ih, iw, ic= img.shape
            b_box=int(b_boxc.xmin * iw), int(b_boxc.ymin *ih), \
                   int(b_boxc.width * iw), int(b_boxc.height *ih)
            
            cv2.rectangle(img, b_box, (2255,0,255), 2)
            
            cv2.putText(img, f'{int(detection.score[0]*100)} %',
                        (b_box[0],b_box[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255,0,255), 2)
            
            
            
    
    fps = cap.get(cv2.CAP_PROP_FPS) 
    delay = int(1000 / fps) 
    
    c_time=time.time()
    fps=1/(c_time-p_time) 
    p_time=c_time
    
    cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 2)
    if not success:
        print("Failed to capture image")
        break  
    cv2.imshow("image", img)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
