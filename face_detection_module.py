import cv2
import mediapipe as mp
import time

class face_detector():
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con=min_detection_con
        self.mp_face_detection=mp.solutions.face_detection
        self.mp_draw=mp.solutions.drawing_utils
        self.face_detection=self.mp_face_detection.FaceDetection(min_detection_con)
        

    def find_faces(self, img, draw=True):
        img_RGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.face_detection.process(img_RGB)
        # print(self.results)
        b_boxs=[]
    
        if self.results.detections:
            for  id, detection in enumerate(self.results.detections):
                b_boxc=detection.location_data.relative_bounding_box
                ih, iw, ic= img.shape
                b_box=int(b_boxc.xmin * iw), int(b_boxc.ymin *ih), \
                    int(b_boxc.width * iw), int(b_boxc.height *ih)
                b_boxs.append([id, b_box, detection.score])
                if draw:
                    img=self.fancy_draw(img, b_box)
                    cv2.putText(img, f'{int(detection.score[0]*100)} %',
                                (b_box[0],b_box[1]-20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255,0,255), 2)
        return img, b_boxs        
        
    def fancy_draw(self, img, b_box, l=30, t=5, rt=1):
        x, y, w, h= b_box
        x1, y1= x+w, y+h
        cv2.rectangle(img, b_box, (2255,0,255), rt)
        #This formating is for Top Left(x,y)
        cv2.line(img, (x,y), (x+l,y),(255,0,255), t)
        cv2.line(img, (x,y), (x,y+l),(255,0,255), t)
        #This formating is for Top Right(x1,y1)
        cv2.line(img, (x1,y), (x1-l,y),(255,0,255), t)
        cv2.line(img, (x1,y), (x1,y+l),(255,0,255), t)
         #This formating is for Bottom Left(x,y)
        cv2.line(img, (x,y1), (x+l,y1),(255,0,255), t)
        cv2.line(img, (x,y1), (x,y1-l),(255,0,255), t)
        #This formating is for Bottom Right(x1,y1)
        cv2.line(img, (x1,y1), (x1-l,y1),(255,0,255), t)
        cv2.line(img, (x1,y1), (x1,y1-l),(255,0,255), t)
        return img
    
def main():
    cap = cv2.VideoCapture(r"D:\ComputerVision\Face_Detection_Project\Videos\3.mp4")
    # cap=cv2.VideoCapture(0)
    p_time=0
    detector=face_detector()
    while True:
        success, img = cap.read()
        img,b_boxs=detector.find_faces(img)
        print(b_boxs)
        c_time=time.time()
        fps=1/(c_time-p_time) 
        p_time=c_time
        
        fps = cap.get(cv2.CAP_PROP_FPS) 
        delay = int(1000 / fps) 
        
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
    

if __name__ == "__main__":
    main()