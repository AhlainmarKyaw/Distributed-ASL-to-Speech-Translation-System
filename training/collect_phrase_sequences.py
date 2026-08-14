import argparse, time
from pathlib import Path
import cv2, mediapipe as mp, numpy as np

def norm(hand):
    p=np.array([[x.x,x.y,x.z] for x in hand.landmark],dtype=np.float32); p-=p[0]
    s=np.max(np.linalg.norm(p[:,:2],axis=1))
    if s>1e-6:p/=s
    return p.reshape(-1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--label",required=True); ap.add_argument("--samples",type=int,default=40)
    ap.add_argument("--camera",type=int,default=0); a=ap.parse_args()
    out=Path("datasets/phrases")/a.label.upper().replace(" ","_"); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(a.camera); hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=.5,min_tracking_confidence=.5)
    for n in range(a.samples):
        seq=[]; start=time.time()
        while len(seq)<30:
            ok,f=cap.read()
            if not ok: break
            f=cv2.flip(f,1); r=hands.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            cv2.putText(f,f"{a.label} sample {n+1}/{a.samples}  frames {len(seq)}/30",(20,35),0,.8,(255,255,255),2)
            cv2.imshow("Phrase collector - Q to quit",f)
            if cv2.waitKey(1)&0xFF==ord("q"): cap.release(); cv2.destroyAllWindows(); return
            if r.multi_hand_landmarks: seq.append(norm(r.multi_hand_landmarks[0]))
        if len(seq)==30:
            np.save(out/f"{n:04d}.npy",np.asarray(seq,dtype=np.float32)); print("saved",n+1)
        time.sleep(.35)
    cap.release(); hands.close(); cv2.destroyAllWindows()
if __name__=="__main__":main()
