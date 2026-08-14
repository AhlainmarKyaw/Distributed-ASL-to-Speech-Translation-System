from pathlib import Path
import argparse, csv, cv2, mediapipe as mp, numpy as np

def features_from_hand(hand):
    pts=np.array([[p.x,p.y,p.z] for p in hand.landmark],dtype=np.float32)
    pts-=pts[0]; s=np.max(np.linalg.norm(pts[:,:2],axis=1))
    if s>1e-6: pts/=s
    return pts.reshape(-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="datasets/alphabet_images",help="Folder containing A/, B/, ... Z/ image folders")
    ap.add_argument("--output",default="datasets/alphabet_landmarks.csv")
    args=ap.parse_args()
    root=Path(args.input); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    hands=mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=.45)
    rows=[]; total=used=0
    for classdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label=classdir.name.upper()
        if len(label)!=1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": continue
        for fp in classdir.rglob("*"):
            if fp.suffix.lower() not in exts: continue
            total+=1; img=cv2.imread(str(fp))
            if img is None: continue
            res=hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks: continue
            rows.append([label,*features_from_hand(res.multi_hand_landmarks[0]).tolist()]); used+=1
            if used%500==0: print("extracted",used)
    hands.close()
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["label",*[f"f{i}" for i in range(63)]]); w.writerows(rows)
    print(f"Done. Detected hands in {used}/{total} images. Saved {out}")

if __name__=="__main__": main()
