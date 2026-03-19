import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0)

known_db = {}

for person in os.listdir("known_faces"):
    embs = []

    for f in os.listdir(f"known_faces/{person}"):
        img = cv2.imread(f"known_faces/{person}/{f}")
        faces = app.get(img)

        if len(faces)==0:
            continue

        e = faces[0].embedding
        e = e / np.linalg.norm(e)
        embs.append(e)


    if len(embs)>0:
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        known_db[person] = avg

        print("Added:",person)

np.save("known_db.npy",known_db)
print("Known DB saved")
