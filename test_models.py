import numpy as np

db = np.load("known_db.npy", allow_pickle=True).item()

print("Total identities:", len(db))

for k,v in db.items():
    print(k, type(v), v.shape, np.linalg.norm(v))
