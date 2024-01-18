import uuid
import json
import numpy as np

from ai_config import config as config

# dummy embedding
init_embedding = {"id": str(uuid.uuid4()), "embedding": list(np.zeros(config.ME_DIMENSIONS))}

# dump embedding to a local file
with open("embeddings_0.json", "w") as f:
    json.dump(init_embedding, f)
