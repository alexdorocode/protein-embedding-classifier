from src.core.db import load_db_config, create_engine_from_config
from src.core.embeddings import EmbeddingStore
from tasks.enzyme_vs_not.dataset import build_dataset
from src.classifiers.registry import get_classifier

cfg = load_db_config("config/db.yaml")
engine = create_engine_from_config(cfg)

store = EmbeddingStore(engine, normalize=True)

X, y, acc = build_dataset(
    store,
    embedding_type="Prot-T5",
    layer=5,
)

clf = get_classifier("logistic", C=1.0)
clf.fit(X, y)

y_pred = clf.predict(X)

print("Accuracy (train):", (y_pred == y).mean())
