from protein_embedding_classifier.core.db import load_db_config, create_engine_from_config
from protein_embedding_classifier.core.embeddings import EmbeddingStore
from protein_embedding_classifier.tasks.enzyme_vs_not.dataset import build_dataset
from protein_embedding_classifier.classifiers.registry import get_classifier

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
