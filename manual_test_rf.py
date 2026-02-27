# manual_test_rf.py
from sqlalchemy import create_engine
from protein_embedding_classifier.core.embeddings import EmbeddingStore
from protein_embedding_classifier.tasks.go_term_prediction.dataset import build_dataset

# 1️⃣ Connectar amb la base de dades
from protein_embedding_classifier.core.db import load_db_config, create_engine_from_config

cfg = load_db_config("config/db.yaml")
engine = create_engine_from_config(cfg)

print("Connected to database.")

# 2️⃣ Crear un store d'embeddings
store = EmbeddingStore(engine)

print("Created EmbeddingStore.")

# 3️⃣ Construir el dataset
X, Y, accessions, go_terms = build_dataset(
    store, 
    embedding_type="ESM", 
    layer=33
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("n GO terms:", len(go_terms))
print("n proteins:", len(accessions))

# 4️⃣ Entrenar un Random Forest com a prova
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print(classification_report(Y_test, Y_pred, target_names=go_terms))