from protein_embedding_classifier.tasks.dataset_task import DatasetTask

class LocalizationTask(DatasetTask):
    LABELS_SQL_PATH = "tasks/localization/labels.sql"
    LABEL_MAP = {
        "mitochondrion": 0,
        "nucleus": 1,
        "other": 2,
    }
