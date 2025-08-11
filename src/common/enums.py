from enum import Enum


class TaskType(Enum):
    UNDEFINED = "undefined"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    OTHER = "other"
