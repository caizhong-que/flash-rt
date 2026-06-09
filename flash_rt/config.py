"""Dataset-specific constants and mappings."""

NODE_TYPE_MAP = {
    "cadets": {
        "SUBJECT_PROCESS": 0,
        "FILE_OBJECT_FILE": 1,
        "FILE_OBJECT_UNIX_SOCKET": 2,
        "UnnamedPipeObject": 3,
        "NetFlowObject": 4,
        "FILE_OBJECT_DIR": 5,
    },
    "fivedirections": {
        "SUBJECT_PROCESS": 0,
        "FILE_OBJECT_CHAR": 1,
        "FILE_OBJECT_UNIX_SOCKET": 2,
        "FILE_OBJECT_BLOCK": 3,
        "NetFlowObject": 4,
        "SUBJECT_THREAD": 5,
        "SRCSINK_DATABASE": 6,
        "SRCSINK_PROCESS_MANAGEMENT": 7,
        "VALUE_TYPE_SRC": 8,
    },
    "theia": {
        "SUBJECT_PROCESS": 0,
        "MemoryObject": 1,
        "FILE_OBJECT_BLOCK": 2,
        "NetFlowObject": 3,
        "PRINCIPAL_LOCAL": 4,
    },
    "trace": {
        "SUBJECT_PROCESS": 0,
        "MemoryObject": 1,
        "FILE_OBJECT_CHAR": 2,
        "FILE_OBJECT_FILE": 3,
        "FILE_OBJECT_DIR": 4,
        "SUBJECT_UNIT": 5,
        "UnnamedPipeObject": 6,
        "FILE_OBJECT_UNIX_SOCKET": 7,
        "SRCSINK_UNKNOWN": 8,
        "FILE_OBJECT_LINK": 9,
        "NetFlowObject": 10,
    },
}
