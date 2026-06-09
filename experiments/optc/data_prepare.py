import os
import gzip
import io
import json
import gdown

def preview_gzip_json(filepath, max_lines=6):
    with gzip.open(filepath, "rb") as f:
        for i, line in enumerate(f):
            print(json.loads(line))
            if i == max_lines - 1:
                break

def extract_logs(filepath, hostid, pathfile=None):
    search_pattern = f"SysClient{hostid}"
    output_filename = f"{pathfile}\\SysClient.systemia.com.txt"

    with gzip.open(filepath, "rt", encoding="utf-8") as fin:
        with open(output_filename, "ab") as f:
            out = io.BufferedWriter(f)
            for line in fin:
                if search_pattern in line:
                    out.write(line.encode("utf-8"))
            out.flush()

def prepare_test_set():
    urls = [
        "https://drive.google.com/file/d/1xIr8gw-4zc8ESjUpYtrFsbOwhPGUSd15/view?usp=drive_link",
        "https://drive.google.com/file/d/1PvlCp2oQaxEBEFGSQWfcFVj19zLOe7yH/view?usp=drive_link"
    ]
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    for url in urls:
        gdown.download(url, quiet=False, use_cookies=False, fuzzy=True)

    log_files = [
        (".\\dataset\\OpTC\\AIA-201-225.ecar-2019-12-08T11-05-10.046.json.gz", "0201"),
        (".\\dataset\\OpTC\\AIA-201-225.ecar-last.json.gz", "0201"),
        (".\\dataset\\OpTC\\AIA-501-525.ecar-2019-11-17T04-01-58.625.json.gz", "0501"),
        (".\\dataset\\OpTC\\AIA-501-525.ecar-last.json.gz", "0501"),
        (".\\dataset\\OpTC\\AIA-51-75.ecar-last.json.gz", "0051")
    ]

    for file, code in log_files:
        extract_logs(file, code)