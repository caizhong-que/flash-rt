"""Download datasets from Google Drive."""

import os
import sys
import logging
import argparse
import gdown

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_data")

DATASETS = {
    "cadets": [
        "https://drive.google.com/file/d/1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb/view",
        "https://drive.google.com/file/d/1EycO23tEvZVnN3VxOHZ7gdbSCwqEZTI1/view",
    ],
    "trace": ["https://drive.google.com/file/d/1GG1aUnPjjzzdbxznVTN8X6oVfA-K4oIV/view"],
    "theia": [
        "https://drive.google.com/file/d/10cecNtR3VsHfV0N-gNEeoVeB89kCnse5/view",
        "https://drive.google.com/file/d/1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0/view",
    ],
    "fivedirections": [
        "https://drive.google.com/file/d/1BeP80zUUmm4eZl0UuU43PsKNkl_xgskj/view"
    ],
    "optc": [
        "https://drive.google.com/file/d/1HFSyvmgH0jvdnnnTdKfWRjZYOrLWoIkv/view",
        "https://drive.google.com/file/d/1pJLxJsDV8sngiedbfVajMetczIgM3PQd/view",
        "https://drive.google.com/file/d/1fRQqc68r8-z5BL7H_eAKIDOeHp7okDuM/view",
        "https://drive.google.com/file/d/1VfyGr8wfSe8LBIHBWuYBlU8c2CyEgO5C/view",
        "https://drive.google.com/file/d/10N9ZPolq_L8HivBqzf_jFKbwjSxddsZp/view",
    ],
}


def download_dataset(dataset_name: str, output_dir: str = None, proxy: str = None, verbose: bool = False) -> bool:
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}; available: {list(DATASETS)}")
        return False
    if proxy:
        os.environ["https_proxy"] = proxy
    od = (output_dir or "") + os.sep if output_dir else ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    urls = DATASETS[dataset_name]
    success = 0
    for url in urls:
        out = gdown.download(url, output=od, quiet=not verbose, fuzzy=True, use_cookies=False)
        if out:
            success += 1
            logger.info(f"Downloaded: {out}")
    ok = success == len(urls)
    logger.info(f"Completed: {success}/{len(urls)} {'OK' if ok else 'FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download datasets from Google Drive")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--proxy", default=None, help="HTTPS proxy")
    parser.add_argument("--list", action="store_true", help="List datasets")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.list:
        logger.info(f"Available: {list(DATASETS)}")
        return
    sys.exit(0 if download_dataset(args.dataset, args.output, args.proxy, args.verbose) else 1)


if __name__ == "__main__":
    main()
