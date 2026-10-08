import argparse
import _bootstrap
from src.datasets.download import download_file, safe_extract_zip


def main():
    parser = argparse.ArgumentParser(description='Download an authorized dataset with mandatory SHA-256; no credentials or licenses are bypassed.')
    parser.add_argument('--url', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--accept-license', action='store_true')
    parser.add_argument('--extract')
    args = parser.parse_args()
    result = download_file(args.url, args.output, args.sha256, args.accept_license)
    if args.extract:
        safe_extract_zip(result, args.extract)
    print(result)


if __name__ == '__main__':
    main()
