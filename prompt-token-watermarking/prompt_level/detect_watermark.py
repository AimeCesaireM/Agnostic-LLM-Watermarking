import sys
import json
import argparse

# Default filename to scan
FILE_NAME = 'watermarked_responses/prompt_watermarked_responses.jsonl'

# Zero-width characters to detect
ZERO_WIDTH_CHARS = {
    0x200B: 'ZERO WIDTH SPACE',
    0x200C: 'ZERO WIDTH NON-JOINER',
    0x200D: 'ZERO WIDTH JOINER',
    0xFEFF: 'ZERO WIDTH NO-BREAK SPACE',
}

def detect_zero_width(text):
    """
    Scan the given text for zero-width characters.
    Returns a list of tuples: (index, codepoint, name).
    """
    findings = []
    for idx, ch in enumerate(text):
        code = ord(ch)
        if code in ZERO_WIDTH_CHARS:
            findings.append((idx, code, ZERO_WIDTH_CHARS[code]))
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Detect zero-width characters in the 'response' field of a .jsonl file."
    )
    parser.add_argument(
        'file',
        nargs='?',  # optional positional
        default=FILE_NAME,
        help=f"Path to the .jsonl file to scan (default: {FILE_NAME})."
    )
    args = parser.parse_args()
    file_to_scan = args.file

    print(f"Scanning file: {file_to_scan}")

    try:
        with open(file_to_scan, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {lineno}: JSON decode error: {e}", file=sys.stderr)
                    continue

                text = record.get('response', '')
                findings = detect_zero_width(text)
                if findings:
                    print(f"{file_to_scan} - Line {lineno}: Detected {len(findings)} zero-width character(s) in response:")
                    for idx, code, name in findings:
                        print(f"  - Position {idx}: U+{code:04X} {name}")
    except FileNotFoundError:
        print(f"Error: file '{file_to_scan}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
