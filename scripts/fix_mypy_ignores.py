from __future__ import annotations

import pathlib
import re
import subprocess

ERR_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): error: .+ \[(?P<code>[-a-z0-9]+)\]$"
)
UNUSED_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): error: unused 'type: ignore' comment"
)


def run_mypy() -> list[str]:
    proc = subprocess.run(
        [
            "mypy",
            "--ignore-missing-imports",
            "--warn-unused-ignores",
            "--show-error-codes",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.splitlines()


def main() -> None:
    lines = run_mypy()
    by_line_codes: dict[tuple[str, int], set[str]] = {}
    unused_ignores: set[tuple[str, int]] = set()

    for line in lines:
        err_match = ERR_RE.match(line)
        unused_match = UNUSED_RE.match(line)
        if err_match:
            key = (err_match.group("path"), int(err_match.group("line")))
            by_line_codes.setdefault(key, set()).add(err_match.group("code"))
        elif unused_match:
            key = (unused_match.group("path"), int(unused_match.group("line")))
            unused_ignores.add(key)

    changed = 0

    for (path, line_no), codes in sorted(by_line_codes.items()):
        file_path = pathlib.Path(path)
        if not file_path.exists():
            continue
        source_lines = file_path.read_text(encoding="utf-8").splitlines()
        idx = line_no - 1
        if not 0 <= idx < len(source_lines):
            continue
        if "# type: ignore" in source_lines[idx] and "[" not in source_lines[idx]:
            source_lines[idx] = source_lines[idx].replace(
                "# type: ignore",
                f"# type: ignore[{','.join(sorted(codes))}]",
            )
            file_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
            changed += 1

    for path, line_no in sorted(unused_ignores):
        file_path = pathlib.Path(path)
        if not file_path.exists():
            continue
        source_lines = file_path.read_text(encoding="utf-8").splitlines()
        idx = line_no - 1
        if not 0 <= idx < len(source_lines):
            continue
        if "# type: ignore" in source_lines[idx]:
            source_lines[idx] = source_lines[idx].replace("# type: ignore", "").rstrip()
            file_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
            changed += 1

    print(f"Updated lines: {changed}")


if __name__ == "__main__":
    main()
