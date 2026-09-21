#!/usr/bin/env python3
"""Frame README PNGs on mid-gray without modifying their pixels or alpha."""

import base64
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[1] / "docs" / "images"
    previews = [("elf-source", 800, 800), ("elf-fill", 200, 200)]
    for aa in (0, 50, 100):
        previews.extend(
            [(f"elf-aa-{aa}", 200, 200), (f"elf-aa-{aa}-detail", 200, 240)]
        )
    for name, width, height in previews:
        encoded = base64.b64encode((directory / f"{name}.png").read_bytes()).decode("ascii")
        (directory / f"{name}-preview.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
            '  <rect width="100%" height="100%" fill="#808080"/>\n'
            f'  <image width="{width}" height="{height}" '
            f'xlink:href="data:image/png;base64,{encoded}"/>\n'
            '</svg>\n',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
