#!/usr/bin/env python3
"""
Generate a new GRYPHGEN API key.

Usage:
    python3 scripts/keygen.py [--label LABEL]

Prints the plaintext key (copy to the client, never stored) and the
SHA-256 hash to append to GRYPHGEN_API_KEYS in .env.

After adding the hash, restart the service:
    sudo systemctl restart gryphgen-agentic
"""

import argparse
import hashlib
import os
import secrets
import sys


def generate_key() -> tuple[str, str]:
    key = "grug_" + secrets.token_urlsafe(32)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return key, digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GRYPHGEN API key")
    parser.add_argument("--label", default="", help="Optional label for your records")
    args = parser.parse_args()

    key, digest = generate_key()

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    existing = ""
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("GRYPHGEN_API_KEYS="):
                    existing = line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass

    if existing:
        new_value = f"{existing},{digest}"
    else:
        new_value = digest

    label_note = f"  ({args.label})" if args.label else ""

    print()
    print("=" * 60)
    print(f"  New API key{label_note}")
    print("=" * 60)
    print(f"\n  KEY (give to client, shown once):\n\n    {key}\n")
    print(f"  HASH (add to .env):\n\n    {digest}\n")
    print("  Updated GRYPHGEN_API_KEYS line for .env:")
    print(f"\n    GRYPHGEN_API_KEYS={new_value}\n")
    print("  Then restart:  sudo systemctl restart gryphgen-agentic")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
