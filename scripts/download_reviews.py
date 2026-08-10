"""Download OpenReview reviews for a paper forum id into a markdown file.

Credentials are taken from --key-file (a yml file with `email` and `key`
fields), or fall back to the OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD
environment variables.
"""

import argparse

import openreview
import yaml


def _load_credentials(key_file: str) -> tuple[str, str]:
    with open(key_file, "r", encoding="utf-8") as f:
        creds = yaml.safe_load(f)
    if not isinstance(creds, dict) or "email" not in creds or "key" not in creds:
        raise ValueError(f"--key-file {key_file} must contain `email` and `key` fields")
    return str(creds["email"]), str(creds["key"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenReview reviews to markdown")
    parser.add_argument("--paper-id", required=True, help="OpenReview forum/paper id")
    parser.add_argument("--output", required=True, help="Output markdown file path")
    parser.add_argument("--key-file", help="yml file with `email` and `key` fields for login")
    args = parser.parse_args()

    kwargs = {}
    if args.key_file:
        email, key = _load_credentials(args.key_file)
        kwargs = {"username": email, "password": key}

    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", **kwargs)

    notes = client.get_all_notes(forum=args.paper_id)
    reviews = [n for n in notes if any(inv.endswith("/Official_Review") for inv in n.invitations)]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# OpenReview Reviews: {args.paper_id}\n\n")
        f.write(f"Notes in thread: {len(notes)}, reviews: {len(reviews)}\n\n")

        for i, note in enumerate(reviews):
            f.write(f"## Review {i + 1}\n\n")
            f.write(f"Invitation: `{'`, `'.join(note.invitations)}`\n\n")

            for k, v in note.content.items():
                value = v.get("value", v) if isinstance(v, dict) else v
                f.write(f"### {k}\n\n")
                f.write(str(value))
                f.write("\n\n")

            f.write("---\n\n")

    print(f"Saved {len(reviews)} reviews to {args.output}")


if __name__ == "__main__":
    main()
