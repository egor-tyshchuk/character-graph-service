import argparse
import secrets
from collections.abc import Sequence

from app.auth.tokens import hash_app_token


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.auth.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    new_token = commands.add_parser("new-app-token", help="generate an app token and its hash")
    new_token.add_argument("--app-id", default="default")
    args = parser.parse_args(argv)

    token = secrets.token_urlsafe(32)
    print(f"token: {token}")
    print(f"APP_TOKENS entry: {args.app_id}:{hash_app_token(token)}")


if __name__ == "__main__":
    main()
