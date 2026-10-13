"""Project-local AI Hub credentials, excluded from Git; never log the token."""
import configparser
import getpass
import json
import os
import sys
import argparse
import warnings
from pathlib import Path
import qai_hub as hub
from dotenv import dotenv_values

CONFIG_PATH = Path(__file__).resolve().parents[1] / ".qai_hub" / "client.ini"


def get_client():
    token = os.environ.get("QAI_HUB_API_TOKEN") or dotenv_values(CONFIG_PATH.parent.parent / ".env", encoding="utf-8-sig").get("QAI_HUB_API_TOKEN")
    if token:
        return hub.Client(hub.ClientConfig(api_token=token.strip()))
    if CONFIG_PATH.exists():
        config = configparser.ConfigParser(interpolation=None)
        config.read(CONFIG_PATH, encoding="utf-8")
        return hub.Client(hub.ClientConfig(api_token=config["api"]["api_token"]))
    return hub.Client()  # Existing SDK configuration in the user's home directory.


def save_token(token, path=CONFIG_PATH):
    # Keep official endpoint defaults rather than allowing credentials to be sent
    # to a configurable URL from a checked-in file.
    config = configparser.ConfigParser(interpolation=None)
    config["api"] = {"api_token": token, "api_url": hub.ClientConfig(token).api_url,
                     "web_url": hub.ClientConfig(token).web_url, "verbose": "false"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        config.write(stream)
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing credentials without a prompt")
    args = parser.parse_args()
    try:
        if args.check:
            devices = get_client().get_devices()
        else:
            if not sys.stdin.isatty():
                raise ValueError("Run setup in an interactive terminal for masked key entry")
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("AI Hub API token (hidden): ").strip()
            if not token or token == "YOUR_TOKEN_HERE" or any(c.isspace() for c in token):
                raise ValueError("A real token without whitespace is required")
            devices = hub.Client(hub.ClientConfig(api_token=token)).get_devices()
            save_token(token)
            del token
            print(f"Credential saved locally to {CONFIG_PATH}; excluded from Git.")
        output = Path("results/hub/devices.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([{"name": d.name, "os": d.os, "attributes": d.attributes}
                                      for d in devices], indent=2), encoding="utf-8")
        print(f"Authentication verified. {len(devices)} devices saved to {output}.")
    except Exception as error:
        # SDK response bodies are deliberately not echoed during authentication.
        print(f"AI Hub setup failed ({type(error).__name__}); verify the key and network access.", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
