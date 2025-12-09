import json
import logging

# mypy: disable-error-code="unused-ignore"
import os
from typing import Any

import requests
from anthropic import AnthropicVertex
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gcp_config(provider_key: str) -> dict[str, Any] | None:
    """Load GCP configuration for a specific provider."""
    base_key_path = os.path.join(os.path.expanduser("~"), "Documents", "ScienceAI")
    target_key = os.path.join(base_key_path, "scienceai-keys.json")

    print(f"Looking for config at: {target_key}")
    if not os.path.exists(target_key):
        print("Config file not found.")
        return None

    try:
        with open(target_key) as file:
            key_list = json.load(file)
            return key_list.get(provider_key)  # type: ignore
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading config: {e}")
        return None


def main():
    print("--- Testing Anthropic Vertex (Global Only) ---")

    # 1. Load Config
    gcp_config = load_gcp_config("anthropic_vertex")
    if not gcp_config:
        print("Could not load 'anthropic_vertex' config from scienceai-keys.json")
        return

    print(f"Loaded config: {gcp_config}")

    sa_path = gcp_config.get("service_account_path")
    project_id = gcp_config.get("project_id")
    region = "us-east5"

    print(f"Using Region: {region}")

    if not sa_path or not os.path.exists(sa_path):
        print(f"Service account path invalid or missing: {sa_path}")
        return

    # 2. Set Credentials
    print(f"Setting GOOGLE_APPLICATION_CREDENTIALS to {sa_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    # 3. Initialize Client (Inference)
    try:
        print(f"\n[1] Initializing Call Client (region={region})...")
        client = AnthropicVertex(project_id=str(project_id), region=region)

        # Test Model
        model_name = "claude-3-5-sonnet@20240620"
        # model_name = "claude-3-sonnet@20240229"

        print(f"Sending test inference request to model: {model_name}")
        response = client.messages.create(
            model=model_name, max_tokens=10, messages=[{"role": "user", "content": "Hello!"}]
        )
        print("SUCCESS! Response received:")
        print(response.content[0].text)  # type: ignore

    except Exception as e:
        print(f"FAILED to run inference: {e}")

    # 4. Test Token Counting (REST API)
    try:
        print(f"\n[2] Testing Token Counting (REST API, region={region})...")

        # Authenticate for REST
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        access_token = creds.token

        if region == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{region}-aiplatform.googleapis.com"

        base_url = f"https://{host}/v1/projects/{project_id}/locations/{region}/publishers/anthropic/models/count-tokens:rawPredict"

        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json; charset=utf-8"}

        data = {"model": model_name, "messages": [{"role": "user", "content": "Hello!"}]}

        print(f"POST {base_url}")
        resp = requests.post(base_url, headers=headers, json=data, timeout=30)

        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print(f"SUCCESS! Response: {resp.text}")
        else:
            print(f"FAILED! Response: {resp.text}")

    except Exception as e:
        print(f"FAILED to count tokens: {e}")


if __name__ == "__main__":
    main()
