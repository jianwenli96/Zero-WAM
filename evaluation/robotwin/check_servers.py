"""Check all seven policy HTTP endpoints before starting the simulators."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
import urllib.request


def check(host, port):
    # Policy traffic is direct, including when loopback is an SSH tunnel.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    address = f"[{host}]" if ":" in host else host
    url = f"http://{address}:{port}/healthz"
    try:
        with opener.open(url, timeout=2) as response:
            if response.status != 200 or response.read(32).strip() != b"OK":
                return f"{url}: unexpected health response"
    except Exception as exc:
        return f"{url}: {exc}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--start-port", type=int, default=29556)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.start_port <= 65528 or args.timeout < 0:
        parser.error("invalid port base or negative timeout")
    ports = range(args.start_port + 1, args.start_port + 8)
    deadline = time.monotonic() + args.timeout
    with ThreadPoolExecutor(max_workers=7) as pool:
        while True:
            errors = list(filter(None, pool.map(lambda p: check(args.host, p), ports)))
            if not errors:
                print(f"All seven policy servers at {args.host} are healthy.", flush=True)
                return 0
            if time.monotonic() >= deadline:
                print("Policy servers are not ready:\n" + "\n".join(errors), file=sys.stderr)
                print("Check server logs and HOST. For cross-machine/container use, start "
                      "launch_reverse_tunnel.sh on the server and use HOST=127.0.0.1 here. "
                      "Ping does not test these TCP ports.", file=sys.stderr)
                return 1
            print(f"Waiting for {len(errors)} policy servers; {errors[0]}", flush=True)
            time.sleep(min(2, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    sys.exit(main())
