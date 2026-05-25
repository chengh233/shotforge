#!/usr/bin/env python3
"""Install cloudflared (if needed) and start a BACKGROUND tunnel to the local
ComfyUI (127.0.0.1:$PORT), then print the public https URL for the GUI. Detached,
so it keeps running across cells/commands.

    python scripts/tunnel.py
"""
from __future__ import annotations

import os
import re
import subprocess
import time

PORT = int(os.environ.get("PORT", "8188"))
BIN = "/usr/local/bin/cloudflared"
LOG = "/content/cloudflared.log"
_URL = re.compile(r"https://[-\w.]+\.trycloudflare\.com")


def _ensure_cloudflared() -> None:
    if os.path.isfile(BIN):
        return
    print("[tunnel] installing cloudflared")
    subprocess.run(
        f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-amd64 -O {BIN} && chmod +x {BIN}",
        shell=True, check=True,
    )


def main() -> None:
    _ensure_cloudflared()
    subprocess.run("pkill -f 'cloudflared tunnel'", shell=True)  # drop any old tunnel
    time.sleep(1)
    open(LOG, "w").close()
    logf = open(LOG, "ab")
    subprocess.Popen(
        [BIN, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--protocol", "http2"],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    for _ in range(40):
        time.sleep(1)
        try:
            m = _URL.search(open(LOG, encoding="utf-8", errors="replace").read())
        except OSError:
            m = None
        if m:
            print(f"\n[tunnel] ComfyUI GUI ->  {m.group(0)}\n")
            return
    print(f"[tunnel] 还没拿到地址，稍等几秒看日志： tail {LOG}")


if __name__ == "__main__":
    main()
