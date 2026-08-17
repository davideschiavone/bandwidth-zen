#!/usr/bin/env bash
# View a .kanata trace in Konata, fetching Konata itself on first use.
#
#   scripts/konata.sh ../matmul-10k.kanata
#   scripts/konata.sh trace-a.kanata trace-b.kanata      # two, side by side
#
# Konata (https://github.com/shioyadan/Konata) is a browser application, not a
# Python package, so it cannot live in the venv. Since v1.0.0 it is a static
# index.html plus a helper that serves one or two traces over loopback; this
# script fetches a *pinned* release, checks its hash, and hands over to that
# helper. Nothing is vendored into the repo: the release is 520 KB of somebody
# else's build output, and a checksum pins it more honestly than a copy would.
#
# The cache lives outside the repo and survives across clones:
#   ${XDG_CACHE_HOME:-~/.cache}/bandwidth-zen/konata-<version>/
#
# Offline? Download konata-v<version>.zip by hand, extract it there, and this
# script will use it without touching the network.
set -euo pipefail

KONATA_VERSION="${KONATA_VERSION:-v1.1.0}"
KONATA_SHA256="554561738a8ace96ebb959d2c4def6cf2b1961cdd329a17e4c429deebe33f446"
KONATA_URL="https://github.com/shioyadan/Konata/releases/download/${KONATA_VERSION}/konata-${KONATA_VERSION}.zip"

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/bandwidth-zen"
install_dir="${cache_root}/konata-${KONATA_VERSION}"
payload="${install_dir}/konata-${KONATA_VERSION}"
helper="${payload}/konata.sh"
index="${payload}/index.html"

# A cached copy counts only if BOTH pieces are there. The helper refuses to run
# without index.html beside it, so checking only for the helper turns an
# interrupted download into a dead end: "index.html was not found next to
# konata.sh", with no hint that deleting the cache would fix it.
have_konata() { [ -f "$helper" ] && [ -f "$index" ]; }

usage() {
    cat >&2 <<'EOF'
Usage: scripts/konata.sh TRACE1 [TRACE2]

Generate a trace first, from backend/:
  uv run bwz matmul -M 10000 -N 10000 -K 10000 -c a100_80gb -d fp16 --ideal \
    --kanata ../matmul-10k.kanata
  uv run bwz run -m llama3_8b -c a100_80gb --kanata llama.kanata   # one per phase

Environment:
  KONATA_VERSION   release tag to use (default v1.1.0; changing it skips the
                   hash check, since the pinned hash belongs to the default)
  KONATA_PORT      loopback port for the viewer (default 30080)
EOF
    exit 2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi
for trace in "$@"; do
    if [ ! -r "$trace" ]; then
        echo "konata.sh: not a readable file: $trace" >&2
        exit 1
    fi
done

if have_konata; then
    : # cached and complete — no network access at all
else
    if [ -e "$install_dir" ]; then
        echo "konata.sh: cached Konata at ${install_dir} is incomplete; refetching" >&2
    fi
    echo "konata.sh: fetching Konata ${KONATA_VERSION} into ${install_dir}" >&2
    for tool in curl unzip python3; do
        command -v "$tool" >/dev/null || { echo "konata.sh: need $tool" >&2; exit 1; }
    done

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    if ! curl -fsSL -o "$tmp/konata.zip" "$KONATA_URL"; then
        echo "konata.sh: download failed. Fetch $KONATA_URL by hand and extract it" >&2
        echo "           into $install_dir, then re-run." >&2
        exit 1
    fi

    # Only the pinned default is hash-checked; an explicit KONATA_VERSION is the
    # caller taking responsibility for what they asked for.
    if [ "$KONATA_VERSION" = "v1.1.0" ]; then
        actual="$(sha256sum "$tmp/konata.zip" | cut -d' ' -f1)"
        if [ "$actual" != "$KONATA_SHA256" ]; then
            echo "konata.sh: checksum mismatch for $KONATA_URL" >&2
            echo "  expected $KONATA_SHA256" >&2
            echo "  got      $actual" >&2
            exit 1
        fi
    fi

    # Extract to a staging directory and swap it in, so an interrupted run
    # leaves either the old cache or none — never a half one that passes the
    # check above.
    unzip -oq "$tmp/konata.zip" -d "$tmp/stage"
    if [ ! -f "$tmp/stage/konata-${KONATA_VERSION}/konata.sh" ] ||
       [ ! -f "$tmp/stage/konata-${KONATA_VERSION}/index.html" ]; then
        echo "konata.sh: the archive did not contain konata-${KONATA_VERSION}/{konata.sh,index.html}" >&2
        exit 1
    fi
    chmod +x "$tmp/stage/konata-${KONATA_VERSION}/konata.sh"
    mkdir -p "$cache_root"
    rm -rf "$install_dir"
    mv "$tmp/stage" "$install_dir"

    # `exec` below replaces this shell, so the EXIT trap would never fire and the
    # downloaded zip would be left behind in /tmp. Clean up while we still can.
    rm -rf "$tmp"
    trap - EXIT
fi

have_konata || { echo "konata.sh: install failed; remove $install_dir and retry" >&2; exit 1; }

# The helper binds a port and lets Python's traceback do the talking if it is
# taken — which it usually is because a previous viewer is still running in
# another terminal. Pick a free one instead, unless the caller pinned it.
start_port="${KONATA_PORT:-30080}"
pinned=0
[ -n "${KONATA_PORT:-}" ] && pinned=1

if ! port="$(python3 - "$start_port" "$pinned" <<'PYEOF'
import socket
import sys

start, pinned = int(sys.argv[1]), sys.argv[2] == "1"
for candidate in range(start, start + (1 if pinned else 20)):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", candidate))
        except OSError:
            continue
    print(candidate)
    break
else:
    sys.exit(1)
PYEOF
)"; then
    if [ "$pinned" = 1 ]; then
        echo "konata.sh: port $start_port is already in use." >&2
        echo "  Another viewer is probably still running — stop it with Ctrl+C in its" >&2
        echo "  terminal, or unset KONATA_PORT to let this script pick a free port." >&2
    else
        echo "konata.sh: no free port in ${start_port}..$((start_port + 19))." >&2
    fi
    exit 1
fi

if [ "$port" != "$start_port" ]; then
    echo "konata.sh: port $start_port is in use (another viewer?); using $port instead" >&2
fi
export KONATA_PORT="$port"

# Konata's own helper serves index.html and the traces over 127.0.0.1 and blocks
# until interrupted. It prints the URL to open.
exec "$helper" "$@"
