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
helper="${install_dir}/konata-${KONATA_VERSION}/konata.sh"

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

if [ ! -f "$helper" ]; then
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

    mkdir -p "$install_dir"
    unzip -oq "$tmp/konata.zip" -d "$install_dir"
    chmod +x "$helper"
fi

# Konata's own helper serves index.html and the traces over 127.0.0.1 and blocks
# until interrupted. It prints the URL to open.
exec "$helper" "$@"
