#!/usr/bin/env bash
# Shared helpers for the image build scripts. Sourced, not executed.
set -euo pipefail

VENVS=/opt/venvs
# shellcheck disable=SC2034  # used by the install scripts that source this file
MODELS=/opt/models
# shellcheck disable=SC2034
ENVS=/opt/build/envs

# Shallow-fetch one exact commit (GitHub allows fetching any reachable SHA).
clone_pinned() {
    local url=$1 sha=$2 dest=$3
    rm -rf "$dest"
    git init -q "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch -q --depth 1 origin "$sha"
    git -C "$dest" checkout -q FETCH_HEAD
    # Full-depth submodules: pinned submodule commits are not always fetchable shallowly.
    git -C "$dest" submodule update -q --init --recursive
    rm -rf "$dest/.git"
    echo "$url @ $sha" > "$dest/.pinned"
}

# Model venvs see the shared torch stack installed into the system interpreter,
# so each one only adds its own (possibly conflicting) packages on top.
make_venv() {
    local name=$1 python=$2
    "$python" -m venv --system-site-packages "$VENVS/$name"
    # Ubuntu's ensurepip bundles setuptools 59, which ignores pyproject [project]
    # metadata: packages built without isolation would install as "UNKNOWN".
    "$VENVS/$name/bin/python" -m pip install --no-cache-dir --upgrade pip "setuptools>=75" wheel
}

vpip() {
    local name=$1; shift
    "$VENVS/$name/bin/python" -m pip install --no-cache-dir "$@"
}

# Import modules on a GPU-less builder (the driver stub stands in for libcuda.so.1).
# Pass "soft" as 4th argument for imports that may legitimately need a GPU: a failure
# is reported but does not fail the build; the on-pod smoke test covers those.
check_import() {
    local name=$1 workdir=$2 code=$3 mode=${4:-strict}
    local stub status=0
    stub=$(mktemp -d)
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so "$stub/libcuda.so.1"
    (cd "$workdir" && LD_LIBRARY_PATH="$stub:${LD_LIBRARY_PATH:-}" "$VENVS/$name/bin/python" -c "$code") || status=$?
    rm -rf "$stub"
    if [ "$status" -eq 0 ]; then
        echo "import check passed for $name: $code"
    elif [ "$mode" = soft ]; then
        echo "WARNING: import check failed for $name (may need a GPU): $code"
    else
        return "$status"
    fi
}
