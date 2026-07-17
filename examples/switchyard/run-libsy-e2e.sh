#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end demo of the in-process libsy decision backend.
#
# Unlike run-real-e2e.sh, no Switchyard server runs: routing decisions are made
# in-process by libsy's LLM-classifier algorithm, and every model call libsy
# offloads (the classifier call and the routed call) is fulfilled by Relay's
# own dispatch chain against the fake upstream.

set -euo pipefail

relay_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$relay_root/examples/switchyard/e2e-common.sh"
work_dir="$(mktemp -d)"
upstream_log="$work_dir/upstream.jsonl"

cleanup() {
  local status=$?
  e2e_stop_processes
  if [[ $status -eq 0 ]]; then
    rm -rf "$work_dir"
  else
    echo "E2E logs preserved in $work_dir" >&2
    e2e_tail_logs "$work_dir"
  fi
}
trap cleanup EXIT

python3 "$relay_root/examples/switchyard/libsy-fake-upstream.py" \
  --port 4102 --log "$upstream_log" >"$work_dir/upstream.log" 2>&1 &
e2e_add_pid "$!"

# Relay is built from the repository (which pins its Rust toolchain) but run
# with the temp dir as cwd, where the pin does not apply; carry it over.
if command -v rustup >/dev/null 2>&1; then
  RUSTUP_TOOLCHAIN="$(cd "$relay_root" && rustup show active-toolchain | awk '{print $1}')"
  export RUSTUP_TOOLCHAIN
fi

(
  cd "$work_dir"
  cargo run \
    --manifest-path "$relay_root/Cargo.toml" -p nemo-relay-cli --features switchyard -- \
    --plugin-config-path "$relay_root/examples/switchyard/libsy-plugins.toml" \
    --bind 127.0.0.1:4042
) >"$work_dir/relay.log" 2>&1 &
e2e_add_pid "$!"

e2e_wait_for http://127.0.0.1:4042/healthz

request() {
  local prompt="$1"
  curl --fail --silent http://127.0.0.1:4042/v1/chat/completions \
    -H 'content-type: application/json' \
    -H 'x-nemo-relay-session-id: libsy-e2e' \
    --data-binary "{\"model\":\"client-model\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}"
}

stream_request() {
  local prompt="$1"
  curl --fail --silent --no-buffer http://127.0.0.1:4042/v1/chat/completions \
    -H 'content-type: application/json' \
    -H 'x-nemo-relay-session-id: libsy-e2e' \
    --data-binary "{\"model\":\"client-model\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}"
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local label="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "FAIL: $label: expected $needle in: $haystack" >&2
    exit 1
  fi
  echo "ok: $label"
}

easy="$(request 'What is 2 plus 2?')"
assert_contains "$easy" "answer from weak-model" "easy prompt routes weak"

hard="$(request 'This is a hard problem: prove the Collatz conjecture.')"
assert_contains "$hard" "answer from strong-model" "hard prompt routes strong"

# Streamed requests are routed too: the classifier stream is collected for its
# score in-process and the routed provider stream is bridged back as SSE.
stream_hard="$(stream_request 'This is a hard problem: prove the Riemann hypothesis.')"
assert_contains "$stream_hard" "answer from strong-model" "streamed hard prompt routes strong"

upstream_calls="$(wc -l <"$upstream_log" | tr -d ' ')"
if [[ "$upstream_calls" -ne 6 ]]; then
  echo "FAIL: expected 6 upstream calls (3 classifier + 3 routed), saw $upstream_calls" >&2
  exit 1
fi
echo "ok: classifier and routed calls all flowed through Relay dispatch (6 upstream calls)"

echo "libsy in-process routing e2e passed"
