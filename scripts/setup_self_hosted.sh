#!/usr/bin/env bash
# Turn a plain Ubuntu box (Contabo VPS, workstation, anything) into the machine
# this project's GitHub Actions workflows run on.
#
# Why this rather than just running the scripts by hand: every workflow stays
# exactly as written. The same jobs, the same annotations carrying the output, the
# same run records attached to each commit, the same Python 3.11/3.12 matrix -- but
# executing on hardware with real cores and no per-minute billing. Self-hosted
# runners are free, including on private repositories.
#
# Security, stated plainly because it is the real cost of this approach: a
# self-hosted runner executes whatever a workflow in the repository says to
# execute, on this machine, as this user. That is acceptable here because the
# repository is PRIVATE and only its owner can push to it. It would NOT be
# acceptable on a public repository, where a pull request from any stranger would
# run their code on your box -- GitHub documents this and so does this comment,
# because the two decisions interact: going public to save Actions minutes and
# running a self-hosted runner are mutually exclusive.
#
# The runner is installed under a dedicated unprivileged user for the same reason.
#
#   sudo ./scripts/setup_self_hosted.sh deps
#   sudo ./scripts/setup_self_hosted.sh runner <owner/repo> <registration-token>
#
# The registration token comes from the repository page:
#   Settings -> Actions -> Runners -> New self-hosted runner -> Linux
# It expires in an hour; it is not a personal access token and is safe to paste
# into this command.
#
# Afterwards, switch the workflows over by setting a repository variable:
#   Settings -> Secrets and variables -> Actions -> Variables -> New variable
#   Name: RUNNER_LABEL      Value: self-hosted
# Deleting that variable sends everything back to GitHub's runners. Nothing in the
# workflow files needs editing either way.

set -euo pipefail

RUNNER_USER=${RUNNER_USER:-ghrunner}
RUNNER_HOME=${RUNNER_HOME:-/opt/actions-runner}
RUNNER_VERSION=${RUNNER_VERSION:-2.321.0}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo." >&2
    exit 1
  fi
}

case "${1:-}" in
  deps)
    need_root
    echo "== System packages"
    apt-get update -qq
    # OpenBabel is the fallback ligand preparer; the rest is what building RDKit
    # wheels and running the runner's own scripts need.
    apt-get install -y -qq \
      openbabel python3 python3-venv python3-pip python3-dev \
      build-essential curl jq git ca-certificates

    echo
    echo "== What this machine offers"
    echo "  cores:  $(nproc)"
    echo "  memory: $(free -g | awk '/^Mem:/{print $2}') GB"
    echo "  disk:   $(df -h / | awk 'NR==2{print $4}') free on /"
    echo
    echo "Vina scales with cores, so the core count above is the number that"
    echo "matters. Memory is not the constraint: RDKit and Vina together stay"
    echo "well under 2 GB for a box this size."
    echo
    echo "NOTE for shared-vCPU hosts such as Contabo: the vCPUs are"
    echo "oversubscribed, so per-core speed varies with what the neighbours are"
    echo "doing. Pin Vina's thread count (--cpu) rather than leaving it at 0, and"
    echo "run scripts/probe_cpu_determinism.py to find out whether that thread"
    echo "count changes a score on this machine. A number that moves with the"
    echo "hardware is not comparable with the numbers already in the README."
    ;;

  runner)
    need_root
    repo=${2:-}
    token=${3:-}
    if [ -z "$repo" ] || [ -z "$token" ]; then
      echo "Usage: sudo $0 runner <owner/repo> <registration-token>" >&2
      exit 1
    fi

    id -u "$RUNNER_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$RUNNER_USER"
    mkdir -p "$RUNNER_HOME"
    chown "$RUNNER_USER:$RUNNER_USER" "$RUNNER_HOME"

    if [ ! -f "$RUNNER_HOME/config.sh" ]; then
      echo "== Downloading the runner"
      arch=x64
      [ "$(uname -m)" = "aarch64" ] && arch=arm64
      url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${arch}-${RUNNER_VERSION}.tar.gz"
      sudo -u "$RUNNER_USER" curl -fsSL -o /tmp/runner.tar.gz "$url"
      sudo -u "$RUNNER_USER" tar xzf /tmp/runner.tar.gz -C "$RUNNER_HOME"
      rm -f /tmp/runner.tar.gz
      # Installs the runner's own apt dependencies.
      "$RUNNER_HOME/bin/installdependencies.sh"
    fi

    echo "== Registering with $repo"
    sudo -u "$RUNNER_USER" bash -c "cd '$RUNNER_HOME' && ./config.sh \
      --url 'https://github.com/$repo' \
      --token '$token' \
      --name '$(hostname)' \
      --labels self-hosted,linux,x64,chemdisco \
      --work _work \
      --unattended --replace"

    echo "== Installing as a service so it survives reboots"
    (cd "$RUNNER_HOME" && ./svc.sh install "$RUNNER_USER" && ./svc.sh start)

    echo
    echo "Registered. Now set the repository variable so the workflows use it:"
    echo "  Settings -> Secrets and variables -> Actions -> Variables"
    echo "  RUNNER_LABEL = self-hosted"
    echo
    echo "Then dispatch any workflow as usual. Remove the variable to go back to"
    echo "GitHub's runners; no workflow file changes either way."
    ;;

  status)
    systemctl list-units --type=service | grep -i actions.runner || \
      echo "No runner service found."
    ;;

  *)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
