#!/usr/bin/env bash
# Run the full pipeline on one machine, without GitHub Actions.
#
# Everything the Actions workflows do, minus the sharding -- which existed only to
# fit a six-hour job limit and a two-core runner, and is pure overhead on a machine
# with real cores. One process docks the whole list, so there is one ligand
# signature by construction and nothing to pool.
#
# Needs: Ubuntu (or any Linux), Python 3.11+, open network (PyPI, ChEMBL, RCSB).
#
#   chmod +x scripts/run_local.sh
#   ./scripts/run_local.sh setup      # once
#   ./scripts/run_local.sh tests      # the full suite
#   ./scripts/run_local.sh cpu-probe  # does the thread count change a score?
#   ./scripts/run_local.sh discover   # the pipeline, end to end
#
# Results land in runs/ and are worth committing: on a local machine the log is
# the only record, where an Actions run had an immutable one attached to the commit.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-.venv}
CPU=${CPU:-0}
EXHAUSTIVENESS=${EXHAUSTIVENESS:-8}

activate() {
  if [ ! -d "$VENV" ]; then
    echo "No $VENV -- run '$0 setup' first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

case "${1:-}" in
  setup)
    echo "== System packages (OpenBabel is the fallback ligand preparer)"
    sudo apt-get update -qq
    sudo apt-get install -y -qq openbabel

    echo "== Virtualenv"
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -e ".[chem,dev]"
    # vina's own metadata omits these two; installing vina alone gives an
    # ImportError at first use rather than at install time.
    python -m pip install vina scipy gemmi meeko

    echo "== Versions under test"
    python -c "import rdkit; print('RDKit', rdkit.__version__)"
    python -c "from vina import Vina; print('Vina importable')"
    obabel -V || true
    echo
    echo "This machine reports $(nproc) CPU(s)."
    echo "Set CPU=<n> to pin Vina's thread count; see 'cpu-probe' for why that matters."
    ;;

  tests)
    activate
    pytest -v --cov=chemdisco --cov-report=term-missing
    echo
    echo "== Nothing may skip for want of RDKit here"
    pytest -q -rs 2>&1 | tee /tmp/chemdisco_skips.txt | tail -5
    if grep -qi "RDKit is not installed" /tmp/chemdisco_skips.txt; then
      echo "FAIL: a test skipped for want of RDKit on a machine that has it" >&2
      exit 1
    fi
    ;;

  cpu-probe)
    activate
    mkdir -p runs
    python scripts/probe_cpu_determinism.py \
      --exhaustiveness "$EXHAUSTIVENESS" 2>&1 | tee runs/cpu_determinism.log
    ;;

  redock)
    activate
    mkdir -p runs
    python scripts/validate_docking.py \
      --pdb "${2:-4FRS}" --exhaustiveness 16 2>&1 | tee "runs/redock_${2:-4FRS}.log"
    ;;

  enrichment)
    activate
    mkdir -p runs
    # No sharding: one process, one ligand list, nothing to pool.
    python scripts/validate_enrichment.py \
      --exhaustiveness "$EXHAUSTIVENESS" 2>&1 | tee runs/enrichment.log
    ;;

  discover)
    activate
    mkdir -p runs
    python scripts/discover.py \
      --exhaustiveness "$EXHAUSTIVENESS" \
      --cpu "$CPU" \
      --shard 0 --n-shards 1 \
      --time-budget 86400 \
      --output runs/discover_shard_0.json 2>&1 | tee runs/discover.log
    python scripts/discover.py --combine runs \
      --output runs/shortlist.json 2>&1 | tee runs/shortlist.log
    echo
    echo "Shortlist in runs/shortlist.json, log in runs/shortlist.log."
    ;;

  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
