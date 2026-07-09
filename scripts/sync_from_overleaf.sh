#!/usr/bin/env bash
# Pulls main.tex, appendix.tex, and references.bib from a local clone of
# nba-paper-overleaf back into paper/, reversing the path adjustment applied
# by sync_overleaf.sh.
#
# Usage:
#   ./scripts/sync_from_overleaf.sh              # sync unconditionally
#   ./scripts/sync_from_overleaf.sh --if-changed # only sync if overleaf files changed in HEAD
#
# Config:
#   Set OVERLEAF_REPO env var to override the default path.

set -euo pipefail

OVERLEAF_REPO="${OVERLEAF_REPO:-$HOME/nba-paper-overleaf}"
MAIN_REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PAPER_DIR="$MAIN_REPO/paper"

if [[ ! -d "$OVERLEAF_REPO/.git" ]]; then
  echo "[overleaf-sync] ERROR: $OVERLEAF_REPO is not a git repo."
  echo "  Clone nba-paper-overleaf there first:"
  echo "  git clone git@github.com:<you>/nba-paper-overleaf.git $OVERLEAF_REPO"
  exit 1
fi

# --if-changed: only run if the last commit in the overleaf repo touched tex/bib files
if [[ "${1:-}" == "--if-changed" ]]; then
  changed=$(git -C "$OVERLEAF_REPO" diff-tree --no-commit-id -r HEAD --name-only \
    | grep -E '^(main\.tex|appendix\.tex|references\.bib)$' || true)
  if [[ -z "$changed" ]]; then
    echo "[overleaf-sync] No tex/bib changes in overleaf HEAD, skipping."
    exit 0
  fi
fi

echo "[overleaf-sync] Pulling from $OVERLEAF_REPO ..."
git -C "$OVERLEAF_REPO" pull --ff-only

# --- main.tex: reverse the path adjustment (model_output/ -> ../model_output/) ---
sed 's|model_output/|../model_output/|g' \
  "$OVERLEAF_REPO/main.tex" > "$PAPER_DIR/main.tex"

# --- appendix.tex: same path reversal ---
if [[ -f "$OVERLEAF_REPO/appendix.tex" ]]; then
  sed 's|model_output/|../model_output/|g' \
    "$OVERLEAF_REPO/appendix.tex" > "$PAPER_DIR/appendix.tex"

  # Regenerate label stubs so \ref{} in main.tex resolves without compiling
  # the full appendix (no figures/tables rendered locally).
  {
    printf '%% Auto-generated from appendix.tex -- do not edit directly.\n'
    grep -oP '\\label\{[^}]+\}' "$PAPER_DIR/appendix.tex" | sort -u
  } > "$PAPER_DIR/appendix_labels.tex"

  # Ensure main.tex inputs the stubs (inject before \end{document} if absent)
  if ! grep -q '\\input{appendix_labels}' "$PAPER_DIR/main.tex"; then
    sed -i 's|\\end{document}|\\input{appendix_labels}\n\\end{document}|' "$PAPER_DIR/main.tex"
  fi
fi

# --- references.bib ---
cp "$OVERLEAF_REPO/references.bib" "$PAPER_DIR/references.bib"

# --- commit ---
cd "$MAIN_REPO"
git add paper/main.tex paper/references.bib
[[ -f "$PAPER_DIR/appendix.tex" ]] && git add paper/appendix.tex paper/appendix_labels.tex || true

if git diff --cached --quiet; then
  echo "[overleaf-sync] Nothing changed in paper/."
  exit 0
fi

commit_msg=$(git -C "$OVERLEAF_REPO" log -1 --pretty=format:"sync from overleaf: %s")
git commit -m "$commit_msg"

echo "[overleaf-sync] Done. Review and push when ready."
