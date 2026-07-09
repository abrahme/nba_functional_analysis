#!/usr/bin/env bash
# Syncs generated images and \input-ed .tex tables (from model_output/) and
# stylesheets to a local clone of nba-paper-overleaf, then commits and pushes.
# By default does NOT sync main.tex/appendix.tex — prose normally lives in Overleaf
# and flows back via sync_from_overleaf.sh. Pass --with-tex to ALSO push them.
#
# Usage:
#   ./scripts/sync_overleaf.sh              # images/tables/styles only, unconditionally
#   ./scripts/sync_overleaf.sh --if-changed # only sync if outputs changed in HEAD
#   ./scripts/sync_overleaf.sh --with-tex   # ALSO push main.tex + appendix.tex (repo -> overleaf)
#
# WARNING: --with-tex makes THIS repo authoritative for main.tex/appendix.tex and
# OVERWRITES Overleaf's copies (only the ../model_output/ -> model_output/ path
# rewrite is reversed; the \input{appendix_labels} stub is preserved). Any edits
# made directly in Overleaf since the last pull are lost — keep one source of truth.
#
# Config:
#   Set OVERLEAF_REPO env var to override the default path.

set -euo pipefail

OVERLEAF_REPO="${OVERLEAF_REPO:-$HOME/nba-paper-overleaf}"
MAIN_REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PAPER_DIR="$MAIN_REPO/paper"

# --- parse flags ---
IF_CHANGED=0
WITH_TEX=0
for arg in "$@"; do
  case "$arg" in
    --if-changed) IF_CHANGED=1 ;;
    --with-tex)   WITH_TEX=1 ;;
    *) echo "[overleaf-sync] WARNING: ignoring unknown arg '$arg'" ;;
  esac
done

# --if-changed: only run if the last commit touched generated outputs or stylesheets
# (also paper/main.tex or paper/appendix.tex when --with-tex is set).
if [[ "$IF_CHANGED" == "1" ]]; then
  trigger='^model_output/.*\.(png|pdf|tex)$|^paper/[^/]+\.(cls|sty|bst)$'
  [[ "$WITH_TEX" == "1" ]] && trigger="${trigger}"'|^paper/(main|appendix)\.tex$'
  changed=$(git -C "$MAIN_REPO" diff-tree --no-commit-id -r HEAD --name-only \
    | grep -E "$trigger" || true)
  if [[ -z "$changed" ]]; then
    echo "[overleaf-sync] No relevant changes in this commit, skipping."
    exit 0
  fi
fi

if [[ ! -d "$OVERLEAF_REPO/.git" ]]; then
  echo "[overleaf-sync] ERROR: $OVERLEAF_REPO is not a git repo."
  echo "  Clone nba-paper-overleaf there first:"
  echo "  git clone git@github.com:<you>/nba-paper-overleaf.git $OVERLEAF_REPO"
  exit 1
fi

echo "[overleaf-sync] Syncing to $OVERLEAF_REPO ..."

# --- stylesheets: .cls, .sty, .bst ---
for f in "$PAPER_DIR"/*.{cls,sty,bst}; do
  [[ -f "$f" ]] && cp "$f" "$OVERLEAF_REPO/$(basename "$f")"
done

# --- appendix label stubs (lets overleaf main.tex resolve \ref without full appendix) ---
if [[ -f "$PAPER_DIR/appendix_labels.tex" ]]; then
  cp "$PAPER_DIR/appendix_labels.tex" "$OVERLEAF_REPO/appendix_labels.tex"
  # Inject \input{appendix_labels} into overleaf's main.tex if not already present
  if [[ -f "$OVERLEAF_REPO/main.tex" ]] && ! grep -q '\\input{appendix_labels}' "$OVERLEAF_REPO/main.tex"; then
    sed -i 's|\\end{document}|\\input{appendix_labels}\n\\end{document}|' "$OVERLEAF_REPO/main.tex"
  fi
fi

# --- images and \input-ed .tex files referenced in main.tex and appendix.tex ---
{
  for tex_src in "$PAPER_DIR/main.tex" "$PAPER_DIR/appendix.tex"; do
    [[ -f "$tex_src" ]] || continue
    # strip comment lines before extracting references
    uncommented=$(grep -v '^\s*%' "$tex_src")
    # all \includegraphics paths
    grep -oP '\\includegraphics\[?[^\]]*\]?\{[^}]+\}' <<< "$uncommented" \
      | grep -oP '\{[^}]+\}' | tr -d '{}'
    # only \input paths that point into model_output/ (generated tables/figures)
    grep -oP '\\input\{[^}]+\}' <<< "$uncommented" \
      | grep -oP '\{[^}]+\}' | tr -d '{}' \
      | grep '\.\./model_output/'
  done
} | sort -u | while IFS= read -r file; do
      # file is relative to paper/, e.g. ../model_output/foo.png or images/foo.png
      src="$PAPER_DIR/$file"
      dest_rel="${file#../}"          # strip leading ../ for model_output paths
      dest="$OVERLEAF_REPO/$dest_rel"

      if [[ -f "$src" ]]; then
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
      else
        echo "[overleaf-sync] WARNING: missing $file"
      fi
    done

# --- optional: push main.tex / appendix.tex (inverse of sync_from_overleaf's path rewrite) ---
if [[ "$WITH_TEX" == "1" ]]; then
  for t in main.tex appendix.tex; do
    if [[ -f "$PAPER_DIR/$t" ]]; then
      sed 's|\.\./model_output/|model_output/|g' "$PAPER_DIR/$t" > "$OVERLEAF_REPO/$t"
      echo "[overleaf-sync] pushed $t (../model_output/ -> model_output/)"
    fi
  done
fi

# --- commit and push ---
cd "$OVERLEAF_REPO"
git add -A

if git diff --cached --quiet; then
  echo "[overleaf-sync] Nothing changed in overleaf repo."
  exit 0
fi

commit_msg=$(git -C "$MAIN_REPO" log -1 --pretty=format:"sync: %s")
git commit -m "$commit_msg"
git push origin main

echo "[overleaf-sync] Done."
