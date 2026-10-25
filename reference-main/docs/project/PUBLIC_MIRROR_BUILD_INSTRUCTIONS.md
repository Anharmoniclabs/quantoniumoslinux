# Public Mirror Build Instructions

## Prerequisites

- `git` installed
- `gh` CLI authenticated (for creating the public repo)
- Access to the private repo at `/workspaces/quantoniumos` (or wherever it lives)

---

## Step 1: Create a clean staging directory

```bash
MIRROR_DIR="$(mktemp -d)/quantoniumos-public"
mkdir -p "$MIRROR_DIR"
echo "Staging directory: $MIRROR_DIR"
```

## Step 2: Copy only allowlisted files

```bash
PRIVATE_REPO="/workspaces/quantoniumos"

# Read allowlist (skip comments and blank lines)
ALLOWLIST=$(grep -v '^#' "$PRIVATE_REPO/docs/project/PUBLIC_MIRROR_ALLOWLIST.txt" \
  | grep -v '^[[:space:]]*$')

# Copy each file, preserving directory structure
while IFS= read -r file; do
  if [[ -f "$PRIVATE_REPO/$file" ]]; then
    mkdir -p "$MIRROR_DIR/$(dirname "$file")"
    cp "$PRIVATE_REPO/$file" "$MIRROR_DIR/$file"
    echo "  copied: $file"
  else
    echo "  WARNING: not found: $file"
  fi
done <<< "$ALLOWLIST"

# Replace the README with the mirror-specific version
cp "$PRIVATE_REPO/docs/project/PUBLIC_MIRROR_README.md" "$MIRROR_DIR/README.md"
echo "  replaced: README.md with PUBLIC_MIRROR_README.md"
```

## Step 3: Verify — no blocked content leaked

```bash
# Check against blocklist patterns
BLOCKLIST=$(grep -v '^#' "$PRIVATE_REPO/docs/project/PUBLIC_MIRROR_BLOCKLIST.txt" \
  | grep -v '^[[:space:]]*$')

LEAKED=0
while IFS= read -r pattern; do
  # Skip directory globs (ending in /) — check if directory exists
  if [[ "$pattern" == */ ]]; then
    if [[ -d "$MIRROR_DIR/$pattern" ]]; then
      echo "  LEAK: directory $pattern exists in mirror!"
      LEAKED=1
    fi
  else
    # Check for exact file or glob match
    matches=$(find "$MIRROR_DIR" -path "*/$pattern" 2>/dev/null)
    if [[ -n "$matches" ]]; then
      echo "  LEAK: $pattern found in mirror!"
      LEAKED=1
    fi
  fi
done <<< "$BLOCKLIST"

if [[ $LEAKED -eq 0 ]]; then
  echo "  OK: No blocked content detected."
fi
```

## Step 4: Verify — print final tree

```bash
echo "=== PUBLIC MIRROR FILE TREE ==="
find "$MIRROR_DIR" -type f | sed "s|$MIRROR_DIR/||" | sort
echo ""
echo "Total files: $(find "$MIRROR_DIR" -type f | wc -l)"
```

## Step 5: Verify — no PII in non-license files

```bash
echo "=== PII CHECK ==="
# Email in LICENSE*, CITATION.cff, and licensing docs is expected (contact for commercial licensing).
# Check for email in OTHER files only.
PII_HITS=$(grep -rn 'luisminier79@gmail.com' "$MIRROR_DIR" 2>/dev/null \
  | grep -v 'CITATION.cff' \
  | grep -v 'LICENSE' \
  | grep -v 'docs/licensing/' \
  | grep -v 'PATENT_NOTICE.md' \
  | grep -v 'PHI_RFT_PROOFS.tex' \
  | grep -v 'DEFINITION_FIREWALL.md' \
  || true)
if [[ -z "$PII_HITS" ]]; then
  echo "  OK: No personal email outside license/citation/legal files"
else
  echo "  WARNING: Personal email found in non-license files:"
  echo "$PII_HITS"
fi
```

## Step 6: Initialize git repo

```bash
cd "$MIRROR_DIR"
git init
git add -A
git commit -m "Initial public mirror — research positioning and proofs only

Source: private repo LMMinier/quantoniumos (main branch)
Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Files: $(find . -type f -not -path './.git/*' | wc -l) (allowlisted subset)"
```

## Step 7: Create public repo and push

```bash
# Create the public repo on GitHub (will prompt for confirmation)
gh repo create LMMinier/quantoniumos-public \
  --public \
  --description "Research positioning and mathematical proofs for the Resonant Fourier Transform (RFT). Mirror of private repo." \
  --source . \
  --push

echo "Done. Public mirror live at: https://github.com/LMMinier/quantoniumos-public"
```

---

## Updating the Mirror

When the private repo changes and the mirror needs updating:

```bash
# 1. Delete old staging dir
rm -rf "$MIRROR_DIR"

# 2. Re-run Steps 1-6 above

# 3. Force-push to the existing public repo
cd "$MIRROR_DIR"
git remote add origin https://github.com/LMMinier/quantoniumos-public.git
git push --force origin main
```

> **Warning:** Force-pushing rewrites public history. This is intentional — the mirror is a snapshot, not a development history.

---

## Checklist Before Pushing

- [ ] Ran Steps 4-5 (tree + PII check) and both passed
- [ ] Reviewed `README.md` in mirror (should be the mirror-specific version, not the private repo README)
- [ ] No `.py`, `.c`, `.sv`, `.asm` files in the mirror
- [ ] No `algorithms/`, `src/`, `tests/`, `benchmarks/` directories in the mirror
- [ ] No `docs/private/` or `docs/patent/` in the mirror
- [ ] `CITATION.cff` email is acceptable (standard for citation metadata)
