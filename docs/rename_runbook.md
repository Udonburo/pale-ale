# Rename Runbook: `pale-ale-core` -> `pale-ale`

## Scope

This runbook covers manual operations that cannot be completed by the coding agent:

- GitHub repository rename
- Local git remote update
- PyPI release-side setup for the new package name

## Naming Contract (Do/Do Not Change)

- Product name: `pale-ale`
- GitHub repository: `pale-ale` (after manual rename)
- PyPI package name: `pale-ale`
- Python import/module name: `pale_ale_core` (keep)
- Rust lib crate name: `pale-ale-core` (keep)
- Rust CLI binary/crate identity: `pale-ale` (keep)

## 1. Rename the GitHub Repository (Manual UI)

1. Open `https://github.com/Udonburo/pale-ale-core`.
2. Go to `Settings` -> `General`.
3. In `Repository name`, change:
   `pale-ale-core` -> `pale-ale`
4. Confirm rename.

GitHub will keep redirects from old URLs, but local remotes should still be updated explicitly.

## 2. Update Local Git Remote

Run in local clone root (`pale-ale-core` directory before rename, `pale-ale` after rename):

```bash
git remote -v
git remote set-url origin git@github.com:Udonburo/pale-ale.git
git remote -v
```

If you use HTTPS remotes instead of SSH:

```bash
git remote set-url origin https://github.com/Udonburo/pale-ale.git
git remote -v
```

Optional sanity checks:

```bash
git fetch origin --prune
git branch -vv
```

## 3. PyPI Side: Publish Under `pale-ale`

PyPI project names are not renamed in-place in the general case. Treat this as a new project rollout under `pale-ale`.

1. Ensure `pyproject.toml` has:
   `name = "pale-ale"`
2. Create/prepare the `pale-ale` project on PyPI if it does not already exist.
3. If using Trusted Publishing, add or update publisher settings for:
   - Owner: `Udonburo`
   - Repository: `pale-ale`
   - Workflow file: `.github/workflows/release.yml`
   - Environment (if used): `pypi`
4. Publish release artifacts for `pale-ale`.

## 4. Post-Rename Verification

Verify user-facing install/import behavior:

```bash
pip install pale-ale
python -c "import pale_ale_core; print('ok', pale_ale_core.__name__)"
```

Verify Cargo identity remains intentional:

```bash
cargo metadata --no-deps | rg "pale-ale-core|pale-ale-cli|pale_ale_core"
```

## 5. Final Checklist

- [ ] GitHub repository renamed to `pale-ale`
- [ ] `origin` remote updated (`git remote -v` shows `.../pale-ale.git`)
- [ ] PyPI project/publisher for `pale-ale` configured
- [ ] `pip install pale-ale` works
- [ ] `import pale_ale_core` still works
- [ ] Rust crate/module identifiers unchanged where required
