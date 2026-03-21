# Git Hooks

This repo uses `.githooks/` as its local `core.hooksPath`.

The current hook set exists for one concrete safety rule:

- never push tracked `*.local.md` files

If a branch tip or outbound commit history contains tracked local-only markdown,
the pre-push hook blocks the push.
