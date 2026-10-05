.PHONY: verify
verify:
	@set -e; for d in [0-9][0-9]-*/; do echo "== $$d"; (cd "$$d" && ./verify.sh); done
