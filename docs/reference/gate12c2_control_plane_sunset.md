# Gate12C-2 legacy control plane

Status: retired on 2026-08-12

The historical Gate12C-2 resource, authorization, recovery, baseline-
commitment, and activation-lineage tools were retired after a bounded
development incident. They are intentionally absent from the curated public
tree and are not an active path for future work.

The retired surface comprised resource qualification, replay and campaign,
activation lineage, original-baseline commitments, draw-profile authorization,
preflight, and closeout-recovery machinery. Its archival source and receipts
are retained outside the active public tree; passing any historical test would
not confer execution or scientific authority.

The retirement preserves these facts:

- the legacy development payloads were not interpreted as a Gate12C-2 result;
- no locked or held-out Gate12C-2 surface was opened;
- old receipt and resource states are historical, not pending successor work;
  and
- no retired authorization or execution state carries into this public
  implementation.

The frozen replacement implementation lives in `tools/gate12c2_minimal/`. It
does not import legacy Gate12C-2 modules and uses no custom authority or
OS-containment protocol. It is retained for audit and synthetic exploration;
it does not authorize another locked or held-out run.
