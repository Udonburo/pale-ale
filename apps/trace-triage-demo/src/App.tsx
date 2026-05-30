import { useMemo, useState } from "react";
import traceFixture from "../../../docs/demo/trace-triage/synthetic-trace.json";

type TraceRow = {
  row_id: string;
  stage: string;
  text: string;
  review_signal: string | null;
  shortlisted: boolean;
};

type ShortlistCard = {
  row_id: string;
  title: string;
  why_shortlisted: string;
  reviewer_question: string;
};

type EvidenceCard = {
  card: string;
  text: string;
  highlighted_text: string[];
};

type BudgetRow = {
  method: string;
  selection_rule: string;
  selected_row_ids: string[];
  rows_inspected: number;
  first_inspect_targets_included: number;
  first_inspect_target_row_ids_included: string[];
};

type TraceFixture = {
  demo: string;
  purpose: string;
  core_message: string;
  plain_language_boundary: string;
  scalar_only_evaluation: {
    final_answer: string;
    llm_as_judge: string;
    similarity_check: string;
    hidden_issue: string;
  };
  trace_triage_output: {
    shortlisted_row_count: number;
    output_type: string;
    inspect_first: Array<{
      row_id: string;
      reason: string;
    }>;
  };
  trace_rows: TraceRow[];
  reviewer_shortlist_cards: ShortlistCard[];
  evidence_comparison: EvidenceCard[];
  same_review_budget_comparison: {
    prompt: string;
    label: string;
    metric: string;
    metric_definition: string;
    rows: BudgetRow[];
  };
  claim_boundaries: string[];
};

const data = traceFixture as TraceFixture;

const budgetMetricSummary =
  "Counts only first-inspect target rows: 03, 04, and 06. Row 05 can help later, but is supporting context here.";

const evidenceMap: Record<string, string[]> = {
  "03": ["company-issued equipment", "manager pre-approval", "personal-device exclusion missing"],
  "04": ["manager pre-approval required", "receipt is enough"],
  "06": ["personal devices excluded", "personal monitor", "after purchase", "receipt is enough"]
};

const artifactRelations = [
  "storyboard.md defines the first-contact screen order and claim boundaries.",
  "synthetic-trace.json provides ordered trace rows, shortlist cards, evidence text, and budget comparison data.",
  "This app renders the fixture as a static review triage walkthrough.",
  "No backend, model call, telemetry upload, or external API is used."
];

const links = [
  ["GitHub repository", "https://github.com/Udonburo/pale-ale"],
  ["Common validation memo", "https://github.com/Udonburo/pale-ale/blob/main/docs/outreach/common-memo-v0.4.2-validation.md"],
  ["Gate12A frozen technical report", "https://doi.org/10.5281/zenodo.19483162"],
  ["Transport-first telemetry note", "https://doi.org/10.5281/zenodo.19569052"],
  ["Gate12B observer-relative closure signatures", "https://doi.org/10.5281/zenodo.20080003"]
] as const;

function App() {
  const [selectedRowId, setSelectedRowId] = useState("06");
  const [activeShortlistId, setActiveShortlistId] = useState("06");

  const selectedRow = useMemo(
    () => data.trace_rows.find((row) => row.row_id === selectedRowId) ?? data.trace_rows[0],
    [selectedRowId]
  );

  const evidenceFocusText = evidenceMap[activeShortlistId] ?? [];

  const selectRow = (rowId: string) => {
    setSelectedRowId(rowId);
    if (evidenceMap[rowId]) {
      setActiveShortlistId(rowId);
    }
  };

  const selectShortlist = (rowId: string) => {
    setSelectedRowId(rowId);
    setActiveShortlistId(rowId);
  };

  const inspectHeroRow = (rowId: string) => {
    selectShortlist(rowId);
    window.requestAnimationFrame(() => {
      document.getElementById(`trace-row-${rowId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  return (
    <main>
      <HeroComparison onInspectRow={inspectHeroRow} />
      <TraceTimeline selectedRow={selectedRow} selectedRowId={selectedRowId} onSelectRow={selectRow} />
      <ReviewerShortlist activeShortlistId={activeShortlistId} onSelectShortlist={selectShortlist} />
      <EvidenceComparison focusText={evidenceFocusText} />
      <ReviewBudgetComparison />
      <ClaimBoundary />
      <TechnicalAppendix />
    </main>
  );
}

function HeroComparison({ onInspectRow }: { onInspectRow: (rowId: string) => void }) {
  return (
    <section className="hero-section" aria-labelledby="hero-title">
      <div className="hero-copy">
        <p className="eyebrow">pale-ale Trace Triage</p>
        <h1 id="hero-title">{data.core_message}</h1>
        <p>{data.plain_language_boundary}</p>
      </div>

      <div className="hero-grid" aria-label="Before and after comparison">
        <article className="comparison-card scalar-card">
          <div>
            <p className="card-kicker">Scalar-only evaluation</p>
            <h2>Passes checks</h2>
          </div>
          <StatusList
            items={[
              ["Final answer", data.scalar_only_evaluation.final_answer],
              ["LLM-as-judge", data.scalar_only_evaluation.llm_as_judge],
              ["Similarity check", data.scalar_only_evaluation.similarity_check]
            ]}
            tone="pass"
          />
          <p className="hidden-note">Hidden: {data.scalar_only_evaluation.hidden_issue}.</p>
        </article>

        <article className="comparison-card triage-card">
          <div>
            <p className="card-kicker">pale-ale Trace Triage</p>
            <h2>Inspect these rows first</h2>
          </div>
          <p className="shortlist-count">{data.trace_triage_output.shortlisted_row_count} review targets</p>
          <p className="output-type">Rows for human review, not a model score.</p>
          <p className="verdict-note">Review targets, not verdicts.</p>
          <ol className="inspect-list">
            {data.trace_triage_output.inspect_first.map((item) => (
              <li key={item.row_id}>
                <button className="inspect-row-button" onClick={() => onInspectRow(item.row_id)} type="button">
                  <span>Row {item.row_id}</span>
                  {item.reason}
                </button>
              </li>
            ))}
          </ol>
        </article>
      </div>
    </section>
  );
}

function StatusList({ items, tone }: { items: Array<[string, string]>; tone: "pass" | "review" }) {
  return (
    <dl className={`status-list ${tone}`}>
      {items.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function TraceTimeline({
  selectedRow,
  selectedRowId,
  onSelectRow
}: {
  selectedRow: TraceRow;
  selectedRowId: string;
  onSelectRow: (rowId: string) => void;
}) {
  return (
    <section className="section timeline-section" aria-labelledby="timeline-title">
      <SectionHeading
        eyebrow="Trace timeline"
        title="Where should a reviewer look first?"
        body="Rows 03, 04, and 06 are first-inspect targets. The shortlist is a place to start review, not a failure verdict."
      />
      <div className="timeline-layout">
        <div className="timeline-list" role="list">
          {data.trace_rows.map((row) => (
            <button
              className={`timeline-row ${row.shortlisted ? "shortlisted" : ""} ${
                selectedRowId === row.row_id ? "selected" : ""
              }`}
              id={`trace-row-${row.row_id}`}
              key={row.row_id}
              onClick={() => onSelectRow(row.row_id)}
              type="button"
            >
              <span className="row-id">{row.row_id}</span>
              <span className="row-main">
                <strong>{row.stage}</strong>
                <span>{row.text}</span>
              </span>
              <span className="row-signal-group">
                {row.shortlisted && <span className="review-target-badge">Review target</span>}
                <span className="row-signal">{row.review_signal ?? "No review signal"}</span>
              </span>
            </button>
          ))}
        </div>

        <aside className="selected-row" aria-live="polite">
          <p className="card-kicker">Selected row {selectedRow.row_id}</p>
          <h3>{selectedRow.stage}</h3>
          <p>{selectedRow.text}</p>
          <dl>
            <div>
              <dt>Why inspect</dt>
              <dd>{selectedRow.review_signal ?? "None shown"}</dd>
            </div>
            <div>
              <dt>Review target</dt>
              <dd>{selectedRow.shortlisted ? "Yes, inspect early" : "No, supporting context"}</dd>
            </div>
          </dl>
        </aside>
      </div>
    </section>
  );
}

function ReviewerShortlist({
  activeShortlistId,
  onSelectShortlist
}: {
  activeShortlistId: string;
  onSelectShortlist: (rowId: string) => void;
}) {
  return (
    <section className="section" aria-labelledby="shortlist-title">
      <SectionHeading
        eyebrow="Reviewer shortlist"
        title="Three first-inspect questions"
        body="Each card says what to check. The signal is a review prompt, not an automatic failure label."
      />
      <div className="shortlist-grid">
        {data.reviewer_shortlist_cards.map((card) => (
          <article
            className={`shortlist-card ${activeShortlistId === card.row_id ? "active" : ""}`}
            key={card.row_id}
          >
            <div className="card-title-row">
              <span className="row-pill">Row {card.row_id}</span>
              <button
                aria-pressed={activeShortlistId === card.row_id}
                className="focus-evidence-button"
                onClick={() => onSelectShortlist(card.row_id)}
                type="button"
              >
                Highlight below
              </button>
            </div>
            <h3>{card.title}</h3>
            <div>
              <p className="field-label">Why shortlisted</p>
              <p>{card.why_shortlisted}</p>
            </div>
            <div>
              <p className="field-label">Question to answer</p>
              <p>{card.reviewer_question}</p>
            </div>
            <div>
              <p className="field-label">Not an automatic failure label</p>
              <p>Signal only. A reviewer decides after checking the source, trace context, and final wording.</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function EvidenceComparison({ focusText }: { focusText: string[] }) {
  return (
    <section className="section" aria-labelledby="evidence-title">
      <SectionHeading
        eyebrow="Evidence comparison"
        title="Compare the source, retrieval, and final answer"
        body="Highlighted text shows the source constraints and final-answer phrases a reviewer should inspect."
      />
      <div className="evidence-grid">
        {data.evidence_comparison.map((card) => (
          <article className="evidence-card" key={card.card}>
            <h3>{card.card}</h3>
            <p>{card.text}</p>
            <div className="evidence-text-list" aria-label={`${card.card} highlighted text`}>
              {card.highlighted_text.map((text) => (
                <span className={focusText.includes(text) ? "focused" : ""} key={text}>
                  {text}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ReviewBudgetComparison() {
  return (
    <section className="section budget-section" aria-labelledby="budget-title">
      <SectionHeading
        eyebrow="Same review budget"
        title={data.same_review_budget_comparison.prompt}
        body={budgetMetricSummary}
      />
      <p className="benchmark-label">{data.same_review_budget_comparison.label}</p>
      <div className="budget-grid">
        {data.same_review_budget_comparison.rows.map((row) => (
          <article className={row.method === "pale-ale shortlist" ? "budget-card best" : "budget-card"} key={row.method}>
            <p className="card-kicker">{row.method}</p>
            <p className="budget-result">
              {row.first_inspect_targets_included}
              <span>
                {row.first_inspect_targets_included === 1 ? " first-inspect target" : " first-inspect targets"}
              </span>
            </p>
            <p>Rows inspected: {row.selected_row_ids.join(", ")}</p>
            <p className="selection-rule">{row.selection_rule}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function ClaimBoundary() {
  return (
    <section className="section boundary-section" aria-labelledby="boundary-title">
      <SectionHeading
        eyebrow="Claim boundary"
        title="This demo shows review triage, not model scoring."
        body="Use it as a triage view, not as an evaluation result."
      />
      <ul className="boundary-list">
        {data.claim_boundaries.slice(1).map((claim) => (
          <li key={claim}>{claim}</li>
        ))}
      </ul>
    </section>
  );
}

function TechnicalAppendix() {
  return (
    <section className="section appendix-section" aria-labelledby="appendix-title">
      <details>
        <summary>
          <span>
            <span className="card-kicker">Technical appendix</span>
            <strong>Fixture, artifact relations, and source links</strong>
          </span>
        </summary>
        <div className="appendix-content">
          <div>
            <h3>Artifact relation list</h3>
            <ul>
              {artifactRelations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>

          <div>
            <h3>RANVIER relationship note</h3>
            <p>
              RANVIER remains a document-grounded constraint audit sidecar. Trace Triage is separate and focuses on
              prioritizing long LLM or agent evaluation trace rows for human review.
            </p>
          </div>

          <div>
            <h3>Links</h3>
            <ul className="link-list">
              {links.map(([label, href]) => (
                <li key={href}>
                  <a href={href} rel="noreferrer" target="_blank">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3>Raw fixture JSON</h3>
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </div>
        </div>
      </details>
    </section>
  );
}

function SectionHeading({ eyebrow, title, body }: { eyebrow: string; title: string; body: string }) {
  return (
    <div className="section-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{body}</p>
    </div>
  );
}

export default App;
