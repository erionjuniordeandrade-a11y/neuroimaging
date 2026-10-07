/**
 * Per-bundle evidence strip — one row per displayed tract or Atlas prior.
 *
 * A provenance instrument, not an assessment: each row says what kind of
 * evidence it is (Tract vs Atlas prior, ADR-0001), the closest lesion
 * approach exactly as the along-tract profile served it, the fidelity status
 * the bank response carried, and what would change that state. The last
 * column quotes served refusal reasons (fidelity.py FidelityRefusal) or the
 * fixed server statuses; nothing is computed, ranked or combined here.
 *
 * Rules this module keeps (tested in tests/js/evidence_strip.test.js):
 *  - rows keep the caller's display order; they are never reordered by any
 *    value, and no value across rows is summed or averaged;
 *  - a missing value says it is missing and why, never a default;
 *  - no colour, level or wording that reads as a grade of the tract.
 */

import { closestApproachLabel, profileErrorLine } from "./profile_panel.js";

export const STRIP_FRAMING = "Research/preview only — not navigation";

export const CLASS_TRACT = "Tract";
export const CLASS_PRIOR = "Atlas prior";

const TRACT_NOTE = "streamlines tracked in this patient's DWI";
const PRIOR_NOTE = "population atlas, not this patient; served only after human-signed QC";

/** Exact FidelityRefusal texts (src/tractlab/fidelity.py) for a bank whose
 * provenance the server reports as incomplete. The status header does not say
 * which one applied, so both are quoted and the row says so. */
export const PROVENANCE_REFUSALS = Object.freeze([
  "bank has no provenance — run the backfill first",
  "provenance lacks fod_sha256 — cannot pin the FOD",
]);

const NOTHING_OPEN = "no open refusal served for this row";

function served(text) { return { text, state: "served" }; }
function missing(text) { return { text, state: "missing" }; }
function notApplicable(text) { return { text, state: "not-applicable" }; }

/** Closest lesion approach from a cached profile entry. Served values only. */
export function lesionCell(profile, layerSourceHash) {
  if (!profile) return missing("profile not requested yet; focus this tract to load it");
  const scalar = profile.scalar || "fa";
  if (profile.error) return missing(`profile unavailable: ${profileErrorLine(profile.error, scalar)}`);
  const payload = profile.payload;
  if (!payload) return missing(`profile unavailable: ${profileErrorLine("payload_invalid", scalar)}`);
  // A profile read against an older bank file is not this layer's profile.
  if (profile.sourceHash !== undefined && profile.sourceHash !== layerSourceHash) {
    return missing(`profile unavailable: ${profileErrorLine("bank_source_changed", scalar)}`);
  }
  const track = payload.lesion_distance?.track_mm;
  if (track == null) {
    const why = payload.lesion_distance?.reason;
    return missing(`no lesion distance: ${why ? String(why) : "reason not served"}`);
  }
  const label = closestApproachLabel(payload);
  return label ? served(label) : missing("no lesion distance: served track is empty");
}

/** Fidelity status + what would change it, from the decoded bank response. */
export function fidelityCells(fidelity) {
  const status = fidelity?.status;
  if (status === "ok") {
    if (fidelity.opSource === "signed") {
      const who = fidelity.approvedBy ? ` by ${fidelity.approvedBy}` : " (signer not served)";
      const when = fidelity.opDate ? ` ${fidelity.opDate}` : "";
      return { fidelity: served(`measured · operating point signed${who}${when}`), change: [NOTHING_OPEN] };
    }
    return {
      fidelity: served("measured · operating point pilot, unsigned"),
      change: ["an owner-signed operating-point sheet (docs/qc/OPERATING-POINT-fidelity.md)"],
    };
  }
  if (status === "absent") {
    return {
      fidelity: missing("untested · no fidelity sidecar served for this bank"),
      change: ["a fidelity sidecar built for this bank"],
    };
  }
  if (status === "provenance-incomplete") {
    return {
      fidelity: missing("untested · bank provenance incomplete (server did not name the field)"),
      change: [...PROVENANCE_REFUSALS],
    };
  }
  if (status === "error") {
    return {
      fidelity: missing("unreadable · the fidelity block in the bank response did not decode"),
      change: ["a bank response whose fidelity block decodes (reload this tract)"],
    };
  }
  return { fidelity: missing("fidelity status not served"), change: ["a bank response that carries X-fidelityStatus"] };
}

function tractRow(t) {
  const { fidelity, change } = fidelityCells(t.fidelity);
  return Object.freeze({
    key: `bank:${t.bankId}`,
    label: String(t.label || t.bankId || ""),
    displayed: true,
    evidenceClass: CLASS_TRACT,
    classNote: TRACT_NOTE,
    lesion: lesionCell(t.profile, t.layerSourceHash),
    fidelity,
    change: Object.freeze(change),
  });
}

function refusedRow(r) {
  const reason = typeof r.reason === "string" && r.reason.trim() ? r.reason.trim() : "";
  return Object.freeze({
    key: `refused:${r.bankId}`,
    label: String(r.label || r.bankId || ""),
    displayed: false,
    evidenceClass: CLASS_TRACT,
    classNote: `${TRACT_NOTE}; not displayed, the server refused its fidelity join`,
    lesion: notApplicable("not displayed"),
    fidelity: missing("refused by the server (fidelity-refuse)"),
    change: Object.freeze([reason || "refusal reason not served"]),
  });
}

function priorRow(p) {
  return Object.freeze({
    key: `prior:${p.id}`,
    label: String(p.label || p.id || ""),
    displayed: true,
    evidenceClass: CLASS_PRIOR,
    classNote: PRIOR_NOTE,
    lesion: notApplicable("no streamlines; the profile route serves named tracts only"),
    fidelity: notApplicable("support is measured against the patient's own fODF only (ADR-0005)"),
    change: Object.freeze(["nothing: an Atlas prior never becomes a Tract (ADR-0001)"]),
  });
}

/**
 * Build the strip rows. Order: displayed tracts in the caller's order, then
 * refused tracts, then displayed priors, each group in the caller's order.
 *
 * tracts:  [{bankId, label, fidelity, profile, layerSourceHash}]
 *          profile = {payload, sourceHash, scalar} | {error, scalar} | null
 * refused: [{bankId, label, reason}]  (a 422 fidelity-refuse body)
 * priors:  [{id, label}]
 */
export function buildEvidenceRows({ tracts = [], refused = [], priors = [] } = {}) {
  const shown = new Set(tracts.map((t) => t.bankId));
  return Object.freeze([
    ...tracts.map(tractRow),
    ...refused.filter((r) => !shown.has(r.bankId)).map(refusedRow),
    ...priors.map(priorRow),
  ]);
}

/** Every string a row puts on screen, for the vocabulary checks. */
export function rowStrings(row) {
  return [row.label, row.evidenceClass, row.classNote, row.lesion.text, row.fidelity.text, ...row.change];
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** Mount the strip into host. Empty rows → empty host (nothing displayed). */
export function renderEvidenceStrip(host, rows) {
  if (!host) return;
  const wasOpen = host.querySelector("details")?.open;
  host.replaceChildren();
  if (!rows || !rows.length) return;
  const details = el("details", "evidence-strip");
  details.id = "evidenceStrip";
  details.open = wasOpen ?? true;
  const summary = el("summary", "evidence-strip-summary");
  summary.append(el("span", "evidence-strip-title", "Evidence by tract"), el("span", "evidence-strip-frame", STRIP_FRAMING));
  details.append(summary);
  const list = el("div", "evidence-rows");
  list.setAttribute("role", "list");
  for (const row of rows) {
    const item = el("div", "evidence-row");
    item.setAttribute("role", "listitem");
    item.dataset.key = row.key;
    item.dataset.class = row.evidenceClass === CLASS_PRIOR ? "prior" : "tract";
    item.dataset.displayed = String(row.displayed);
    const head = el("div", "evidence-head");
    head.append(el("span", "evidence-name", row.label), el("span", "evidence-class", row.evidenceClass));
    item.append(head, el("div", "evidence-note", row.classNote));
    const cells = [["Lesion", row.lesion], ["Fidelity", row.fidelity]];
    for (const [name, cell] of cells) {
      const line = el("div", "evidence-cell");
      line.dataset.state = cell.state;
      line.append(el("span", "evidence-key", name), el("span", "evidence-val", cell.text));
      item.append(line);
    }
    const change = el("div", "evidence-cell evidence-change");
    change.append(el("span", "evidence-key", "Would change with"));
    const ul = el("ul", "evidence-reasons");
    for (const reason of row.change) ul.append(el("li", null, reason));
    change.append(ul);
    item.append(change);
    list.append(item);
  }
  details.append(list);
  host.append(details);
}
