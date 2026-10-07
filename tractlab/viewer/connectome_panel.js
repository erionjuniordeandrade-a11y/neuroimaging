/**
 * C1b connections panel — 7x7 network-by-network view of the ASSIGNED-edge
 * matrix, drill-down to the parcel pairs inside a cell, and a "Show" action
 * that loads one edge's exact streamlines as tubes (a normal generated layer
 * — chips/legend/unload/review are unchanged; see review_record.js).
 *
 * Pure DOM + pure helpers only: no fetch. All numbers come from the served
 * /api/connectome payload and the served parcel->network LUT; nothing here
 * recomputes a count the server did not already report. Colour encodes
 * evidence (raw streamline counts, blue "Patient" tier) — never the Yeo-7
 * territory wash, which is identity, not evidence (ADR-0001/0003).
 */
import { YEO7_NETWORKS, networkWashRgba, shortParcelName } from "./parcel.js";

export const ADR0003_NOTE =
  "ASSIGNED — parcel tag on a patient streamline, not cortex identity (ADR-0003)";

const HASH64_RE = /^[a-f0-9]{64}$/i;

/**
 * Pre-render identity/completeness guard for one edge tubes response — the
 * single source of truth index.html's showTrackResponse calls before
 * upserting an edge layer, extracted here as a pure function (any
 * Headers-like object with `.get(name)` works, so a fake response in a test
 * exercises the exact same logic a real fetch Response does) rather than
 * duplicated inline.
 *
 * Refuses unless: X-a/X-b equal the requested pair, X-sourcePopulation names
 * that exact pair, X-edgeSourceHash/X-corpusSourceHash/X-parcellationSourceHash
 * are all 64-hex, X-generationId is non-empty, X-assignmentRadiusMm is a
 * finite number > 0 (never defaults to 0 on a missing/blank header),
 * X-matrixPath/X-assignmentsPath are non-empty, and X-matrixSha256/
 * X-assignmentsSha256 are 64-hex. When `expectedGenerationId` is given (the
 * connections panel's own last-fetched /api/connectome generation id) and
 * disagrees with X-generationId, refuses with the specific "reopen the
 * panel" reason — a rebuild republished a new generation after the panel
 * loaded but before Show was clicked.
 *
 * @param {{ get(name: string): (string|null) }} headers
 * @param {{ a: number, b: number, expectedGenerationId?: (string|null) }} request
 * @returns {{ ok: boolean, reason: (string|null), radiusMm: (number|null) }}
 */
export function validateEdgeTubesHeaders(headers, { a, b, expectedGenerationId = null } = {}) {
  const get = (name) => headers.get(name) || "";
  const hA = get("X-a");
  const hB = get("X-b");
  const sourcePopulation = get("X-sourcePopulation");
  const edgeHash = get("X-edgeSourceHash");
  const corpusHash = get("X-corpusSourceHash");
  const parcHash = get("X-parcellationSourceHash");
  const genId = get("X-generationId");
  const radiusRaw = Number(get("X-assignmentRadiusMm"));
  const matrixPath = get("X-matrixPath");
  const matrixSha = get("X-matrixSha256");
  const assignmentsPath = get("X-assignmentsPath");
  const assignmentsSha = get("X-assignmentsSha256");
  const radiusOk = Number.isFinite(radiusRaw) && radiusRaw > 0;

  const identityOk = HASH64_RE.test(edgeHash) && HASH64_RE.test(corpusHash) && HASH64_RE.test(parcHash)
    && genId !== "" && radiusOk
    && matrixPath !== "" && HASH64_RE.test(matrixSha)
    && assignmentsPath !== "" && HASH64_RE.test(assignmentsSha);

  if (String(a) !== hA || String(b) !== hB || sourcePopulation !== `edge:${a}-${b}` || !identityOk) {
    return {
      ok: false,
      radiusMm: null,
      reason: `Returned edge does not match the requested pair or reported incomplete identity `
        + `(requested ${a}-${b}; got a=${hA} b=${hB} sourcePopulation=${sourcePopulation})`,
    };
  }
  if (expectedGenerationId && genId !== expectedGenerationId) {
    return { ok: false, radiusMm: null, reason: "connectome rebuilt, reopen the panel" };
  }
  return { ok: true, reason: null, radiusMm: radiusRaw };
}

/** Network id (1-7) for each 0-based matrix node index, from the served LUT only. */
export function nodeNetworkIds(nodeCount, lut) {
  const out = new Array(nodeCount).fill(0);
  if (!lut) return out;
  for (let i = 0; i < nodeCount; i++) {
    const row = lut[String(i + 1)] || lut[i + 1];
    const id = Number(row?.networkId);
    out[i] = Number.isFinite(id) ? id : 0;
  }
  return out;
}

/**
 * 1-based node ids lacking a valid Yeo-7 membership (1-7) in the served LUT.
 * A null/missing LUT means every node is missing. Empty array = full
 * coverage. The panel refuses to draw at all when this is non-empty — a
 * heatmap that silently drops uncovered nodes is a lie about total evidence.
 */
export function missingNetworkNodes(nodeCount, lut) {
  if (!lut) {
    const all = [];
    for (let i = 1; i <= nodeCount; i++) all.push(i);
    return all;
  }
  const netIds = nodeNetworkIds(nodeCount, lut);
  const missing = [];
  for (let i = 0; i < nodeCount; i++) {
    if (!(netIds[i] >= 1 && netIds[i] <= 7)) missing.push(i + 1);
  }
  return missing;
}

/**
 * Whether a served LUT's own hash agrees with the hash this connectome's own
 * build recorded for the LUT it was built against
 * (provenance.parcellation.lut_sha256). A missing/blank hash on either side
 * never passes — silence is not agreement.
 */
export function lutMatchesProvenance(lutSha256, provenance) {
  const expected = provenance?.parcellation?.lut_sha256;
  return (
    typeof lutSha256 === "string" && lutSha256.length > 0
    && typeof expected === "string" && expected.length > 0
    && lutSha256 === expected
  );
}

/** 7x7 sums of raw counts (index 0..6 = network 1..7), reading only served matrix cells. */
export function aggregateNetworkMatrix(matrix, netIds) {
  const cells = Array.from({ length: 7 }, () => new Array(7).fill(0));
  const n = matrix.length;
  for (let i = 0; i < n; i++) {
    const p = netIds[i];
    if (!(p >= 1 && p <= 7)) continue;
    const row = matrix[i];
    for (let j = 0; j < n; j++) {
      const q = netIds[j];
      if (!(q >= 1 && q <= 7)) continue;
      cells[p - 1][q - 1] += Number(row[j]) || 0;
    }
  }
  return cells;
}

/** Sum of every served matrix cell — the invariant the browser harness checks against. */
export function matrixTotal(matrix) {
  let total = 0;
  for (const row of matrix) for (const v of row) total += Number(v) || 0;
  return total;
}

/**
 * Top-k parcel pairs inside one network cell, by count, unordered pairs only
 * (each edge listed once). Zero-count pairs are dropped: there is nothing to
 * drill into.
 */
export function topPairsInCell(matrix, nodeLabels, netIds, netA, netB, k = 12) {
  const n = matrix.length;
  const pairs = [];
  for (let i = 0; i < n; i++) {
    if (netIds[i] !== netA) continue;
    for (let j = 0; j < n; j++) {
      if (netIds[j] !== netB) continue;
      if (netA === netB && j <= i) continue; // same cell: count each unordered pair once
      const count = Number(matrix[i][j]) || 0;
      if (count <= 0) continue;
      pairs.push({
        a: i + 1, b: j + 1,
        labelA: shortParcelName(nodeLabels[i]), labelB: shortParcelName(nodeLabels[j]),
        count,
      });
    }
  }
  pairs.sort((x, y) => y.count - x.count);
  return pairs.slice(0, k);
}

/** Log-scale intensity in [0,1], 0 for no evidence. */
export function logAlpha(count, maxCount) {
  if (!(count > 0) || !(maxCount > 0)) return 0;
  return Math.min(1, Math.max(0, Math.log1p(count) / Math.log1p(maxCount)));
}

/** Blue "Patient" evidence fill for one heatmap cell — never the Yeo-7 wash. */
export function cellFill(count, maxCount) {
  const a = logAlpha(count, maxCount);
  if (a <= 0) return "rgba(79,140,255,0.04)";
  return `rgba(79,140,255,${(0.08 + 0.82 * a).toFixed(3)})`;
}

function refusalLine(text) {
  const p = document.createElement("p");
  p.className = "section-hint connectome-refusal";
  p.textContent = text;
  return p;
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

const CELL = 28, GAP = 2, ROW_LABEL_W = 118, HEAD_H = 22;

function buildHeatmap(cells, maxCount, onCellClick) {
  const size = 7 * (CELL + GAP) - GAP;
  const viewW = ROW_LABEL_W + size, viewH = HEAD_H + size;
  // display:block (an inline-level replaced SVG's default UA display is
  // "inline") + an explicit CSS width AND aspect-ratio, all inline so there
  // is one source of truth (these same constants) rather than a duplicated
  // magic ratio in a stylesheet. This is the same pattern
  // viewer/profile_panel.js's .profile-chart uses (display:block + an
  // explicit size) rather than relying on an SVG width="100%" PRESENTATION
  // ATTRIBUTE plus "auto" height derived from the viewBox's intrinsic
  // ratio — a replaced-element sizing path that, on a host whose own width
  // is not yet definite at mount time (a collapsed dock row, a details
  // element still settling its open-toggle layout), can resolve to 0x0 and
  // never recover on its own.
  const svg = svgEl("svg", {
    viewBox: `0 0 ${viewW} ${viewH}`,
    role: "img", "aria-label": "7 by 7 network connection heatmap",
    class: "connectome-heatmap",
    style: `display:block;width:100%;aspect-ratio:${viewW}/${viewH};`,
  });
  for (let q = 0; q < 7; q++) {
    const label = svgEl("text", {
      x: ROW_LABEL_W + q * (CELL + GAP) + CELL / 2, y: HEAD_H - 7,
      "text-anchor": "middle", class: "connectome-axis-head",
    });
    label.textContent = String(q + 1);
    svg.appendChild(label);
  }
  for (let p = 0; p < 7; p++) {
    const y = HEAD_H + p * (CELL + GAP);
    const swatch = svgEl("rect", { x: 0, y: y + 2, width: 10, height: 10, rx: 2, fill: networkWashRgba(p + 1, 0.9) });
    const label = svgEl("text", { x: 14, y: y + CELL / 2 + 4, class: "connectome-axis-row" });
    label.textContent = `${p + 1}. ${YEO7_NETWORKS[p].name}`;
    svg.appendChild(swatch);
    svg.appendChild(label);
    for (let q = 0; q < 7; q++) {
      const count = cells[p][q];
      const rect = svgEl("rect", {
        x: ROW_LABEL_W + q * (CELL + GAP), y, width: CELL, height: CELL, rx: 3,
        fill: cellFill(count, maxCount), stroke: "rgba(255,255,255,0.08)", tabindex: "0",
        role: "button",
        "aria-label": `${YEO7_NETWORKS[p].name} to ${YEO7_NETWORKS[q].name}: ${count} assigned streamlines`,
      });
      rect.dataset.netA = String(p + 1);
      rect.dataset.netB = String(q + 1);
      rect.appendChild(svgEl("title")).textContent =
        `${YEO7_NETWORKS[p].name} → ${YEO7_NETWORKS[q].name}: ${count}`;
      const activate = () => onCellClick(p + 1, q + 1, rect);
      rect.addEventListener("click", activate);
      rect.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
      svg.appendChild(rect);
    }
  }
  return svg;
}

function buildProvenanceChips(provenance) {
  const row = document.createElement("div");
  row.className = "connectome-chips";
  const short = (h) => (typeof h === "string" && h.length >= 8) ? h.slice(0, 8) : "?";
  const chips = [
    `corpus ${short(provenance?.corpus?.sha256)}`,
    `parcellation ${short(provenance?.parcellation?.sha256)}`,
    `radius ${provenance?.radius_mm ?? "?"} mm`,
    `${(Number(provenance?.unassigned_fraction) * 100 || 0).toFixed(1)}% unassigned`,
    provenance?.weighting || "raw counts",
    `derivation ${provenance?.active_derivation ?? "none"}`,
  ];
  for (const text of chips) {
    const chip = document.createElement("span");
    chip.className = "tier-tag patient connectome-chip";
    chip.textContent = text;
    row.appendChild(chip);
  }
  const note = document.createElement("p");
  note.className = "section-hint";
  note.textContent = provenance?.label_note || ADR0003_NOTE;
  const wrap = document.createElement("div");
  wrap.append(row, note);
  return wrap;
}

/**
 * Mount the C1b connections panel. Returns the panel element.
 * `payload` = { matrix, nodeLabels, provenance, lut, lutSha256 } — all
 * already fetched; `lutSha256` is the served /api/parcellation/lut route's
 * own `lutSha256` field.
 * `onShowEdge(a, b)` is async and must both load the edge tubes as a
 * generated layer AND return the served counts/lesion info for display here
 * (this module never fetches on its own).
 *
 * Fail-closed on the LUT: a missing/failed fetch (lut null), a node with no
 * network membership, or a served LUT whose own hash disagrees with this
 * connectome's build-time provenance.parcellation.lut_sha256 all render ONE
 * honest refusal line and stop — never a heatmap missing some of its mass.
 */
export function mountConnectomePanel(host, payload, { onShowEdge } = {}) {
  if (!host) return null;
  const existing = host.querySelector("#connectomePanel");
  if (existing) existing.remove();
  const { matrix, nodeLabels, provenance, lut, lutSha256 } = payload || {};
  if (!Array.isArray(matrix) || !Array.isArray(nodeLabels)) return null;

  const panel = document.createElement("div");
  panel.id = "connectomePanel";
  panel.className = "section connectome-panel";
  panel.setAttribute("aria-label", "Connections — assigned edge matrix");

  const title = document.createElement("div");
  title.className = "section-title";
  title.appendChild(document.createTextNode("Connections "));
  const tag = document.createElement("span");
  tag.className = "tier-tag patient";
  tag.textContent = "Patient";
  title.appendChild(tag);
  panel.appendChild(title);
  panel.appendChild(buildProvenanceChips(provenance));

  if (!lutMatchesProvenance(lutSha256, provenance)) {
    panel.appendChild(refusalLine(
      "Parcellation LUT unavailable, unreadable, or does not match this connectome's "
      + "built parcellation — refusing to draw a partial heatmap.",
    ));
    host.appendChild(panel);
    return panel;
  }
  const missing = missingNetworkNodes(matrix.length, lut);
  if (missing.length) {
    panel.appendChild(refusalLine(
      `Parcellation LUT is missing network membership for ${missing.length} of `
      + `${matrix.length} parcel(s) — refusing to draw a partial heatmap.`,
    ));
    host.appendChild(panel);
    return panel;
  }

  const netIds = nodeNetworkIds(matrix.length, lut);
  const cells = aggregateNetworkMatrix(matrix, netIds);
  const maxCount = Math.max(0, ...cells.flat());

  const drilldown = document.createElement("div");
  drilldown.id = "connectomeDrilldown";
  drilldown.className = "connectome-drilldown";
  drilldown.setAttribute("aria-live", "polite");

  const renderCell = (netA, netB) => {
    drilldown.replaceChildren();
    const pairs = topPairsInCell(matrix, nodeLabels, netIds, netA, netB, 12);
    const heading = document.createElement("div");
    heading.className = "section-hint";
    heading.textContent = pairs.length
      ? `${YEO7_NETWORKS[netA - 1].name} → ${YEO7_NETWORKS[netB - 1].name} — top ${pairs.length} parcel pair(s)`
      : `${YEO7_NETWORKS[netA - 1].name} → ${YEO7_NETWORKS[netB - 1].name} — no assigned edges`;
    drilldown.appendChild(heading);
    for (const row of pairs) {
      const line = document.createElement("div");
      line.className = "connectotomy-row connectome-pair-row";
      const name = document.createElement("span");
      name.className = "connectotomy-name";
      name.textContent = `${row.labelA} ↔ ${row.labelB}`;
      const verb = document.createElement("span");
      verb.className = "connectotomy-verb metric";
      verb.textContent = `${row.count} streamline(s)`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ghost connectome-show-btn";
      btn.textContent = "Show";
      const info = document.createElement("div");
      info.className = "section-hint connectome-edge-info";
      btn.addEventListener("click", async () => {
        if (typeof onShowEdge !== "function") return;
        btn.disabled = true;
        info.textContent = "Loading…";
        try {
          const edge = await onShowEdge(row.a, row.b);
          const lesionText = edge?.lesion
            ? `lesion: ${edge.lesion.hits}/${edge.lesion.total} streamlines`
            : `lesion: ${edge?.lesionReason || "not available"}`;
          info.textContent =
            `matrix ${edge?.matrixCount} | assignment rows ${edge?.assignmentRowCount} | `
            + `extracted ${edge?.extractedCount} | ${lesionText}`;
        } catch (e) {
          info.textContent = `Failed: ${e.message}`;
        } finally {
          btn.disabled = false;
        }
      });
      line.append(name, verb, btn);
      drilldown.append(line, info);
    }
  };

  const svg = buildHeatmap(cells, maxCount, (netA, netB, rect) => {
    for (const other of svg.querySelectorAll("rect[data-net-a]")) other.classList.remove("selected");
    rect.classList.add("selected");
    renderCell(netA, netB);
  });
  panel.appendChild(svg);
  panel.appendChild(drilldown);
  host.appendChild(panel);
  return panel;
}

/**
 * One-shot self-heal for a host that was 0-wide (or otherwise not yet laid
 * out — a collapsed dock row, a <details> still settling its open-toggle
 * layout) at the moment its panel was mounted. CSS (width:100% +
 * aspect-ratio on the heatmap svg) does the actual sizing once the host has
 * a real width; this only guards the case where the FIRST paint happened
 * before that width existed at all and nothing else would ever trigger a
 * fresh layout pass.
 *
 * Call this ONCE per panel-open, from the caller that owns the fetch/mount
 * lifecycle (never from inside mountConnectomePanel itself — a callback
 * that re-mounts and re-arms on every invocation would loop forever, since
 * ResizeObserver.observe() fires immediately with the CURRENT size and a
 * freshly-mounted, correctly-sized host would immediately "recover" again).
 * `render` should re-run the same mount with the same payload/opts.
 */
export function watchConnectomeHostResize(host, render) {
  if (typeof ResizeObserver === "undefined" || !host) return;
  if (host._connectomeResizeObserver) host._connectomeResizeObserver.disconnect();
  let fired = false;
  const ro = new ResizeObserver((entries) => {
    if (fired) return;
    const width = entries[0]?.contentRect?.width || 0;
    if (width <= 0) return;
    fired = true;
    ro.disconnect();
    host._connectomeResizeObserver = null;
    void host.offsetWidth; // force a synchronous reflow before re-mounting
    render();
  });
  ro.observe(host);
  host._connectomeResizeObserver = ro;
}
