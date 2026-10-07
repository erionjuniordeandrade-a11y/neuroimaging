/** C1 connectotomy panel — lesion cavity ∩ named banks. Fail-closed: no payload → no DOM. */

export const CONNECTOTOMY_FORBIDDEN =
  /\b(margin|navigation|at[\s-]risk|ASSIGNED|Schaefer|Yeo|patient\s+network)\b/i;

export function formatCutLine(row) {
  const nCut = Number(row?.n_cut) || 0;
  const nBank = Number(row?.n_bank) || 0;
  return `cuts ${nCut} of ${nBank} bank streamlines`;
}

export function bankShortName(row) {
  const raw = String(row?.label || row?.id || "");
  return raw.replace(/^bank_/, "").replace(/_/g, "-");
}

export function panelCopyIsClean(text) {
  return !CONNECTOTOMY_FORBIDDEN.test(String(text || ""));
}

/**
 * Mount the C1 panel. Returns the panel element, or null when there is no report
 * (404 / missing lesion). The panel is never inserted on a null payload.
 */
export function mountConnectotomyPanel(host, report, { onRowClick } = {}) {
  if (!host) return null;
  const existing = host.querySelector("#connectotomyPanel");
  if (existing) existing.remove();
  if (!report || !Array.isArray(report.banks)) return null;

  const panel = document.createElement("div");
  panel.id = "connectotomyPanel";
  panel.className = "section connectotomy-panel";
  panel.setAttribute("aria-label", "Connectotomy cavity cuts");

  const title = document.createElement("div");
  title.className = "section-title";
  title.appendChild(document.createTextNode("Cavity ∩ banks "));
  const tag = document.createElement("span");
  tag.className = "tier-tag patient";
  tag.textContent = "Patient";
  title.appendChild(tag);
  panel.appendChild(title);

  // The honesty note carries THIS case's recorded floor and is served with the
  // payload (connectotomy_note in connectotomy.py). No client-side fallback:
  // a derivation-blind literal here would contradict a corrected case (S-14).
  if (report.note) {
    const note = document.createElement("p");
    note.className = "section-hint";
    note.id = "connectotomyNote";
    note.textContent = report.note;
    panel.appendChild(note);
  }

  const list = document.createElement("div");
  list.id = "connectotomyRows";
  list.className = "connectotomy-rows";
  for (const row of report.banks) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "connectotomy-row";
    btn.dataset.bankId = row.id;
    const name = document.createElement("span");
    name.className = "connectotomy-name";
    name.textContent = bankShortName(row);
    const verb = document.createElement("span");
    verb.className = "connectotomy-verb metric";
    verb.textContent = formatCutLine(row);
    btn.appendChild(name);
    btn.appendChild(verb);
    if (typeof onRowClick === "function") {
      btn.addEventListener("click", () => onRowClick(row));
    }
    list.appendChild(btn);
  }
  panel.appendChild(list);

  const text = panel.textContent || "";
  if (!panelCopyIsClean(text)) {
    throw new Error("connectotomy panel copy failed the C1 word gate");
  }
  host.appendChild(panel);
  return panel;
}
