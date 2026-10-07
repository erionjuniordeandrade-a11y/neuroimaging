/**
 * Deliberately broken pre-fix classifier used only as a regression control.
 * This fixture must not be imported by the shipped viewer.
 */
export function brokenShortBankLabelProseSide(b) {
  const id = String(b?.id || "");
  const lab = String(b?.label || "");
  // Historical bug: /slf.*r/ matches the "r" in "frontal".
  if (/slf3.?r|slf3_r/i.test(id) || /slf.*r/i.test(lab)) return "SLF3-R";
  if (/slf3.?l|slf3_l/i.test(id) || /slf.*l/i.test(lab)) return "SLF3-L";
  if (/fat.?r|fat_r/i.test(id) || /fat-?r/i.test(lab)) return "FAT-R";
  if (/fat.?l|fat_l/i.test(id) || /fat-?l/i.test(lab)) return "FAT-L";
  return lab || id || "?";
}
