import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const html = fs.readFileSync(new URL('../../viewer/index.html', import.meta.url), 'utf8');

test('MPR canvases preserve their intrinsic aspect ratios', () => {
  assert.match(html, /height:\s*auto;\s*max-height:\s*140px;\s*object-fit:\s*contain/);
  assert.match(html, /canvas id="cax" class="mpr-axial" width="185" height="185"/);
  assert.match(html, /canvas id="ccor" class="mpr-wide" width="185" height="109"/);
  assert.match(html, /canvas id="csag" class="mpr-wide" width="185" height="109"/);
  assert.doesNotMatch(html, /\.mpr canvas \{[^}]*min-height:\s*140px/);
});
