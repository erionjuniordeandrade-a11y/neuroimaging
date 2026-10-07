import test from "node:test";
import assert from "node:assert/strict";

import { createRenderScheduler } from "../../viewer/render_scheduler.js";

function schedulerHarness(updateResults = [false]) {
  const frames = [];
  const renders = [];
  let updateIndex = 0;
  const scheduler = createRenderScheduler({
    requestFrame(callback) {
      frames.push(callback);
      return frames.length;
    },
    cancelFrame() {},
    update() {
      const value = updateResults[Math.min(updateIndex, updateResults.length - 1)];
      updateIndex += 1;
      return value;
    },
    render(timestamp) {
      renders.push(timestamp);
    },
  });
  return { scheduler, frames, renders };
}

test("an invalidation coalesces into one render and then stays idle", () => {
  const { scheduler, frames, renders } = schedulerHarness([false]);

  assert.equal(scheduler.request(), true);
  assert.equal(scheduler.request(), false);
  assert.equal(frames.length, 1);

  frames.shift()(16);

  assert.deepEqual(renders, [16]);
  assert.equal(frames.length, 0, "a settled scene must not schedule another frame");
  assert.equal(scheduler.pending, false);
});

test("camera damping schedules frames only until controls settle", () => {
  const { scheduler, frames, renders } = schedulerHarness([true, true, false]);

  scheduler.request();
  frames.shift()(10);
  assert.equal(frames.length, 1);
  frames.shift()(20);
  assert.equal(frames.length, 1);
  frames.shift()(30);

  assert.deepEqual(renders, [10, 20, 30]);
  assert.equal(frames.length, 0);
  assert.equal(scheduler.pending, false);
});

test("dispose cancels a pending frame and refuses later work", () => {
  let cancelled = null;
  const scheduler = createRenderScheduler({
    requestFrame() { return 41; },
    cancelFrame(id) { cancelled = id; },
    update() { return false; },
    render() { assert.fail("disposed scheduler rendered"); },
  });

  scheduler.request();
  scheduler.dispose();

  assert.equal(cancelled, 41);
  assert.equal(scheduler.pending, false);
  assert.equal(scheduler.request(), false);
});
