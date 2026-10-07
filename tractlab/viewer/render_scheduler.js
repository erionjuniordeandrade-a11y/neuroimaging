/**
 * Coalescing, on-demand render scheduler.
 *
 * A request produces one update/render frame. If the controls report that
 * damping is still moving the camera, the scheduler asks for exactly one more
 * frame. A settled scene owns no animation-frame callback and performs no GPU
 * draw work.
 */
export function createRenderScheduler({
  requestFrame,
  cancelFrame = () => {},
  update,
  render,
}) {
  if (typeof requestFrame !== "function") throw new TypeError("requestFrame must be a function");
  if (typeof update !== "function") throw new TypeError("update must be a function");
  if (typeof render !== "function") throw new TypeError("render must be a function");

  let disposed = false;
  let pending = false;
  let frameId = null;

  function request() {
    if (disposed || pending) return false;
    pending = true;
    frameId = requestFrame(runFrame);
    return true;
  }

  function runFrame(timestamp) {
    if (disposed) return;
    pending = false;
    frameId = null;
    const cameraStillMoving = Boolean(update(timestamp));
    render(timestamp);
    if (cameraStillMoving) request();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    if (pending) cancelFrame(frameId);
    pending = false;
    frameId = null;
  }

  return {
    request,
    dispose,
    get pending() { return pending; },
  };
}
