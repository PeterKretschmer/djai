/* Waveform view maths: zoom, time <-> pixel, grid lines, band buckets.
 *
 * Pure functions with no DOM, so the page draws with them and the test suite
 * runs the very same file under Node. Loaded before app.js; exposes
 * window.WaveMath in the browser and module.exports under Node. */
(function (root) {
  "use strict";

  const WaveMath = {
    /** Seconds in one bar at a tempo. */
    barSeconds(bpm) {
      return bpm > 0 ? 240 / bpm : 2;
    },

    /** Zoom steps, in visible seconds: the whole track, halving each step,
     * down to exactly one bar. The last step is always one bar. */
    zoomSpans(duration, bpm) {
      const bar = WaveMath.barSeconds(bpm);
      const spans = [Math.max(duration, bar)];
      while (spans[spans.length - 1] / 2 > bar) spans.push(spans[spans.length - 1] / 2);
      if (spans[spans.length - 1] !== bar) spans.push(bar);
      return spans;
    },

    /** The visible window [t0, t1]. The whole track when the span covers it;
     * otherwise centred on the playhead, which is how a zoomed deck scrolls. */
    view(duration, span, playhead) {
      if (span >= duration) return [0, duration];
      const t0 = playhead - span / 2;
      return [t0, t0 + span];
    },

    /** Pixel x for a time, in a window of `width` pixels. */
    x(t, view, width) {
      return ((t - view[0]) / (view[1] - view[0])) * width;
    },

    /** Time for a pixel x: the inverse, for dragging on a zoomed view. */
    t(x, view, width) {
      return view[0] + (x / width) * (view[1] - view[0]);
    },

    /** Indices [i0, i1) of the sorted `times` that fall inside the view. */
    inView(times, view) {
      const lower = (value) => {
        let lo = 0, hi = times.length;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (times[mid] < value) lo = mid + 1; else hi = mid;
        }
        return lo;
      };
      return [lower(view[0]), lower(view[1] + 1e-9)];
    },

    /** The coarsest detail level with at least one point per pixel, else the
     * finest. `levels` is [{points_per_second}], coarse or fine in any order. */
    levelFor(pixelsPerSecond, levels) {
      let best = -1, bestPps = Infinity, finest = 0, finestPps = -1;
      levels.forEach((lv, i) => {
        const pps = lv.points_per_second;
        if (pps > finestPps) { finestPps = pps; finest = i; }
        if (pps >= pixelsPerSecond && pps < bestPps) { bestPps = pps; best = i; }
      });
      return best >= 0 ? best : finest;
    },

    /** Point range [p0, p1) a pixel column covers, at a level's density. */
    bucket(column, view, width, pointsPerSecond) {
      const t0 = WaveMath.t(column, view, width);
      const t1 = WaveMath.t(column + 1, view, width);
      const p0 = Math.floor(t0 * pointsPerSecond);
      return [p0, Math.max(p0 + 1, Math.ceil(t1 * pointsPerSecond))];
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = WaveMath;
  else root.WaveMath = WaveMath;
})(typeof window !== "undefined" ? window : this);
