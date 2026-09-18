/* Drag maths for the mixer controls, and the crossfader's fader law.
 *
 * Pure functions with no DOM, so the page drags with them and the test suite
 * runs the very same file under Node. Loaded before app.js; exposes
 * window.DragMath in the browser and module.exports under Node.
 *
 * Every value here is computed as GRAB VALUE PLUS DELTA -- what the control
 * read when the pointer went down, moved by how far the pointer has travelled
 * since. Never from the absolute cursor position: that is what makes a handle
 * jump to the cursor the moment it is touched.
 */
(function (root) {
  "use strict";

  /* The thumb's size along the travel axis, from style.css:
   *   ::-webkit-slider-thumb { width: 34px; height: 20px; }
   * A horizontal slider travels along the thumb's width, a vertical one along
   * its height. The thumb cannot leave the box, so the usable travel is the
   * box minus the thumb -- get this wrong and the value runs out early or
   * never reaches the rail. */
  const THUMB_ALONG_AXIS = { horizontal: 34, vertical: 20 };

  const DragMath = {
    THUMB_ALONG_AXIS,

    /** Decimal places implied by a step, for killing float noise. */
    decimals(step) {
      const s = String(step);
      const dot = s.indexOf(".");
      return dot < 0 ? 0 : s.length - dot - 1;
    },

    /** Clamp to [min, max] and snap to the step grid, measured from min. */
    quantize(value, min, max, step) {
      let v = Math.min(max, Math.max(min, value));
      if (step > 0) {
        v = min + Math.round((v - min) / step) * step;
        v = Math.min(max, Math.max(min, v));
        const d = DragMath.decimals(step);
        v = +v.toFixed(Math.min(20, d));
      }
      return v;
    },

    /** Pixels of usable travel in a control's box.
     *
     * `rect` is a getBoundingClientRect(): CSS pixels, borders included, and
     * the same coordinate space as a pointer event's clientX/clientY. It is
     * deliberately NOT offsetWidth, which is rounded to whole pixels and
     * misses any transform.
     */
    travel(rect, vertical, thumb) {
      const along = vertical ? rect.height : rect.width;
      const cap = thumb === undefined
        ? THUMB_ALONG_AXIS[vertical ? "vertical" : "horizontal"]
        : thumb;
      return Math.max(1, along - cap);
    },

    /** The value a drag has reached: where it was grabbed, plus the distance
     * travelled since, scaled by the control's range.
     *
     * A vertical control increases upward, which is what
     * `writing-mode: vertical-lr; direction: rtl` renders, so its delta is
     * inverted against the y axis.
     *
     * devicePixelRatio deliberately does not appear. clientX/clientY and
     * getBoundingClientRect() are both in CSS pixels, so their difference is
     * already in the units the box is measured in; multiplying by dpr would
     * make the control travel twice as fast on a retina display. The dpr
     * factor belongs to canvas backing stores, and lives in fitCanvas().
     */
    valueFromDrag(opts) {
      const vertical = !!opts.vertical;
      const span = opts.max - opts.min;
      const travel = DragMath.travel(opts.rect, vertical, opts.thumb);
      const delta = vertical
        ? opts.originY - opts.clientY
        : opts.clientX - opts.originX;
      return DragMath.quantize(
        opts.grabValue + (delta / travel) * span,
        opts.min, opts.max, opts.step
      );
    },

    /* ---- crossfader ----------------------------------------------------
     * ui_server.set_crossfader turns one position into two gains with an
     * equal-power law, so the middle is not a level dip. Reading the position
     * back out of the gains has to invert that same law: treating the pair as
     * a linear balance, gb / (ga + gb), is wrong everywhere except 0, 0.5 and
     * 1, by up to 4% of travel -- which the crossfader then jumps by the
     * moment the operator lets go of it.
     */

    /** The two deck gains for a crossfader position. Matches ui_server. */
    gainsFromCrossfader(x) {
      const p = Math.min(1, Math.max(0, x));
      return [
        Math.round(Math.cos((p * Math.PI) / 2) * 1e4) / 1e4,
        Math.round(Math.sin((p * Math.PI) / 2) * 1e4) / 1e4,
      ];
    },

    /** The crossfader position two deck gains represent: the exact inverse. */
    crossfaderFromGains(ga, gb) {
      const a = Math.max(0, ga || 0);
      const b = Math.max(0, gb || 0);
      if (a + b < 1e-4) return 0.5;
      return Math.min(1, Math.max(0, Math.atan2(b, a) / (Math.PI / 2)));
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = DragMath;
  else root.DragMath = DragMath;
})(typeof window !== "undefined" ? window : this);
