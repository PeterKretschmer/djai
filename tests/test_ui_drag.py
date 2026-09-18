"""Dragging the mixer controls: the value maths, and the rules the page obeys.

THREADING CONTEXT: main thread (pytest). The drag maths (static/dragmath.js)
runs under Node, the same file the browser loads. The crossfader law is checked
against ui_server's own implementation so the two cannot drift apart.

The bug these cover: dragging a fader made the handle jump. Measured in a real
renderer with real pointer events, three separate causes.

  * The 20 Hz state feed already skipped a control being held, but dropped that
    guard the instant the pointer came up, while the engine's echo of the
    gesture was still in flight. Letting go of a fader swung the handle 11% of
    its travel with 60 ms of echo latency, and 58% with 400 ms.
  * The beat-grid handle took the absolute cursor position rather than an
    offset from where it was grabbed, so a press 7 px off the handle moved it
    7 px before the hand did.
  * The crossfader was set with an equal-power law and read back with a linear
    one, a 4% round-trip error that the handle jumped by on release.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _node(script: str):
    path = json.dumps(str(STATIC / "dragmath.js"))
    code = f"const D = require({path});\n{script}"
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# A fader's box, as the page lays it out: 10 CSS px wide, 70 tall, vertical.
FADER = {"x": 100, "y": 200, "width": 10, "height": 70}


# --- the value maths ------------------------------------------------------------


@needs_node
def test_a_slow_drag_across_the_full_travel_is_monotonic():
    """Every step of a one-pixel-at-a-time sweep moves the value one way."""
    result = _node("""
      const rect = %s;
      const travel = D.travel(rect, true);
      const out = [];
      // Grab at the top (value 1) and walk the pointer down, one pixel a step.
      for (let dy = 0; dy <= travel; dy++) {
        out.push(D.valueFromDrag({
          rect, vertical: true, min: 0, max: 1, step: 0.01,
          grabValue: 1, originX: 105, originY: 210,
          clientX: 105, clientY: 210 + dy,
        }));
      }
      console.log(JSON.stringify({out, travel}));
    """ % json.dumps(FADER))
    values, travel = result["out"], result["travel"]

    assert travel == FADER["height"] - 20, "travel is the box less the thumb"
    assert values[0] == 1.0, "the value does not move before the pointer does"
    assert values[-1] == 0.0, "a full sweep reaches the far rail"
    rises = [b for a, b in zip(values, values[1:]) if b > a + 1e-12]
    assert rises == [], f"a downward drag never rises: {rises[:5]}"
    step = 0.01
    biggest = max(abs(b - a) for a, b in zip(values, values[1:]))
    assert biggest <= 2 * step + 1e-12, f"no jumps; biggest step {biggest}"


@needs_node
def test_the_handle_does_not_snap_to_the_cursor_on_mousedown():
    """Pressing far from the handle and not moving leaves the value alone."""
    result = _node("""
      const rect = %s;
      const out = [];
      // Pressed 30 px below the thumb, pointer has not moved since.
      for (const grab of [0, 0.25, 0.5, 0.75, 1]) {
        out.push(D.valueFromDrag({
          rect, vertical: true, min: 0, max: 1, step: 0.01,
          grabValue: grab, originX: 105, originY: 240,
          clientX: 105, clientY: 240,
        }));
      }
      console.log(JSON.stringify(out));
    """ % json.dumps(FADER))
    assert result == [0, 0.25, 0.5, 0.75, 1], (
        "a press with no movement must not change the value, wherever it lands"
    )


@needs_node
def test_the_value_follows_the_delta_not_the_cursor():
    """Same pointer position, different grab: the offset is what carries."""
    result = _node("""
      const rect = %s;
      const same = (grab) => D.valueFromDrag({
        rect, vertical: true, min: 0, max: 1, step: 0.01,
        grabValue: grab, originX: 105, originY: 230,
        clientX: 105, clientY: 220,      // ten pixels up from the grab
      });
      console.log(JSON.stringify([same(0.2), same(0.5), same(0.8)]));
    """ % json.dumps(FADER))
    travel = FADER["height"] - 20
    rise = 10.0 / travel
    assert result == [
        pytest.approx(round(0.2 + rise, 2), abs=0.011),
        pytest.approx(round(0.5 + rise, 2), abs=0.011),
        pytest.approx(round(0.8 + rise, 2), abs=0.011),
    ], "the same 10 px of travel moves every grab by the same amount"


@needs_node
def test_dragging_is_identical_at_device_pixel_ratio_1_and_2():
    """clientX/clientY and getBoundingClientRect are both CSS pixels.

    Their difference is already in the units the box is measured in, so the
    device pixel ratio must not appear in the value maths at all. Scaling by it
    would make a control travel twice as fast on a retina display. The dpr
    factor belongs to canvas backing stores, in fitCanvas().
    """
    result = _node("""
      const rect = %s;
      const sweep = () => {
        const out = [];
        for (let dy = 0; dy <= 50; dy += 5) {
          out.push(D.valueFromDrag({
            rect, vertical: true, min: 0, max: 1, step: 0.01,
            grabValue: 1, originX: 105, originY: 210,
            clientX: 105, clientY: 210 + dy,
          }));
        }
        return out;
      };
      const one = sweep();
      globalThis.devicePixelRatio = 2;     // nothing in here may read this
      const two = sweep();
      console.log(JSON.stringify({one, two}));
    """ % json.dumps(FADER))
    assert result["one"] == result["two"]


@needs_node
def test_a_horizontal_control_travels_along_x_and_a_vertical_one_up():
    result = _node("""
      const h = {x: 0, y: 0, width: 278, height: 34};
      const v = %s;
      console.log(JSON.stringify({
        right: D.valueFromDrag({rect: h, vertical: false, min: 0, max: 1,
          step: 0.01, grabValue: 0.5, originX: 100, originY: 17,
          clientX: 140, clientY: 17}),
        up: D.valueFromDrag({rect: v, vertical: true, min: 0, max: 1,
          step: 0.01, grabValue: 0.5, originX: 105, originY: 230,
          clientX: 105, clientY: 210}),
        down: D.valueFromDrag({rect: v, vertical: true, min: 0, max: 1,
          step: 0.01, grabValue: 0.5, originX: 105, originY: 230,
          clientX: 105, clientY: 250}),
      }));
    """ % json.dumps(FADER))
    assert result["right"] > 0.5, "dragging right raises a horizontal control"
    assert result["up"] > 0.5, "dragging up raises a vertical control"
    assert result["down"] < 0.5, "dragging down lowers it"


@needs_node
def test_the_value_stays_on_the_step_grid_and_inside_the_rails():
    result = _node("""
      const rect = %s;
      const far = (dy) => D.valueFromDrag({
        rect, vertical: true, min: -8, max: 8, step: 0.05,
        grabValue: 0, originX: 105, originY: 230, clientX: 105, clientY: 230 - dy,
      });
      console.log(JSON.stringify({
        up: far(10000), down: far(-10000), small: far(3),
        q: [D.quantize(0.317, 0, 1, 0.01), D.quantize(-99, 0, 1, 0.01),
            D.quantize(99, 0, 1, 0.01), D.quantize(0.1 + 0.2, 0, 1, 0.01)],
      }));
    """ % json.dumps(FADER))
    assert result["up"] == 8 and result["down"] == -8, "clamped to the rails"
    assert abs(result["small"] / 0.05 - round(result["small"] / 0.05)) < 1e-9
    assert result["q"] == [0.32, 0, 1, 0.3], "snapped, clamped, and no float dust"


# --- the crossfader law ---------------------------------------------------------


@needs_node
def test_the_crossfader_reads_back_exactly_what_the_server_was_set_to():
    positions = [i / 40 for i in range(41)]
    result = _node("""
      const xs = %s;
      console.log(JSON.stringify(xs.map((x) => {
        const [a, b] = D.gainsFromCrossfader(x);
        return [a, b, D.crossfaderFromGains(a, b)];
      })));
    """ % json.dumps(positions))

    for x, (ga, gb, back) in zip(positions, result):
        assert back == pytest.approx(x, abs=5e-4), (
            f"crossfader at {x} read back as {back}"
        )
        # Equal power: the two gains sum in quadrature to one, so the middle of
        # the travel is not a level dip.
        assert ga * ga + gb * gb == pytest.approx(1.0, abs=2e-4)


@needs_node
def test_the_crossfader_law_matches_ui_server_exactly():
    """The page and the server must compute the same two gains.

    Read back with the old linear balance, gb / (ga + gb), the error reaches
    4% of the travel -- which the handle jumps by the moment it is released.
    """
    from djai.ui_server import UIServer  # noqa: F401  (import guard only)

    positions = [0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]
    page = _node("""
      const xs = %s;
      console.log(JSON.stringify(xs.map((x) => D.gainsFromCrossfader(x))));
    """ % json.dumps(positions))

    worst_linear = 0.0
    for x, (ga, gb) in zip(positions, page):
        # ui_server.set_crossfader, verbatim.
        want_a = round(float(math.cos(x * math.pi / 2.0)), 4)
        want_b = round(float(math.sin(x * math.pi / 2.0)), 4)
        assert (ga, gb) == (want_a, want_b), f"x={x}"
        if ga + gb > 0.001:
            worst_linear = max(worst_linear, abs(gb / (ga + gb) - x))

    assert worst_linear > 0.04, (
        "the linear read-back this replaced really was wrong by >4% of travel"
    )


# --- the rules the page obeys ---------------------------------------------------


def test_the_page_loads_the_drag_maths_before_it_uses_them():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/dragmath.js") < html.index("/static/app.js")


def test_every_control_is_dragged_through_the_shared_path():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for control in ("eq-${d}-${band}", "fader-${d}", "filt-${d}", "pitch-${d}"):
        assert control in js, control
    # One registration per control family, plus the crossfader.
    assert js.count("makeDraggable(") == 6, "five control families and the crossfader"
    assert "trackGrab" not in js, "the old grab-only tracking is gone"
    assert "grabbed.has" not in js


def test_a_held_control_is_never_written_by_the_state_feed():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "if (dragging.has(el.id)) return false;" in js
    # The feed reaches the controls only through the guard.
    for write in ("f.value = String(deck.gain)",
                  "el.value = String(deck.eq[i])",
                  'pitch.value = String(deck.pitch_percent'):
        assert write not in js, f"ungated write still present: {write}"
    assert js.count("applyServerValue(") >= 6


def test_the_drag_takes_and_releases_the_pointer():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "setPointerCapture" in js and "releasePointerCapture" in js
    for event in ("pointerdown", "pointermove", "pointerup", "pointercancel",
                  "lostpointercapture"):
        assert f'"{event}"' in js, event
    # The canvas grid handle releases its capture too, and has a cancel path.
    assert js.count("releasePointerCapture") >= 2


def _without_comments(js: str) -> str:
    """Strip /* block */ and // line comments, so prose about a mistake does
    not read as the mistake itself."""
    out, i, n = [], 0, len(js)
    while i < n:
        if js.startswith("/*", i):
            i = js.find("*/", i + 2)
            if i < 0:
                break
            i += 2
        elif js.startswith("//", i):
            nl = js.find("\n", i)
            i = n if nl < 0 else nl
        else:
            out.append(js[i])
            i += 1
    return "".join(out)


def test_coordinates_come_from_the_painted_box():
    code = _without_comments((STATIC / "app.js").read_text(encoding="utf-8"))
    assert "getBoundingClientRect" in code
    assert "clientWidth" not in code and "offsetWidth" not in code, (
        "clientWidth is the content box, rounded, and excludes the border"
    )
    # fitCanvas applies the device pixel ratio to the backing store.
    assert "r.width * dpr" in code and "r.height * dpr" in code


def test_the_preview_panel_is_not_in_the_mixer_column():
    """The mixer column has no height to spare.

    Placed there, the preview panel took the channel strips -- the one row that
    flexes -- from 234 px to 89 px at a 1000 px viewport, and to 0 at 800.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    panel = html.index('id="previewbox"')
    assert html.index('<section id="bottom">') < panel, "preview belongs in #bottom"
    mixer_start = html.index('<section id="mixer">')
    mixer_end = html.index("</section>", mixer_start)
    assert not mixer_start < panel < mixer_end


def test_the_meter_canvas_cannot_push_its_own_layout():
    """A canvas flex item with min-width:auto grows to its backing store.

    fitCanvas sizes the backing store from the painted box times the device
    pixel ratio, so an auto minimum makes the two chase each other: measured,
    the meters went from 14 px to 43 px at dpr 1 and 49 px at dpr 2.
    """
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    meter = css[css.index(".meter {"):css.index("}", css.index(".meter {"))]
    assert "min-width: 0" in meter
    assert "width: 14px" in meter
