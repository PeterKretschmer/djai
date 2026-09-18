/* djai controller UI. Vanilla JS + Canvas, no build step, no framework.
 *
 * Three channels, deliberately separate:
 *   - plain HTTP POSTs for CUT / KILL BASS / STOP
 *   - plain HTTP POSTs for the manual overrides (freeze / go / cue)
 *   - a WebSocket carrying 20 Hz state, chat, and the controller gestures
 *
 * The panic buttons never touch the socket, so they still work when the model
 * is wedged, the socket is dead, and the numbers on screen are stale. That
 * wiring is unchanged from the previous UI and must stay that way.
 *
 * This file decides nothing musical. Dropping a track sends a "load" and the
 * server decides whether it lands now or at the next phrase; there is no
 * scheduling, transition or selection logic on this side of the socket.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const DECKS = ["a", "b"];
const COLOUR = { a: "#4fc3f7", b: "#ffb74d" };

let state = null;
let ws = null;
const waves = { a: null, b: null };   // waveform payload per deck
const wanted = { a: null, b: null };  // track_id the payload belongs to

/* ---------------- panic: HTTP only, never the socket ---------------- */

async function panic(action, btn) {
  btn.classList.add("armed");
  setTimeout(() => btn.classList.remove("armed"), 250);
  try {
    await fetch(`/panic/${action}`, { method: "POST" });
    log(`${action} sent`, "sys");
  } catch (e) {
    log(`PANIC FAILED: ${e}`, "err");
  }
}
$("btn-cut").onclick = (e) => panic("cut", e.currentTarget);
$("btn-kill").onclick = (e) => panic("killbass", e.currentTarget);
$("btn-stop").onclick = (e) => panic("stop", e.currentTarget);

/* ---------------- manual override: also HTTP, also not the model ---------- */

async function override(action, arg, btn) {
  if (btn) {
    btn.classList.add("armed");
    setTimeout(() => btn.classList.remove("armed"), 250);
  }
  try {
    const q = arg ? `?arg=${encodeURIComponent(arg)}` : "";
    const r = await fetch(`/override/${action}${q}`, { method: "POST" });
    const data = await r.json();
    log(data.text || data.error || action, data.ok ? "sys" : "err");
  } catch (e) {
    log(`override failed: ${e}`, "err");
  }
}
$("btn-freeze").onclick = (e) =>
  override(state && state.frozen ? "resume" : "freeze", "", e.currentTarget);
$("btn-go").onclick = (e) => override("go", "", e.currentTarget);
$("btn-cue-a").onclick = (e) =>
  override("cue", state && state.cue.deck === "a" ? "off" : "a", e.currentTarget);
$("btn-cue-b").onclick = (e) =>
  override("cue", state && state.cue.deck === "b" ? "off" : "b", e.currentTarget);

/* ---------------- controller gestures: the socket ---------------- */

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  else log("not connected", "err");
}

DECKS.forEach((d) => {
  $(`play-${d}`).onclick = () => {
    const deck = state && state.decks[d];
    if (!deck || !deck.title) { log(`deck ${d.toUpperCase()} is empty`, "err"); return; }
    send({ type: "transport", deck: d, playing: !deck.playing });
  };
  $(`cuept-${d}`).onclick = () => send({ type: "cue_point", deck: d });
});

$("btn-resume-auto").onclick = () => send({ type: "automation", held: false });

/* Hot cues: stored markers inside the track. Plain click jumps, shift-click
 * stores the playhead, alt-click clears. Eight fixed slots per deck so the
 * buttons never move around under the operator's hand. */
const HOT_CUE_SLOTS = 8;

function buildHotCues(d) {
  const box = $(`hotcues-${d}`);
  for (let i = 1; i <= HOT_CUE_SLOTS; i++) {
    const btn = document.createElement("button");
    btn.className = "hc";
    btn.id = `hc-${d}-${i}`;
    btn.dataset.index = String(i);
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = String(i);
    const l = document.createElement("span");
    l.className = "l";
    btn.appendChild(n);
    btn.appendChild(l);
    btn.onclick = (e) => {
      const op = e.shiftKey ? "set" : e.altKey ? "clear" : "jump";
      send({ type: "hotcue", deck: d, index: i, op });
    };
    box.appendChild(btn);
  }
}
DECKS.forEach(buildHotCues);

/* ---------------- performance controls ----------------
 * Loops, rolls, beat jump, pitch fader and sync. Every button is one message
 * to the server, which does the grid maths and holds the automation. Rolls
 * are momentary: held down they roll, released they let go in phase. */
function perform(d, op, value) {
  send({ type: "perform", deck: d, op, value });
}

function perfButton(label, title, onClick) {
  const b = document.createElement("button");
  b.className = "btn pf";
  b.textContent = label;
  b.title = title;
  if (onClick) b.onclick = onClick;
  return b;
}

function perfRow(name, children) {
  const row = document.createElement("div");
  row.className = "pfrow";
  const l = document.createElement("span");
  l.className = "lbl";
  l.textContent = name;
  row.appendChild(l);
  children.forEach((c) => row.appendChild(c));
  return row;
}

const sendPitch = throttled((d, v) => perform(d, "pitch", v), 60);

function buildPerf(d) {
  const box = $(`perf-${d}`);
  const exit = perfButton("EXIT", "Leave the loop, in phase", () => perform(d, "loop_exit"));
  exit.id = `loopexit-${d}`;
  box.appendChild(perfRow("LOOP", [
    perfButton("IN", "Loop in", () => perform(d, "loop_in")),
    perfButton("OUT", "Loop out", () => perform(d, "loop_out")),
    exit,
    perfButton("½", "Halve the loop", () => perform(d, "loop_halve")),
    perfButton("×2", "Double the loop", () => perform(d, "loop_double")),
    perfButton("4B", "Four-beat loop", () => perform(d, "auto_loop", 4)),
  ]));

  const rolls = [["1/8", 0.125], ["1/4", 0.25], ["1/2", 0.5], ["1", 1], ["2", 2], ["4", 4]]
    .map(([label, bars]) => {
      const b = perfButton(label, `Roll ${label} bar - hold`);
      b.addEventListener("pointerdown", () => perform(d, "roll", bars));
      const release = () => perform(d, "roll_off");
      b.addEventListener("pointerup", release);
      b.addEventListener("pointerleave", (e) => { if (e.buttons) release(); });
      return b;
    });
  box.appendChild(perfRow("ROLL", rolls));

  box.appendChild(perfRow("JUMP", [-16, -8, -4, -1, 1, 4, 8, 16].map((n) =>
    perfButton(n > 0 ? `+${n}` : String(n), `Beat jump ${n} bars`, () => perform(d, "jump", n))
  )));

  const pitch = document.createElement("input");
  pitch.type = "range";
  pitch.min = "-8"; pitch.max = "8"; pitch.step = "0.05"; pitch.value = "0";
  pitch.id = `pitch-${d}`;
  pitch.className = "pitch";
  makeDraggable(pitch);
  pitch.addEventListener("input", () => {
    $(`pitchval-${d}`).textContent = `${parseFloat(pitch.value).toFixed(2)}%`;
    sendPitch(d, parseFloat(pitch.value));
  });
  const val = document.createElement("span");
  val.className = "num";
  val.id = `pitchval-${d}`;
  val.textContent = "0.00%";
  box.appendChild(perfRow("PITCH", [
    pitch, val,
    perfButton("RESET", "Pitch to 0%", () => perform(d, "pitch_reset")),
    perfButton("SYNC", "Match tempo and phase to the master", () => perform(d, "sync")),
  ]));
}
DECKS.forEach(buildPerf);

$("btn-quantize").onclick = () =>
  perform(state ? state.live_deck : "a", "quantize", !(state && state.quantize));

$("style-pick").onchange = (e) => send({ type: "style", style: e.target.value });
$("phase-pick").onchange = (e) => send({ type: "phase", phase: e.target.value });

/* ---------------- waveform zoom ---------------- */

// Zoom step per deck: 0 is the whole track, the last step is one bar. The
// steps themselves come from WaveMath.zoomSpans, per track.
const zoom = { a: 0, b: 0 };
const zoomMax = { a: 0, b: 0 };
// The window each deck's waveform was last drawn with, for pointer maths.
const lastView = { a: null, b: null };

function setZoom(d, step) {
  zoom[d] = Math.max(0, Math.min(zoomMax[d] || 0, step));
}
DECKS.forEach((d) => {
  $(`wz-in-${d}`).onclick = () => setZoom(d, zoom[d] + 1);
  $(`wz-out-${d}`).onclick = () => setZoom(d, zoom[d] - 1);
});

/* ---------------- beat grid correction ---------------- */

// The first-downbeat handle's position while it is being dragged, per deck.
const gridDrag = { a: null, b: null };

function sendGrid(d, op, extra) {
  send(Object.assign({ type: "grid", deck: d, op }, extra || {}));
  // The server drops its cached waveform on a correction; fetch the new grid
  // and the library's review badges once the change has landed.
  setTimeout(() => { fetchWave(d); loadLibrary(); }, 250);
}

function buildGridControls(d) {
  const box = $(`gridctl-${d}`);
  if (!box) return;
  const controls = [
    ["½", "halve the tempo", () => sendGrid(d, "halve")],
    ["×2", "double the tempo", () => sendGrid(d, "double")],
    ["◀ 10ms", "nudge the grid 10 ms earlier", () => sendGrid(d, "nudge", { ms: -10 })],
    ["10ms ▶", "nudge the grid 10 ms later", () => sendGrid(d, "nudge", { ms: 10 })],
    ["TAP", "tap the tempo: 8 or more steady taps on the beat", () => sendGrid(d, "tap")],
  ];
  for (const [label, title, fn] of controls) {
    const b = document.createElement("button");
    b.className = "btn small grid";
    b.textContent = label;
    b.title = title;
    b.onclick = fn;
    box.appendChild(b);
  }
}

DECKS.forEach((d) => {
  buildGridControls(d);
  const c = $(`wave-${d}`);
  // Pointer maths go through the view the last frame drew, so dragging bar 1
  // works the same at any zoom.
  const secondsAt = (clientX) => {
    const view = lastView[d];
    if (!view) return null;
    const r = c.getBoundingClientRect();
    return WaveMath.t(clientX - r.left, view, r.width);
  };
  c.addEventListener("wheel", (e) => {
    e.preventDefault();
    setZoom(d, zoom[d] + (e.deltaY < 0 ? 1 : -1));
  }, { passive: false });
  // Where the handle was when it was grabbed, and how far the pointer was
  // from it. The handle moves by the pointer's delta, never to the pointer's
  // absolute position: the grab tolerance is 8 px, so an absolute reading
  // teleports the handle by up to 8 px -- nearly 4 s of track at full zoom --
  // before the hand has moved.
  const grab = { seconds: 0, pointer: null, clientX: 0 };

  c.addEventListener("pointerdown", (e) => {
    const wave = waves[d];
    const view = lastView[d];
    if (!wave || !view || !isFinite(wave.first_downbeat)) return;
    const r = c.getBoundingClientRect();
    const handleX = WaveMath.x(wave.first_downbeat, view, r.width);
    if (Math.abs(e.clientX - r.left - handleX) > 8) return;
    gridDrag[d] = wave.first_downbeat;
    grab.seconds = wave.first_downbeat;
    grab.clientX = e.clientX;
    grab.pointer = e.pointerId;
    c.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  c.addEventListener("pointermove", (e) => {
    if (gridDrag[d] === null || e.pointerId !== grab.pointer) return;
    const view = lastView[d];
    if (!view) return;
    const r = c.getBoundingClientRect();
    // Seconds per CSS pixel, from the window the last frame actually drew.
    const perPixel = (view[1] - view[0]) / r.width;
    gridDrag[d] = Math.max(0, grab.seconds + (e.clientX - grab.clientX) * perPixel);
  });

  // pointerup commits; pointercancel -- the gesture taken away mid-drag --
  // abandons it and puts the handle back where it was.
  const endGrid = (commit) => (e) => {
    if (gridDrag[d] === null || (e && e.pointerId !== grab.pointer)) return;
    const seconds = gridDrag[d];
    gridDrag[d] = null;
    if (grab.pointer !== null) {
      try { c.releasePointerCapture(grab.pointer); } catch (err) { /* gone */ }
    }
    grab.pointer = null;
    if (commit) sendGrid(d, "downbeat", { seconds });
  };
  c.addEventListener("pointerup", endGrid(true));
  c.addEventListener("pointercancel", endGrid(false));
});

/* Mixer. Throttled: a drag fires `input` per pixel, and every send becomes a
 * command on the engine queue. 20 Hz matches the state feed and is far finer
 * than a hand can move a fader. */
function throttled(fn, ms) {
  let last = 0, pendingArgs = null, timer = null;
  return (...args) => {
    const now = performance.now();
    if (now - last >= ms) { last = now; fn(...args); return; }
    pendingArgs = args;
    if (!timer) {
      timer = setTimeout(() => {
        timer = null; last = performance.now();
        if (pendingArgs) { fn(...pendingArgs); pendingArgs = null; }
      }, ms - (now - last));
    }
  };
}

const sendEq = throttled((deck, band, v) => {
  const msg = { type: "eq", deck: deck };
  msg[band] = v;
  send(msg);
}, 50);
const sendLevel = throttled((deck, v) => send({ type: "level", deck, gain: v }), 50);
const sendXf = throttled((v) => send({ type: "crossfader", x: v }), 50);
const sendFilter = throttled((deck, v) => send({ type: "filter", deck, position: v }), 50);

/* Filter knob. The slider snaps to exactly 0 near the middle -- the detent --
 * and the label shows where the cutoff is. Readout only: the coefficients
 * that matter come from the deck's own table on the server. */
const FILTER_SNAP = 0.04;
function filterLabel(v) {
  if (Math.abs(v) <= 0.02) return "OFF";
  const m = Math.min(1, Math.abs(v));
  const hz = v < 0 ? 20000 * Math.pow(60 / 20000, m) : 20 * Math.pow(12000 / 20, m);
  const txt = hz >= 1000 ? `${(hz / 1000).toFixed(1)}k` : `${Math.round(hz)}`;
  return `${v < 0 ? "LP" : "HP"} ${txt}`;
}
function showFilter(d, v) { $(`filt-${d}-val`).textContent = filterLabel(v); }

/* Dragging a mixer control.
 *
 * A control the operator is holding must not be yanked around by the state
 * feed -- and must not be yanked around by it on release either. The engine's
 * echo of the last gesture is still in flight at that moment, so lifting the
 * guard the instant the pointer comes up writes a stale value straight into
 * the control: measured on this page, letting go of a fader swung the handle
 * 11% of its travel back up with 60 ms of echo latency, and 58% with 400 ms,
 * before crawling down again. So each control carries a `dragging` flag while
 * it is held, and a reconciliation window afterwards, during which the value
 * the operator set stands until the engine agrees with it -- or until the
 * window expires, after which the engine wins. The server is still the truth;
 * it is just not allowed to overwrite a fresh gesture with a stale echo.
 *
 * The drag itself is handled here rather than left to the native range input,
 * so the value can be computed as the grab value plus the pointer delta. A
 * native range jumps its thumb to the cursor on the first move after a press
 * anywhere on the track: measured here, a press 27% of the travel away from
 * the thumb moved it 27% of the travel before the hand had moved at all.
 */
const dragging = new Set();
const reconciling = new Map();   // control id -> {value, until}

/* Long enough to cover the 50 ms send throttle, the command queue and a
 * couple of 20 Hz state frames; short enough that a control never lies about
 * the engine for long. */
const RECONCILE_MS = 900;

function beginReconcile(el) {
  reconciling.set(el.id, {
    value: parseFloat(el.value),
    until: performance.now() + RECONCILE_MS,
  });
}

function makeDraggable(el) {
  // `orient` is what index.html declares and what style.css selects on, so it
  // decides the axis rather than the measured shape of the box.
  const vertical = el.getAttribute("orient") === "vertical";
  // The browser must not start a gesture of its own -- scroll, text
  // selection, or the native thumb drag -- underneath this one.
  el.style.touchAction = "none";
  let pointer = null, grabValue = 0, originX = 0, originY = 0, rect = null;

  const emit = (v) => {
    if (String(v) === el.value) return;
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };

  el.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    pointer = e.pointerId;
    grabValue = parseFloat(el.value);
    originX = e.clientX;
    originY = e.clientY;
    // getBoundingClientRect, not offsetWidth: CSS pixels, borders included,
    // and the same coordinate space clientX/clientY are measured in.
    rect = el.getBoundingClientRect();
    dragging.add(el.id);
    reconciling.delete(el.id);
    // Without capture, a drag that wanders outside the box stops being
    // delivered and the value freezes until the pointer comes back.
    try { el.setPointerCapture(e.pointerId); } catch (err) { /* no capture */ }
    e.preventDefault();   // suppress the native range drag
    el.focus({ preventScroll: true });
  });

  el.addEventListener("pointermove", (e) => {
    if (pointer === null || e.pointerId !== pointer) return;
    e.preventDefault();
    emit(DragMath.valueFromDrag({
      rect, vertical,
      min: parseFloat(el.min), max: parseFloat(el.max), step: parseFloat(el.step),
      grabValue, originX, originY, clientX: e.clientX, clientY: e.clientY,
    }));
  });

  // pointerup, pointercancel (the gesture taken away mid-drag) and
  // lostpointercapture all end the drag; whichever arrives first wins and the
  // rest are no-ops.
  const end = (e) => {
    if (pointer === null || (e && e.pointerId !== undefined && e.pointerId !== pointer)) return;
    try { el.releasePointerCapture(pointer); } catch (err) { /* already gone */ }
    pointer = null;
    dragging.delete(el.id);
    beginReconcile(el);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  el.addEventListener("lostpointercapture", end);

  // Arrow keys on a focused control produce a value the feed would otherwise
  // overwrite before the engine had echoed it.
  el.addEventListener("input", () => {
    if (!dragging.has(el.id)) beginReconcile(el);
  });
}

/* Write an engine value into a control, unless the operator owns it. Returns
 * whether the write happened. `apply` is for controls that carry a readout
 * alongside the value. */
function applyServerValue(el, value, apply) {
  if (dragging.has(el.id)) return false;
  const pending = reconciling.get(el.id);
  if (pending) {
    const step = parseFloat(el.step) || 0.001;
    if (Math.abs(value - pending.value) <= step * 1.5) {
      reconciling.delete(el.id);            // the engine caught up
    } else if (performance.now() < pending.until) {
      return false;                         // still in flight: hold the gesture
    } else {
      reconciling.delete(el.id);            // waited long enough: engine wins
    }
  }
  if (apply) apply(value); else el.value = String(value);
  return true;
}

DECKS.forEach((d) => {
  ["low", "mid", "high"].forEach((band) => {
    const el = $(`eq-${d}-${band}`);
    makeDraggable(el);
    el.addEventListener("input", () => sendEq(d, band, parseFloat(el.value)));
  });
  const f = $(`fader-${d}`);
  makeDraggable(f);
  f.addEventListener("input", () => sendLevel(d, parseFloat(f.value)));

  const flt = $(`filt-${d}`);
  makeDraggable(flt);
  flt.addEventListener("input", () => {
    let v = parseFloat(flt.value);
    if (Math.abs(v) < FILTER_SNAP) { v = 0; flt.value = "0"; }
    showFilter(d, v);
    sendFilter(d, v);
  });
  flt.addEventListener("dblclick", () => {
    flt.value = "0";
    showFilter(d, 0);
    sendFilter(d, 0);
  });

  $(`keylock-${d}`).onclick = () => {
    const deck = state && state.decks ? state.decks[d] : null;
    send({ type: "keylock", deck: d, on: !(deck && deck.key_lock) });
  };
});
const xf = $("xfader");
makeDraggable(xf);
xf.addEventListener("input", () => sendXf(parseFloat(xf.value)));

/* Keyboard. Escape cuts; the rest mirror the on-screen controls. All ignored
 * while typing, so a chat message never fires the transport. */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { panic("cut", $("btn-cut")); return; }
  const el = document.activeElement;
  if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  switch (e.key.toLowerCase()) {
    case "f": $("btn-freeze").click(); break;
    case "g": $("btn-go").click(); break;
    case "1": $("btn-cue-a").click(); break;
    case "2": $("btn-cue-b").click(); break;
    case "q": $("play-a").click(); break;
    case "p": $("play-b").click(); break;
  }
});

/* ---------------- library ---------------- */

let library = [];
let sortKey = "title";
let sortDesc = false;

async function loadLibrary() {
  try {
    const r = await fetch("/api/library");
    library = (await r.json()).tracks || [];
    renderLibrary();
  } catch (e) {
    log(`library failed: ${e}`, "err");
  }
}

function mmss(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function renderLibrary() {
  const q = $("lib-filter").value.trim().toLowerCase();
  const rows = library.filter(
    (t) => !q || t.title.toLowerCase().includes(q) ||
           (t.camelot || "").toLowerCase().includes(q)
  );
  rows.sort((x, y) => {
    const a = x[sortKey], b = y[sortKey];
    const c = typeof a === "number" ? a - b : String(a).localeCompare(String(b));
    return sortDesc ? -c : c;
  });

  const box = $("lib-rows");
  box.textContent = "";
  for (const t of rows) {
    const el = document.createElement("div");
    el.className = "trk";
    el.draggable = true;
    el.dataset.trackId = t.track_id;
    el.title = t.warn
      ? `grid confidence ${t.grid_confidence}` +
        (t.estimated ? ", mix points estimated" : "")
      : t.title;

    const parts = [
      ["t", t.title],
      ["n", t.bpm.toFixed(1)],
      ["n", t.camelot || "--"],
      ["n", mmss(t.duration_s)],
      ["w", t.warn ? "⚠" : ""],
    ];
    for (const [cls, text] of parts) {
      const s = document.createElement("span");
      s.className = cls;
      s.textContent = text;
      el.appendChild(s);
    }
    if (t.review) {
      // Inside the title cell, so the row's column layout is unchanged.
      const badge = document.createElement("span");
      badge.className = "rv";
      badge.textContent = "REVIEW";
      badge.title = t.tempo_ambiguous
        ? "tempo may be half-time: check the grid"
        : "weak beat grid: check it before a set";
      el.firstChild.prepend(badge);
    }
    el.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", t.track_id);
      e.dataTransfer.effectAllowed = "copy";
    });
    box.appendChild(el);
  }
  $("lib-count").textContent = `${rows.length} / ${library.length}`;
}

$("lib-filter").addEventListener("input", renderLibrary);
document.querySelectorAll(".btn.sort").forEach((btn) => {
  btn.onclick = () => {
    const key = btn.dataset.key;
    sortDesc = sortKey === key ? !sortDesc : false;
    sortKey = key;
    document.querySelectorAll(".btn.sort").forEach((b) =>
      b.classList.toggle("on", b === btn)
    );
    renderLibrary();
  };
});
$("lib-toggle").onclick = () => $("bottom").classList.toggle("collapsed");

/* Drop targets: the whole deck panel. */
DECKS.forEach((d) => {
  const panel = $(`deck-${d}`);
  panel.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    panel.classList.add("dragover");
  });
  panel.addEventListener("dragleave", () => panel.classList.remove("dragover"));
  panel.addEventListener("drop", (e) => {
    e.preventDefault();
    panel.classList.remove("dragover");
    const id = e.dataTransfer.getData("text/plain");
    if (id) send({ type: "load", deck: d, track_id: id });
  });
});

/* ---------------- chat ---------------- */

function log(text, cls) {
  const el = document.createElement("div");
  el.className = "msg " + (cls || "dj");
  el.textContent = text;
  const box = $("log");
  box.appendChild(el);
  while (box.childNodes.length > 200) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

$("entry").onsubmit = (e) => {
  e.preventDefault();
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  log(text, "you");
  send({ type: "text", text });
  input.value = "";
};

/* ---------------- socket ---------------- */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => log("connected", "sys");
  ws.onclose = () => {
    log("disconnected - retrying", "err");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "reply") { if (msg.text) log(msg.text); return; }
    if (msg.type === "ack") {
      if (!msg.ok) log(msg.error || "rejected", "err");
      return;
    }
    if (msg.type === "library") { library = msg.tracks || []; renderLibrary(); return; }
    state = msg;
    (msg.notices || []).forEach((n) => log(n, "sys"));
    render();
  };
}

/* ---------------- render ---------------- */

function fmtDrift(ms) {
  if (ms === null || ms === undefined) return ["--", ""];
  const a = Math.abs(ms);
  return [`${ms.toFixed(1)} ms`, a < 10 ? "ok" : a < 30 ? "warn" : "bad"];
}

/* Camelot compatibility: same key, neighbour number, or the relative
 * major/minor. Display only -- it changes nothing about what gets played. */
function keyCompat(x, y) {
  if (!x || !y) return ["--", ""];
  const px = /^(\d+)([AB])$/.exec(x.toUpperCase());
  const py = /^(\d+)([AB])$/.exec(y.toUpperCase());
  if (!px || !py) return [`${x} / ${y}`, ""];
  const nx = +px[1], ny = +py[1];
  const step = Math.min((nx - ny + 12) % 12, (ny - nx + 12) % 12);
  const good = (step === 0 && px[2] === py[2]) ||
               (step === 0 && px[2] !== py[2]) ||
               (step === 1 && px[2] === py[2]);
  return [`${x} ${good ? "✓" : "✗"} ${y}`, good ? "ok" : "warn"];
}

/* The transition preview: what the pre-roll render measured, and what the
 * revision changed. Read-only -- nothing on this panel steers anything, and
 * the transition has already been committed by the time it appears. */
const PREVIEW_LABEL = {
  running: "previewing...",
  committed: "committed",
  revised: "revised",
  reverted: "reverted",
  budget: "over budget - committed",
  failed: "preview failed - original",
  cut: "grid clash - forced cut",
  skipped: "skipped",
};

/* Which measurements to show, in the order an operator would want them, with
 * the number of decimals each deserves. */
const PREVIEW_ROWS = [
  ["peak_dbfs", "peak", 1, "dB"],
  ["integrated_lufs", "lufs", 1, ""],
  ["loudness_dip_db", "dip", 1, "dB"],
  ["low_end_overlap_bars", "low overlap", 0, "b"],
  ["vocal_overlap_bars", "vocals", 0, "b"],
  ["spectral_clash", "clash", 2, ""],
  ["transient_density_ratio", "transients", 2, ""],
  ["phase_coherence", "phase", 2, ""],
];

function renderPreview(p) {
  const box = $("previewbox");
  if (!box) return;
  const status = $("prev-status");
  if (!p) {
    // The panel stays in place, so nothing moves when the first one arrives.
    status.textContent = "no preview yet";
    status.className = "pstate";
    return;
  }

  status.textContent = PREVIEW_LABEL[p.status] || p.status || "-";
  status.className = "pstate " + (p.status || "");
  $("prev-time").textContent = p.elapsed_ms
    ? `${Math.round(p.elapsed_ms)}/${Math.round(p.budget_ms || 0)} ms` +
      (p.rounds ? `, ${p.rounds} rnd` : "")
    : "";
  $("prev-time").title = p.elapsed_ms
    ? `${Math.round(p.elapsed_ms)} ms of a ${Math.round(p.budget_ms || 0)} ms budget, ` +
      `${p.rounds} measured round${p.rounds === 1 ? "" : "s"}`
    : "";

  // The measurements, with the first round's value alongside the last when
  // they differ: the point of the panel is what the revision moved.
  const meas = $("prev-meas");
  meas.textContent = "";
  const first = p.first || {};
  const final = p.final || {};
  for (const [key, label, places, unit] of PREVIEW_ROWS) {
    const now = final[key] !== undefined ? final[key] : first[key];
    if (now === undefined || now === null) continue;
    const was = first[key];
    const row = document.createElement("div");
    row.className = "mrow";
    const l = document.createElement("span");
    l.className = "ml";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "mv";
    const moved = was !== undefined && was !== null && was !== now;
    v.textContent = moved
      ? `${was.toFixed(places)} -> ${now.toFixed(places)}${unit}`
      : `${now.toFixed(places)}${unit}`;
    if (moved) v.classList.add("moved");
    row.appendChild(l);
    row.appendChild(v);
    meas.appendChild(row);
  }

  // What changed, and which rule changed it.
  const diff = $("prev-diff");
  diff.textContent = "";
  for (const r of p.revisions || []) {
    const el = document.createElement("div");
    el.className = "drow";
    el.textContent =
      `${r.field}: ${JSON.stringify(r.before)} -> ${JSON.stringify(r.after)}`;
    el.title = `${r.rule}: ${r.because}`;
    diff.appendChild(el);
  }
  for (const note of p.events || []) {
    const el = document.createElement("div");
    el.className = "drow dim";
    el.textContent = note;
    diff.appendChild(el);
  }
}

function render() {
  if (!state) return;

  // held banner
  const held = !!state.automation_held;
  $("held").hidden = !held;
  $("held-text").textContent = state.forced_next
    ? `AUTOMATION HELD - manual control (next when you say go: ${state.forced_next})`
    : "AUTOMATION HELD - you have manual control";
  $("btn-freeze").textContent = state.frozen ? "RESUME" : "FREEZE";
  $("btn-freeze").classList.toggle("on", !!state.frozen);
  $("btn-quantize").classList.toggle("on", !!state.quantize);

  // master bar
  $("master-bpm").textContent = state.master_bpm.toFixed(1);
  const [kc, kcls] = keyCompat(state.decks.a.camelot, state.decks.b.camelot);
  const kEl = $("key-compat");
  kEl.textContent = kc;
  kEl.className = "num " + kcls;

  // Master peak, in dBFS. Red when the mix is hot enough that the limiter is
  // pulling it down: that is the moment an operator wants to reach for a
  // fader, and it is invisible from the output level alone because the
  // limiter is busy hiding it.
  const eng = state.engine || {};
  const peakIn = eng.peak_in || 0;
  const db = peakIn > 1e-5 ? 20 * Math.log10(peakIn) : -99;
  const pk = $("s-peak");
  pk.textContent = db <= -99 ? "--" : `${db > 0 ? "+" : ""}${db.toFixed(1)} dB`;
  const reducing = (eng.limiter_gain !== undefined && eng.limiter_gain < 0.999);
  pk.className =
    "num " + (peakIn > (eng.ceiling || 0.98) ? "bad" : reducing ? "warn" : "ok");
  pk.title =
    `output ${(eng.peak || 0).toFixed(3)}, limiter gain ` +
    `${(eng.limiter_gain !== undefined ? eng.limiter_gain : 1).toFixed(3)}` +
    (eng.limiter_clips ? `, backstop fired ${eng.limiter_clips}x` : "");

  const live = state.live_deck;
  const [dtext, dcls] = fmtDrift(state.drift_ms ? state.drift_ms[live] : null);
  const dEl = $("s-drift");
  dEl.textContent = dtext;
  dEl.className = "num " + dcls;

  const llm = $("s-llm");
  llm.textContent = "";
  const dot = document.createElement("i");
  dot.className = "dot" + (state.llm.available ? " ok" : "");
  llm.appendChild(dot);
  llm.appendChild(
    document.createTextNode(state.llm.available ? "online" : "offline")
  );

  const cueOff = state.cue.mode === "none";
  DECKS.forEach((k) => {
    const btn = $(`btn-cue-${k}`);
    btn.classList.toggle("on", state.cue.deck === k);
    btn.disabled = cueOff;
    btn.title = cueOff
      ? "No cue output configured (--cue-device / --cue-channels)"
      : `Pre-listen deck ${k.toUpperCase()} on ${state.cue.mode}`;
  });

  DECKS.forEach((d) => {
    const deck = state.decks[d];

    $(`title-${d}`).textContent = deck.title || "no track loaded";
    const badge = $(`badge-${d}`);
    badge.textContent = deck.live ? "LIVE" : deck.title ? "CUED" : "-";
    badge.className = "badge " + (deck.live ? "live" : deck.title ? "cued" : "");

    $(`bpm-${d}`).textContent = deck.bpm ? deck.bpm.toFixed(1) : "---.-";
    const pct = (deck.rate - 1) * 100;
    $(`stretch-${d}`).textContent =
      Math.abs(pct) < 0.05 ? "0.0%" : `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
    $(`key-${d}`).textContent = deck.camelot || "--";
    $(`elapsed-${d}`).textContent = mmss(deck.position_s);
    const rem = $(`remain-${d}`);
    rem.textContent = mmss(deck.remaining_s);
    rem.classList.toggle("hot", deck.playing && deck.remaining_s < 30);

    const play = $(`play-${d}`);
    play.textContent = deck.playing ? "PAUSE" : "PLAY";
    play.classList.toggle("playing", !!deck.playing);
    play.classList.toggle("paused", !deck.playing && !!deck.title);

    const sync = $(`sync-${d}`);
    sync.classList.toggle("on", !!deck.stretched);
    sync.textContent = deck.stretched ? "SYNC" : "OFF";

    const pend = state.pending_loads && state.pending_loads[d];
    const pel = $(`pending-${d}`);
    pel.hidden = !pend;
    if (pend) {
      pel.textContent =
        `loads at next phrase: ${pend.title} (${pend.bars_until} bars)`;
    }

    // Sliders follow the engine unless the operator is holding them, or has
    // just let go and the engine has not caught up yet.
    applyServerValue($(`fader-${d}`), deck.gain);
    ["low", "mid", "high"].forEach((band, i) => {
      applyServerValue($(`eq-${d}-${band}`), deck.eq[i]);
    });
    const flt = $(`filt-${d}`);
    applyServerValue(flt, deck.filter || 0, (v) => {
      flt.value = String(v);
      showFilter(d, v);
    });
    $(`keylock-${d}`).classList.toggle("on", !!deck.key_lock);

    // Performance panel follows the deck.
    $(`loopexit-${d}`).classList.toggle("on", !!deck.loop_active);
    $(`loopexit-${d}`).textContent = deck.loop_active ? `EXIT ${deck.loop_beats}B` : "EXIT";
    const pitch = $(`pitch-${d}`);
    applyServerValue(pitch, deck.pitch_percent || 0, (v) => {
      pitch.value = String(v);
      $(`pitchval-${d}`).textContent = `${v.toFixed(2)}%`;
    });

    // Hot cue slots follow the loaded track.
    const cues = {};
    for (const c of deck.hot_cues || []) cues[c.index] = c;
    for (let i = 1; i <= HOT_CUE_SLOTS; i++) {
      const btn = $(`hc-${d}-${i}`);
      const cue = cues[i];
      btn.classList.toggle("set", !!cue);
      btn.disabled = !deck.title;
      btn.querySelector(".l").textContent = cue ? cue.label : "";
      btn.title = cue
        ? `${cue.label} at ${mmss(cue.sample_position / 44100)} - ` +
          "click to jump, shift-click to move here, alt-click to clear"
        : "empty - shift-click to store the playhead here";
    }

    if (deck.track_id !== wanted[d]) {
      wanted[d] = deck.track_id;
      waves[d] = null;
      if (deck.track_id) fetchWave(d);
    }
  });

  renderPreview((state.transition || {}).preview);

  // Transition indicator: what is planned, and which rule chose it.
  const t = state.transition || {};
  const running = t.active;
  $("trans-style").textContent = running
    ? `${(t.style || "").replace(/_/g, " ")} (running)`
    : (t.planned_style || t.requested_style || "auto").replace(/_/g, " ");
  $("trans-rule").textContent = t.planned_rule || "";

  // Per-session diversity. A silent fallback and a dull model look identical
  // from the outside, so both numbers are shown.
  const sources = t.sources || {};
  const total = Object.values(sources).reduce((a, b) => a + b, 0);
  const fromModel = sources.llm || 0;
  const styleBits = Object.entries(t.style_counts || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
    .join(", ");
  $("trans-diversity").textContent = total
    ? `${fromModel}/${total} designed (${Math.round((100 * fromModel) / total)}%)` +
      (styleBits ? ` - ${styleBits}` : "")
    : "no transitions yet";
  // Set phase: options from the server, selection follows the session unless
  // the operator is using the control.
  const phasePick = $("phase-pick");
  const phases = t.set_phases || [];
  if (phases.length && phasePick.options.length !== phases.length) {
    phasePick.replaceChildren(...phases.map((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = `phase: ${p}`;
      return opt;
    }));
  }
  if (t.set_phase && document.activeElement !== phasePick) phasePick.value = t.set_phase;

  const pick = $("style-pick");
  // The server's list is the only list. Rebuilt when its length changes,
  // which in practice is once, on the first update.
  const styles = t.styles || [];
  if (styles.length && pick.options.length !== styles.length) {
    const ordered = ["auto", ...styles.filter((s) => s !== "auto")];
    pick.replaceChildren(...ordered.map((s) => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s === "auto" ? "auto (rules decide)" : s.replace(/_/g, " ");
      return opt;
    }));
  }
  if (document.activeElement !== pick && t.requested_style) {
    pick.value = t.requested_style;
  }

  // Crossfader follows the balance of the two gains unless it is being held.
  // Read back through the same equal-power law the server sets them with: a
  // linear gb / (ga + gb) is wrong by up to 4% of the travel, which the
  // handle then jumps by as soon as it is released.
  applyServerValue(
    $("xfader"),
    DragMath.crossfaderFromGains(state.decks.a.gain, state.decks.b.gain)
  );

  // Up next: what the scheduler is holding, with a bars countdown.
  const q = $("queue-list");
  q.textContent = "";
  const items = (state.queue || []).slice(0, 4);
  if (!items.length) {
    const el = document.createElement("div");
    el.className = "qrow dim";
    el.textContent = state.cued ? `cued: ${state.cued}` : "nothing queued";
    q.appendChild(el);
  } else {
    for (const it of items) {
      const el = document.createElement("div");
      el.className = "qrow";
      el.textContent = `${it.bars_until} bars - ${it.command}`;
      q.appendChild(el);
    }
  }

  draw();
}

async function fetchWave(d) {
  try {
    const r = await fetch(`/api/waveform/${d}`);
    const data = await r.json();
    if (data.track_id !== wanted[d]) return;
    waves[d] = data;
    bands[d] = { trackId: data.track_id, loaded: {}, pending: {} };
    zoom[d] = Math.min(zoom[d], WaveMath.zoomSpans(data.duration_s || 1, data.bpm).length - 1);
    fetchBands(d, "overview");
  } catch (e) {
    log(`waveform ${d} failed: ${e}`, "err");
  }
}

/* Coloured bands: one fetch per detail level, on demand. Each is three rows of
 * uint8 peaks -- low, mid, high -- `points` long. */
const bands = { a: null, b: null };

async function fetchBands(d, key) {
  const b = bands[d];
  if (!b || b.loaded[key] || b.pending[key]) return;
  b.pending[key] = true;
  try {
    const r = await fetch(`/api/waveform/${d}/bands/${key}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = new Uint8Array(await r.arrayBuffer());
    const pps = parseFloat(r.headers.get("X-Points-Per-Second"));
    const n = parseInt(r.headers.get("X-Points"), 10);
    if (bands[d] === b) b.loaded[key] = { data, pps, n };
  } catch (e) {
    log(`waveform bands ${d}/${key} failed: ${e}`, "err");
  } finally {
    b.pending[key] = false;
  }
}

/* The band data to draw at this many pixels per second: the coarsest level
 * that still has a point per pixel. A finer level is fetched the first time a
 * zoom needs it; until it arrives the best loaded one is drawn. */
function pickBands(d, pixelsPerSecond) {
  const b = bands[d];
  const wave = waves[d];
  if (!b || !wave || !wave.bands) return null;
  const options = [{ key: "overview", points_per_second: (wave.bands.overview_points || 1) / (wave.duration_s || 1) }]
    .concat((wave.bands.levels || []).map((lv) => ({ key: String(lv.level), points_per_second: lv.points_per_second })));
  const want = options[WaveMath.levelFor(pixelsPerSecond, options)];
  if (b.loaded[want.key]) return b.loaded[want.key];
  fetchBands(d, want.key);
  let best = null;
  for (const o of options) {
    const got = b.loaded[o.key];
    if (got && (!best || Math.abs(got.pps - pixelsPerSecond) < Math.abs(best.pps - pixelsPerSecond))) best = got;
  }
  return best;
}

/* ---------------- canvas ---------------- */

/* Match the backing store to the box the browser actually painted, times the
 * device pixel ratio.
 *
 * getBoundingClientRect, not clientWidth: clientWidth is the content box,
 * rounded to whole pixels and excluding the border. The meters are 14 CSS
 * pixels wide including a 1 px border each side, so clientWidth sized their
 * backing store to 12 -- a 14 % scale error between the pixels drawn and the
 * box they were drawn into, at every device pixel ratio. */
function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return dpr;
}

/* Low, mid, high: the deck EQ's own three bands. */
const BAND_COLOUR = ["#1e88e5", "#ffa726", "#f5f5f5"];
const BAND_SCALE = [1.0, 0.8, 0.6];
const SECTION_COLOUR = {
  intro: "#5c6bc0", build: "#fdd835", drop: "#e53935", breakdown: "#26a69a", outro: "#8d6e63",
};

// When the last state frame arrived, on this page's clock: the playhead is
// interpolated from there so a 20 Hz feed still scrolls smoothly at 60 fps.
let stateAt = 0;

function playheadNow(d) {
  const deck = state && state.decks[d];
  const wave = waves[d];
  if (!deck) return 0;
  let t = deck.position_s;
  if (deck.playing && stateAt) t += ((performance.now() - stateAt) / 1000) * (deck.rate || 1);
  return wave ? Math.min(t, wave.duration_s || t) : t;
}

function drawWave(d) {
  const c = $(`wave-${d}`);
  const dpr = fitCanvas(c);
  const g = c.getContext("2d");
  const W = c.width, H = c.height;
  g.fillStyle = "#0a0d11";
  g.fillRect(0, 0, W, H);

  const wave = waves[d];
  const deck = state && state.decks[d];
  if (!wave || !deck) { lastView[d] = null; return; }
  const dur = wave.duration_s || 1;

  // The view: whole track at step 0, down to exactly one bar, centred on the
  // playhead once zoomed so the waveform scrolls under a fixed marker.
  const spans = WaveMath.zoomSpans(dur, wave.bpm);
  zoomMax[d] = spans.length - 1;
  const step = Math.min(zoom[d], zoomMax[d]);
  const playhead = playheadNow(d);
  const view = WaveMath.view(dur, spans[step], playhead);
  lastView[d] = [view[0], view[1]];
  const X = (t) => WaveMath.x(t, view, W);
  const bar = WaveMath.barSeconds(wave.bpm);
  const label = $(`wz-lbl-${d}`);
  if (label) {
    label.textContent = step === 0 ? "full"
      : spans[step] <= bar + 1e-9 ? "1 bar" : `${Math.round(spans[step] / bar)} bars`;
  }

  // mix-in / mix-out shading: the region the automation actually uses.
  g.fillStyle = "rgba(255,255,255,0.05)";
  const mi = X(wave.mix_in), mo = X(wave.mix_out);
  if (mi > 0) g.fillRect(0, 0, Math.min(W, mi), H);
  if (mo < W) g.fillRect(Math.max(0, mo), 0, W - Math.max(0, mo), H);

  // section strip along the top, labelled where there is room
  const strip = 12 * dpr;
  g.font = `${9 * dpr}px sans-serif`;
  for (const s of wave.sections || []) {
    const x0 = Math.max(0, X(s.start_s)), x1 = Math.min(W, X(s.end_s));
    if (x1 <= 0 || x0 >= W || x1 <= x0) continue;
    g.fillStyle = SECTION_COLOUR[s.label] || "#555";
    g.fillRect(x0, 0, x1 - x0, strip);
    if (x1 - x0 > 44 * dpr) {
      g.fillStyle = "#0a0d11";
      g.fillText(s.label.toUpperCase(), x0 + 3 * dpr, strip - 3 * dpr);
    }
  }

  // bands: one filled path per band, one column per pixel
  const mid = strip + (H - strip) / 2;
  const half = (H - strip) * 0.46;
  const src = pickBands(d, W / (view[1] - view[0]));
  if (src) {
    for (let band = 0; band < 3; band++) {
      const row = src.n * band;
      const scale = (half * BAND_SCALE[band]) / 255;
      g.fillStyle = BAND_COLOUR[band];
      g.beginPath();
      for (let col = 0; col < W; col++) {
        const [p0, p1] = WaveMath.bucket(col, view, W, src.pps);
        const lo = Math.max(0, p0), hi = Math.min(src.n, p1);
        let peak = 0;
        for (let p = lo; p < hi; p++) { const v = src.data[row + p]; if (v > peak) peak = v; }
        if (peak) { const h = peak * scale; g.rect(col, mid - h, 1, 2 * h); }
      }
      g.fill();
    }
  }

  // the grid: every beat when there is room, downbeats, 8-bar phrase lines
  const beats = wave.beats || [];
  const [b0, b1] = WaveMath.inView(beats, view);
  const beatPx = beats.length > 1 ? W / ((view[1] - view[0]) / (60 / wave.bpm)) : 0;
  if (beatPx >= 4 * dpr) {
    g.strokeStyle = "rgba(255,255,255,0.12)";
    g.lineWidth = 1 * dpr;
    g.beginPath();
    for (let i = b0; i < b1; i++) { const px = X(beats[i]); g.moveTo(px, strip); g.lineTo(px, H); }
    g.stroke();
  }
  const downs = wave.downbeats || [];
  const [d0, d1] = WaveMath.inView(downs, view);
  g.strokeStyle = "rgba(255,255,255,0.28)";
  g.lineWidth = 1 * dpr;
  g.beginPath();
  for (let i = d0; i < d1; i++) { const px = X(downs[i]); g.moveTo(px, H * 0.8); g.lineTo(px, H); }
  g.stroke();
  const phrases = wave.phrases || [];
  const [q0, q1] = WaveMath.inView(phrases, view);
  g.strokeStyle = "rgba(255,255,255,0.55)";
  g.beginPath();
  for (let i = q0; i < q1; i++) { const px = X(phrases[i]); g.moveTo(px, strip); g.lineTo(px, H); }
  g.stroke();

  // hot cues, from the live state so a cue set a moment ago shows at once
  g.font = `${9 * dpr}px sans-serif`;
  for (const cue of deck.hot_cues || []) {
    const px = X(cue.sample_position / 44100);
    if (px < -8 * dpr || px > W + 8 * dpr) continue;
    g.fillStyle = cue.label === "drop" ? "#e53935" : "#ab47bc";
    g.fillRect(px - 0.5 * dpr, strip, 1 * dpr, H - strip);
    g.beginPath();
    g.moveTo(px, strip); g.lineTo(px + 9 * dpr, strip + 6 * dpr); g.lineTo(px, strip + 12 * dpr);
    g.closePath(); g.fill();
    g.fillStyle = "#fff";
    g.fillText(String(cue.index), px + 2 * dpr, strip + 9 * dpr);
  }

  // bar 1: a draggable handle. Blue once a person has corrected the grid.
  const barOne = gridDrag[d] !== null ? gridDrag[d] : wave.first_downbeat;
  if (isFinite(barOne)) {
    const px = X(barOne);
    if (px >= -6 * dpr && px <= W + 6 * dpr) {
      g.fillStyle = wave.grid_corrected ? "#4fc3f7" : "#ffd54f";
      g.fillRect(px - 0.5 * dpr, 0, 1 * dpr, H);
      g.beginPath();
      g.moveTo(px - 6 * dpr, 0); g.lineTo(px + 6 * dpr, 0); g.lineTo(px, 10 * dpr);
      g.closePath();
      g.fill();
    }
  }

  // mix points
  for (const [t, col] of [[wave.mix_in, "#4caf50"], [wave.mix_out, "#e53935"]]) {
    const px = X(t);
    if (px < 0 || px > W) continue;
    g.strokeStyle = col;
    g.lineWidth = 2 * dpr;
    g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke();
  }

  // playhead
  const ph = X(playhead);
  g.strokeStyle = "#fff";
  g.lineWidth = 2 * dpr;
  g.beginPath();
  g.moveTo(ph, 0); g.lineTo(ph, H);
  g.stroke();
}

/* A turntable at 33 1/3 rpm turns once every 1.8 s. Driving the angle from the
 * deck's own playhead means a paused deck stops dead and a stretched deck
 * turns at its actual rate, with no animation loop of its own. */
function drawPlatter(d) {
  const c = $(`platter-${d}`);
  fitCanvas(c);              // CSS decides the size; match the backing store
  const g = c.getContext("2d");
  const W = c.width, H = c.height;
  const cx = W / 2, cy = H / 2, r = Math.min(cx, cy) - 6;
  g.clearRect(0, 0, W, H);

  const deck = state && state.decks[d];
  g.fillStyle = "#14181e";
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
  g.strokeStyle = deck && deck.playing ? COLOUR[d] : "#39424f";
  g.lineWidth = 3;
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.stroke();

  g.strokeStyle = "#2a313b";
  g.lineWidth = 1;
  for (const rr of [r * 0.75, r * 0.5, r * 0.25]) {
    g.beginPath(); g.arc(cx, cy, rr, 0, Math.PI * 2); g.stroke();
  }

  if (!deck || !deck.title) return;
  const angle = ((deck.position_s % 1.8) / 1.8) * Math.PI * 2;
  g.strokeStyle = COLOUR[d];
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(cx, cy);
  g.lineTo(cx + Math.sin(angle) * r * 0.92, cy - Math.cos(angle) * r * 0.92);
  g.stroke();

  g.fillStyle = "#0a0d11";
  g.beginPath(); g.arc(cx, cy, r * 0.16, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "#39424f";
  g.lineWidth = 2;
  g.beginPath(); g.arc(cx, cy, r * 0.16, 0, Math.PI * 2); g.stroke();
}

/* Level meter. Derived from the cached peak envelope at the playhead times the
 * deck's gain -- a view-side estimate. Nothing here reads the audio path. */
function drawMeter(d) {
  const c = $(`meter-${d}`);
  fitCanvas(c);
  const g = c.getContext("2d");
  const W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#0a0d11";
  g.fillRect(0, 0, W, H);

  const wave = waves[d], deck = state && state.decks[d];
  if (!wave || !wave.peaks || !wave.peaks.length || !deck || !deck.playing) return;
  const i = Math.max(0, Math.min(
    wave.peaks.length - 1,
    Math.floor((deck.position_s / (wave.duration_s || 1)) * wave.peaks.length)
  ));
  const level = Math.max(0, Math.min(1, wave.peaks[i] * deck.gain));
  const h = level * H;
  const grad = g.createLinearGradient(0, H, 0, 0);
  grad.addColorStop(0, "#4caf50");
  grad.addColorStop(0.75, "#ffb300");
  grad.addColorStop(1, "#e53935");
  g.fillStyle = grad;
  g.fillRect(0, H - h, W, h);
}

/* Drawing runs on requestAnimationFrame, not on the state feed: the feed only
 * updates the numbers, and every frame both decks' canvases are drawn with the
 * playhead interpolated between feed frames. draw() just notes when fresh
 * state arrived. */
function draw() {
  stateAt = performance.now();
}

/* Frame timing, measured by the page itself: the interval between frames and
 * how long drawing both decks took. Reported to the server every five seconds
 * and readable here as window.djaiFrameStats(). */
const frameTimes = { intervals: [], draws: [], last: 0, sentAt: 0 };

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

window.djaiFrameStats = () => {
  const iv = frameTimes.intervals;
  const p50 = percentile(iv, 0.5);
  return {
    fps: p50 ? 1000 / p50 : 0,
    p50_ms: p50,
    p95_ms: percentile(iv, 0.95),
    p99_ms: percentile(iv, 0.99),
    max_ms: iv.length ? Math.max(...iv) : 0,
    draw_p95_ms: percentile(frameTimes.draws, 0.95),
    frames: iv.length,
    decks_playing: state ? DECKS.filter((d) => state.decks[d] && state.decks[d].playing).length : 0,
  };
};

function frameLoop(now) {
  if (state) {
    const t0 = performance.now();
    DECKS.forEach((d) => { drawWave(d); drawPlatter(d); drawMeter(d); });
    frameTimes.draws.push(performance.now() - t0);
    if (frameTimes.draws.length > 600) frameTimes.draws.shift();
  }
  if (frameTimes.last) {
    frameTimes.intervals.push(now - frameTimes.last);
    if (frameTimes.intervals.length > 600) frameTimes.intervals.shift();
  }
  frameTimes.last = now;
  if (now - frameTimes.sentAt > 5000 && frameTimes.intervals.length >= 60
      && ws && ws.readyState === WebSocket.OPEN) {
    frameTimes.sentAt = now;
    ws.send(JSON.stringify(Object.assign({ type: "frame_stats" }, window.djaiFrameStats())));
  }
  requestAnimationFrame(frameLoop);
}
requestAnimationFrame(frameLoop);

window.addEventListener("resize", () => { if (state) draw(); });

connect();
loadLibrary();
document.querySelector('.btn.sort[data-key="title"]').classList.add("on");
