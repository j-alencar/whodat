"use strict";

const DATA = JSON.parse(document.getElementById("data").textContent);
const root = document.querySelector(".viz-root");
const canvas = document.getElementById("canvas");
const tip = document.getElementById("tip");
const rowIndex = Object.fromEntries(DATA.rows.map((row, i) => [row.id, i]));

function clock(seconds) {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Rows carry a light and a dark color step; null means neutral.
function applyColors() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  DATA.rows.forEach((row, i) => {
    const color = row.light ? (dark ? row.dark : row.light) : "var(--neutral)";
    root.style.setProperty(`--row-${i}`, color);
  });
}

function drawNames() {
  const names = document.getElementById("names");
  const total = DATA.totals.reduce((sum, [, seconds]) => sum + seconds, 0) || 1;
  const stats = Object.fromEntries(DATA.totals.map(([id, seconds, turns]) => [
    id,
    `${clock(seconds)} · ${((100 * seconds) / total).toFixed(1)}% · ${turns} turns`,
  ]));
  for (const [i, row] of DATA.rows.entries()) {
    const line = element("div");
    const swatch = element("span", row.id === "review" ? "swatch review" : "swatch");
    if (row.id !== "review") swatch.style.background = `var(--row-${i})`;
    line.append(swatch, element("span", "", row.name));
    if (stats[row.id]) line.append(element("span", "stats", stats[row.id]));
    names.append(line);
  }
  names.append(element("div"));  // spacer beside the axis
}

function showTip(event, turn) {
  const [start, end, id, score, hint, cluster] = turn;
  tip.replaceChildren(
    element("b", "", DATA.rows[rowIndex[id]].name),
    `${clock(start)} - ${clock(end)}`,
    element("br"),
    `score ${score.toFixed(2)}` + (hint ? ` (top: ${DATA.names[hint] || hint})` : ""),
    element("br"),
    element("span", "muted", `cluster ${cluster}`),
  );
  tip.style.display = "block";
  tip.style.left = `${Math.min(event.clientX + 12, innerWidth - 220)}px`;
  tip.style.top = `${event.clientY + 12}px`;
}

function drawBars() {
  const lanes = DATA.rows.map(() => canvas.appendChild(element("div", "lane")));
  for (const turn of DATA.turns) {
    const [start, end, id] = turn;
    const i = rowIndex[id];
    const bar = element("div", id === "review" ? "bar review" : "bar");
    bar.style.left = `${(100 * start) / DATA.duration}%`;
    bar.style.width = `max(2px, ${(100 * (end - start)) / DATA.duration}%)`;
    if (id !== "review") bar.style.background = `var(--row-${i})`;
    bar.addEventListener("mousemove", (event) => showTip(event, turn));
    bar.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    bar.addEventListener("click", () =>
      window.open(`https://youtu.be/${DATA.video}?t=${Math.floor(start)}`));
    lanes[i].append(bar);
  }
}

const axis = element("div", "axis");

function drawAxis() {
  const width = canvas.getBoundingClientRect().width;
  const step = [60, 300, 600, 1800, 3600]
    .find((s) => (width * s) / DATA.duration > 70) || 7200;
  axis.replaceChildren();
  for (let t = 0; t <= DATA.duration; t += step) {
    const tick = element("span", t + step > DATA.duration ? "last" : "", clock(t));
    tick.style.left = `${(100 * t) / DATA.duration}%`;
    axis.append(tick);
  }
}

function bindZoom() {
  const zoom = document.getElementById("zoom");
  zoom.addEventListener("input", () => {
    canvas.style.width = `${100 * zoom.value}%`;
    document.getElementById("zoom-value").textContent = `${zoom.value}x`;
    drawAxis();
  });
}

applyColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyColors);
drawNames();
drawBars();
canvas.append(axis);
drawAxis();
bindZoom();
