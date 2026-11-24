"""Timeline: self-contained interactive HTML of named Turns for one Video.

`page.html`, `timeline.css` and `timeline.js` are inlined at render time;
the page reads its data from a JSON script block.
"""

import html
import json
from collections import Counter
from importlib.resources import files
from string import Template

from whodat.corpus import Person
from whodat.labels import REVIEW, UNKNOWN
from whodat.match import Prediction

DARK = {
    "#2a78d6": "#3987e5",
    "#eb6834": "#d95926",
    "#1baf7a": "#199e70",
    "#eda100": "#c98500",
    "#e87ba4": "#d55181",
    "#008300": "#0ca30c",
    "#6250d6": "#9085e9",
    "#e34948": "#e66767",
}
"""Validated categorical palette: light step -> dark step, fixed slot order."""


def page_data(
    video: str, duration: float, persons: list[Person], predictions: list[Prediction]
) -> dict:
    rows = [
        {
            "id": p.id,
            "name": p.name,
            "light": p.color,
            "dark": DARK.get(p.color, p.color),
        }
        for p in persons
    ]
    rows += [
        {"id": UNKNOWN, "name": "unknown", "light": None, "dark": None},
        {"id": REVIEW, "name": "review", "light": None, "dark": None},
    ]
    seconds, counts = Counter(), Counter()
    for p in predictions:
        seconds[p.decision.identity] += p.turn.duration
        counts[p.decision.identity] += 1
    return {
        "video": video,
        "duration": duration,
        "rows": rows,
        "names": {p.id: p.name for p in persons},
        "turns": [
            [
                p.turn.start,
                p.turn.end,
                p.decision.identity,
                p.decision.score,
                p.decision.hint,
                p.turn.cluster,
            ]
            for p in predictions
        ],
        "totals": [
            [r["id"], round(seconds[r["id"]], 1), counts[r["id"]]]
            for r in rows
            if counts[r["id"]]
        ],
    }


def render(
    video: str,
    title: str,
    duration: float,
    persons: list[Person],
    predictions: list[Prediction],
) -> str:
    assets = files(__package__)
    data = page_data(video, duration, persons, predictions)
    return Template(assets.joinpath("page.html").read_text()).substitute(
        title=html.escape(title),
        style=assets.joinpath("timeline.css").read_text(),
        script=assets.joinpath("timeline.js").read_text(),
        data=json.dumps(data).replace("</", "<\\/"),
    )
