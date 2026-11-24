import json
import re

from whodat.corpus import Person
from whodat.match import Decision, Prediction, Turn
from whodat.timeline import render


def test_rendered_page_embeds_data_that_survives_closing_tags():
    title = "</script><b>x"
    person = Person("chris", title, "#2a78d6")
    turn = Prediction(Turn(0.0, 2.0, "A"), Decision("chris", 0.9, "chris"))

    page = render("vid", title, 10.0, [person], [turn])

    block = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert json.loads(block.group(1))["names"] == {"chris": title}
