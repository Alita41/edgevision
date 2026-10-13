from html.parser import HTMLParser
from pathlib import Path

class Anchors(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=set();self.hrefs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='a' and 'href' in a:self.hrefs.append(a['href'])


def test_dashboard_section_links_are_real():
    root=Path(__file__).resolve().parents[1]/'edgevision'/'web'
    h=Anchors();h.feed((root/'index.html').read_text(encoding='utf-8'))
    assert {'overview','live-feed','events','controls'} <= h.ids
    for link in h.hrefs:
        if link.startswith('#'):assert link[1:] in h.ids
    for f in ('app.js','style.css','favicon.svg'):
        assert (root/f).exists()
