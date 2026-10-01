"""Build ../deck.html and ../notebook.html from the files in this folder.

Usage:  python3 src/build.py      (Python 3.8+, standard library only)
Shares the page chrome (../../common/shell.py), the deck widgets (../../common/widget.js), the matmul
notebook's grid and controls (../../matmul/src/engine2.js) and the attention notebook's softmax
reference (../../attention/src/engine.js).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.dirname(HERE)
ROOT = os.path.join(DEST, '..')
sys.path.insert(0, os.path.join(ROOT, 'common'))
from shell import deck_page, lab_page  # noqa: E402
os.chdir(HERE)

SHARED = [open(os.path.join(ROOT, *p)).read() for p in (('matmul', 'src', 'engine2.js'), ('common', 'widget.js'), ('attention', 'src', 'engine.js'))]
ENGINE = open('engine.js').read()
CSS = open('encoder.css').read()
FORMULA = ('x → x + Attention(norm(x)) → x + FFN(norm(x))')


def build_lab():
    page = lab_page('Encoder Layer Notebook', 'Interactive notebook · pick a chapter, then turn the knobs',
                    'One encoder layer', FORMULA, SHARED + [ENGINE, open('lab.js').read()], CSS)
    open(os.path.join(DEST, 'notebook.html'), 'w', encoding='utf-8').write(page)


def build_deck():
    page = deck_page('Encoder Layer Deck', 'One encoder layer', open('deck_slides.html').read(), SHARED + [ENGINE], CSS)
    open(os.path.join(DEST, 'deck.html'), 'w', encoding='utf-8').write(page)


build_lab(); build_deck()
print('built deck.html and notebook.html in', DEST)
