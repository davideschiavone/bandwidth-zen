"""Build ../deck.html and ../notebook.html from the files in this folder.

Usage:  python3 src/build.py      (Python 3.8+, standard library only)
Shares the page chrome and deck widgets (../../common/), the matmul notebook's grid and controls, the
attention notebook's softmax reference and S/K/V/O styles, and the encoder notebook's arithmetic, chips
and cost view.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.dirname(HERE)
ROOT = os.path.join(DEST, '..')
sys.path.insert(0, os.path.join(ROOT, 'common'))
from shell import deck_page, lab_page  # noqa: E402
os.chdir(HERE)

read = lambda *p: open(os.path.join(ROOT, *p)).read()  # noqa: E731
SHARED = [read('matmul', 'src', 'engine2.js'), read('common', 'widget.js'), read('attention', 'src', 'engine.js'), read('encoder', 'src', 'engine.js')]
CSS = read('attention', 'src', 'attention.css') + read('encoder', 'src', 'encoder.css') + open('decoder.css').read()
ENGINE = open('engine.js').read()
FORMULA = 'prompt once → then one token at a time, attending to the KV cache'


def build_lab():
    page = lab_page('Decoder Layer Notebook', 'Interactive notebook · pick a chapter, then turn the knobs',
                    'One decoder layer, generating', FORMULA, SHARED + [ENGINE, open('lab.js').read()], CSS)
    open(os.path.join(DEST, 'notebook.html'), 'w', encoding='utf-8').write(page)


def build_deck():
    page = deck_page('Decoder Layer Deck', 'One decoder layer, generating', open('deck_slides.html').read(), SHARED + [ENGINE], CSS)
    open(os.path.join(DEST, 'deck.html'), 'w', encoding='utf-8').write(page)


build_lab(); build_deck()
print('built deck.html and notebook.html in', DEST)
