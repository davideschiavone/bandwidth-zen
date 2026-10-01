"""Build ../deck.html and ../notebook.html from the files in this folder.

Usage:  python3 src/build.py      (Python 3.8+, standard library only)
The page chrome is shared with every notebook in ../../common/shell.py. The attention pages also
carry the matmul notebook's simulator (../../matmul/src/engine2.js): its matrix grid, stepping
controls and counters draw these pages too, and its SingleView shows one block of S = QKᵀ as the
matmul it is.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DEST, '..', 'common'))
from shell import deck_page, lab_page  # noqa: E402
os.chdir(HERE)

MM2 = open(os.path.join(DEST, '..', 'matmul', 'src', 'engine2.js')).read()
WIDGET = open(os.path.join(DEST, '..', 'common', 'widget.js')).read()
ENGINE = open('engine.js').read()
LABJS = open('lab.js').read()
CSS = open('attention.css').read()
FORMULA = ('<span class="tc">O</span> = softmax(<span class="ta">Q</span> <span class="tb">K</span>ᵀ / √d) '
           '<span class="tb">V</span>')


def build_lab():
    page = lab_page('Attention Blocked Notebook', 'Interactive notebook · pick a chapter, then turn the knobs — they are bwz attention\'s own flags',
                    'Attention, blocked', FORMULA, [MM2, WIDGET, ENGINE, LABJS], CSS)
    open(os.path.join(DEST, 'notebook.html'), 'w', encoding='utf-8').write(page)


def build_deck():
    page = deck_page('Attention Blocked Deck', 'Attention, blocked', open('deck_slides.html').read(), [MM2, WIDGET, ENGINE], CSS)
    open(os.path.join(DEST, 'deck.html'), 'w', encoding='utf-8').write(page)


build_lab(); build_deck()
print('built deck.html and notebook.html in', DEST)
