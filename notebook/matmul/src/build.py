"""Build ../deck.html and ../notebook.html from the files in this folder.

Usage:  python3 src/build.py      (Python 3.8+, standard library only)
The page chrome — reset, fonts, deck and notebook CSS/JS — is shared with every notebook in
../../common/shell.py; this file supplies only what is matmul's own.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DEST, '..', 'common'))
from shell import deck_page, lab_page  # noqa: E402
os.chdir(HERE)

JS = open('engine.js').read()
JS2 = open('engine2.js').read() + '\n' + open('engine3.js').read()
LABJS = open('lab.js').read()


def build_lab():
    page = lab_page(
        'Matmul Rearranged Notebook',
        'Interactive notebook · pick a chapter, then turn the knobs',
        'Matmul, rearranged',
        '<span class="ta">A</span>[M, K] @ <span class="tb">B</span>[K, N] → <span class="tc">C</span>[M, N]',
        [JS2, LABJS])
    open(os.path.join(DEST, 'notebook.html'), 'w', encoding='utf-8').write(page)


def build_deck():
    page = deck_page('Matmul Rearranged Deck', 'Matmul, rearranged', open('deck_slides.html').read(), [JS, JS2])
    open(os.path.join(DEST, 'deck.html'), 'w', encoding='utf-8').write(page)


build_lab(); build_deck()
print('built deck.html and notebook.html in', DEST)
