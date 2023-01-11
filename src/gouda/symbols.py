"""Useful unicode symbols for printing"""
import sys
import unicodedata
import warnings
from typing import Optional, Union

if not sys.stdout.encoding.lower().startswith('utf'):
    warnings.warn('Terminal does not support unicode. Some symbols may not display correctly.')

# ANSI Escape Codes - Just the ones that I use regularly. Check out plumbum for a way more complete way to do style text or crayons for a more light weight version.
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
ansi_stop = '\033[0m'
ansi_bold = '\033[1m'
ansi_italics = '\033[3m'
ansi_under = '\033[4m'
ansi_strike = '\033[9m'


def underline(string):
    """Shortcut to underline ANSI text"""
    return ansi_under + string + ansi_stop


# Lower-case Greek Letters - "Greek and Coptic" block
alpha = '\u03B1'
beta = '\u03B2'
gamma = '\u03B3'
delta = '\u03B4'
epsilon = '\u03B5'
zeta = '\u03B6'
eta = '\u03B7'
theta = '\u03B8'
iota = '\u03B9'
kappa = '\u03BA'
lambda_ = '\u03BB'
mu = '\u03BC'
nu = '\u03BD'
xi = '\u03BE'
omicron = '\u03BF'
pi = '\u03C0'
rho = '\u03C1'
final_sigma = '\u03C2'
sigma = '\u03C3'
tau = '\u03C4'
upsilon = '\u03C5'
phi = '\u03C6'
chi = '\u03C7'
psi = '\u03C8'
omega = '\u03C9'

# Upper-case Greek Letters - "Greek and Coptic" block
Alpha = '\u0391'
Beta = '\u0392'
Gamma = '\u0393'
Delta = '\u0394'
Epsilon = '\u0395'
Zeta = '\u0396'
Eta = '\u0397'
Theta = '\u0398'
Theta_symbol = '\u03F4'
Iota = '\u0399'
Kappa = '\u039A'
Lambda = '\u039B'
Mu = '\u039C'
Nu = '\u039D'
Xi = '\u039E'
Omicron = '\u039F'
Pi = '\u03A0'
Rho = '\u03A1'
Sigma = '\u03A3'
Tau = '\u03A4'
Upsilon = '\u03A5'
Phi = '\u03A6'
Chi = '\u03A7'
Psi = '\u03A8'
Omega = '\u03A9'

# Math Operators - "Mathematical Operators" block
# https://www.compart.com/en/unicode/block/U+2200
ForAll = '\u2200'
PartialD = '\u2202'
Exists = '\u2203'
NotExists = '\u2204'
EmptySet = '\u2205'
In = '\u2208'
NotIn = '\u2209'
In_small = '\u220A'
PlusMinus = '\u2213'
Circ = '\u2218'
Inf = '\u221E'
And = '\u2227'
Or = '\u2228'
Cap = '\u2229'
Cup = '\u222A'
Integral = '\u222B'
Integral2 = '\u222C'
Integral3 = '\u222D'
AlmostEq = '\u2248'
NotAlmostEq = '\u2249'
neq = '\u2260'
leq = '\u2264'
geq = '\u2265'
Subset = '\u2282'
NotSubset = '\u2284'
SubsetEq = '\u2286'
NotSubsetEq = '\u2288'
Superset = '\u2283'
NotSuperset = '\u2285'
SupersetEq = '\u2287'
NotSupersetEq = '\u2289'
CircPlus = '\u2295'
CircMinus = '\u2296'
CircX = '\u2297'
CircDot = '\u2299'

# Arrows - "Arrows" block
# https://www.compart.com/en/unicode/block/U+2190
left = '\u2190'
left2 = '\u21d0'
left_bar = '\u21e4'
left_upHarp = '\u21bc'
left_downHarp = '\u21bd'
up = '\u2191'
up2 = '\u21d2'
up_rightHarp = '\u21be'
up_leftHarp = '\u21bf'
right = '\u2192'
right2 = '\u21d2'
right_bar = '\u21e5'
right_upHarp = '\u21c0'
right_downHarp = '\u21c1'
down = '\u2193'
down2 = '\u21d3'
down_rightHarp = '\u21c2'
down_leftHarp = '\u21c3'
leftright = '\u2194'
leftright2 = '\u21c4'
leftright_Harp = '\u21cc'
updown = '\u2195'
updown2 = '\u21c5'
anticlockwise = '\u21ba'
clockwise = '\u21bb'

# Misc - "Miscellaneous Symbols" block
# https://www.compart.com/en/unicode/block/U+2600
sun = '\u2600'
sol = '\u2609'
star = '\u2605'
empty_star = '\u2606'
peace = '\u262e'
box = '\u2610'
box_check = '\u2611'
box_x = '\u2612'
heart = '\u2661'
heart2 = '\u2665'
diamond = '\u2662'
diamond2 = '\u2666'
club = '\u2667'
club2 = '\u2663'
spade = '\u2664'
spade2 = '\u2660'
d6_1 = '\u2680'
d6_2 = '\u2681'
d6_3 = '\u2682'
d6_4 = '\u2683'
d6_5 = '\u2684'
d6_6 = '\u2685'
gear = '\u2699'
gear2 = '\u26ed'
atom = '\u269b'
warning = '\u26a0'

# Playing Cards - "Playing Cards" block
# https://www.compart.com/en/unicode/block/U+1F0A0
num2word = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'}
letter2rank = {'A': 'Ace', 'J': 'Jack', 'Q': 'Queen', 'K': 'King'}


def get_card(suit: str, rank: Optional[Union[str, int, float]] = None) -> str:
    """Get the unicode for a given playing card

    Parameters
    ----------
    suit : str
        The suit of the card
    rank : Optional[Union[str, int, float]], optional
        The rank of the card, by default None

    Returns
    -------
    str
        Unicode for the given card, or \u2205 (empty set) if the card could not be found

    Note
    ----
    Suit can be Red or Black if rank is Joker, Back for the back of the card, or Joker for the Red Joker
    Rank can be 1-10 (number or text), Ace, A, King, K, Queen, Q, Jack, J or Joker
    """
    suit = suit.title()
    if suit in ['Heart', 'Diamond', 'Spade', 'Heart']:
        suit += 's'
    if suit not in ['Back', 'Red', 'Black', 'Spades', 'Hearts', 'Diamonds', 'Clubs']:
        warnings.warn('Could not find suit `{}`'.format(suit))
        return '\u2205'
    suit = suit.title()

    if isinstance(rank, (int, float)):
        rank = str(int(rank))
        rank = num2word.get(rank, rank)
    rank = str(rank).title()
    rank = letter2rank.get(rank, rank)

    if suit == 'Back':
        return unicodedata.lookup('Playing Card Back')
    elif suit == 'Joker' or (suit == 'Red' and rank == 'Joker'):
        return unicodedata.lookup('Playing Card Red Joker')
    elif suit == 'Black' and rank == 'Joker':
        return unicodedata.lookup('Playing Card Black Joker')

    card_name = 'Playing Card {} of {}'.format(rank, suit)
    try:
        card_code = unicodedata.lookup(card_name)
    except KeyError:
        warnings.warn('Could not find playing card `{}`'.format(card_name))
        card_code = '\u2205'

    return card_code


# Maybe? - emojis
# https://www.compart.com/en/unicode/block/U+1F600
