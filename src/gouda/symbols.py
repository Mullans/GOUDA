"""Useful unicode symbols for printing."""

# TODO - make more consistent with LaTeX
from __future__ import annotations

import sys
import unicodedata
import warnings

if not sys.stdout.encoding.lower().startswith("utf"):
    warnings.warn("Terminal does not support unicode. Some symbols may not display correctly.")

# ANSI Escape Codes - Just the ones that I use regularly. Check out plumbum for a way more complete way to do style text or crayons for a more light weight version.
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
ansi_stop = "\033[0m"
ansi_bold = "\033[1m"
ansi_italics = "\033[3m"
ansi_under = "\033[4m"
ansi_strike = "\033[9m"


def underline(string: str) -> str:
    """Shortcut to underline ANSI text."""
    return ansi_under + string + ansi_stop


# Lower-case Greek Letters - "Greek and Coptic" block
alpha = "\u03b1"
beta = "\u03b2"
gamma = "\u03b3"
delta = "\u03b4"
epsilon = "\u03b5"
zeta = "\u03b6"
eta = "\u03b7"
theta = "\u03b8"
iota = "\u03b9"
kappa = "\u03ba"
lambda_ = "\u03bb"
mu = "\u03bc"
nu = "\u03bd"
xi = "\u03be"
omicron = "\u03bf"
pi = "\u03c0"
rho = "\u03c1"
final_sigma = "\u03c2"
sigma = "\u03c3"
tau = "\u03c4"
upsilon = "\u03c5"
phi = "\u03c6"
chi = "\u03c7"
psi = "\u03c8"
omega = "\u03c9"

# Upper-case Greek Letters - "Greek and Coptic" block
Alpha = "\u0391"
Beta = "\u0392"
Gamma = "\u0393"
Delta = "\u0394"
Epsilon = "\u0395"
Zeta = "\u0396"
Eta = "\u0397"
Theta = "\u0398"
Theta_symbol = "\u03f4"
Iota = "\u0399"
Kappa = "\u039a"
Lambda = "\u039b"
Mu = "\u039c"
Nu = "\u039d"
Xi = "\u039e"
Omicron = "\u039f"
Pi = "\u03a0"
Rho = "\u03a1"
Sigma = "\u03a3"
Tau = "\u03a4"
Upsilon = "\u03a5"
Phi = "\u03a6"
Chi = "\u03a7"
Psi = "\u03a8"
Omega = "\u03a9"

# Math Symbols - "Latin-1 Supplement" block
nbsp = "\u00a0"
section = "\u00a7"
Not = "\u00ac"
plusMinus = "\u00b1"
dot = "\u00b7"
division = "\u00f7"

# Math Operators - "Mathematical Operators" block
# https://www.compart.com/en/unicode/block/U+2200
forAll = "\u2200"
partialD = "\u2202"
exists = "\u2203"
notExists = "\u2204"
emptySet = "\u2205"
In = "\u2208"
notIn = "\u2209"
in_small = "\u220a"
minusPlus = "\u2213"
circ = "\u2218"
inf = "\u221e"
And = "\u2227"
Or = "\u2228"
cap = "\u2229"
cup = "\u222a"
integral = "\u222b"
integral2 = "\u222c"
integral3 = "\u222d"
almostEq = "\u2248"
notAlmostEq = "\u2249"
neq = "\u2260"
leq = "\u2264"
nleq = "\u2270"
geq = "\u2265"
ngeq = "\u2271"
subset = "\u2282"
notSubset = "\u2284"
subsetEq = "\u2286"
notSubsetEq = "\u2288"
superset = "\u2283"
notSuperset = "\u2285"
supersetEq = "\u2287"
notSupersetEq = "\u2289"
circPlus = "\u2295"
circMinus = "\u2296"
circX = "\u2297"
circDot = "\u2299"

# Arrows - "Arrows" block
# https://www.compart.com/en/unicode/block/U+2190
left = "\u2190"
left2 = "\u21d0"
left_bar = "\u21e4"
left_upHarp = "\u21bc"
left_downHarp = "\u21bd"
up = "\u2191"
up2 = "\u21d2"
up_rightHarp = "\u21be"
up_leftHarp = "\u21bf"
right = "\u2192"
right2 = "\u21d2"
right_bar = "\u21e5"
right_upHarp = "\u21c0"
right_downHarp = "\u21c1"
down = "\u2193"
down2 = "\u21d3"
down_rightHarp = "\u21c2"
down_leftHarp = "\u21c3"
leftright = "\u2194"
leftright2 = "\u21c4"
leftright_Harp = "\u21cc"
updown = "\u2195"
updown2 = "\u21c5"
anticlockwise = "\u21ba"
clockwise = "\u21bb"

# Misc - "Miscellaneous Symbols" block
# https://www.compart.com/en/unicode/block/U+2600
sun = "\u2600"
sol = "\u2609"
star = "\u2605"
empty_star = "\u2606"
peace = "\u262e"
box = "\u2610"
box_check = "\u2611"
box_x = "\u2612"
heart = "\u2661"
heart2 = "\u2665"
diamond = "\u2662"
diamond2 = "\u2666"
club = "\u2667"
club2 = "\u2663"
spade = "\u2664"
spade2 = "\u2660"
d6_1 = "\u2680"
d6_2 = "\u2681"
d6_3 = "\u2682"
d6_4 = "\u2683"
d6_5 = "\u2684"
d6_6 = "\u2685"
gear = "\u2699"
gear2 = "\u26ed"
atom = "\u269b"
warning = "\u26a0"

# Playing Cards - "Playing Cards" block
# https://www.compart.com/en/unicode/block/U+1F0A0
num2word = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
letter2rank = {"A": "Ace", "J": "Jack", "Q": "Queen", "K": "King"}


def get_card(suit: str, rank: str | int | float | None = None) -> str:
    """Get the unicode for a given playing card.

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
    if suit in ["Heart", "Diamond", "Spade", "Heart"]:
        suit += "s"
    if suit not in ["Back", "Red", "Black", "Spades", "Hearts", "Diamonds", "Clubs"]:
        warnings.warn(f"Could not find suit `{suit}`")
        return "\u2205"
    suit = suit.title()

    if isinstance(rank, int | float):
        rank = str(int(rank))
        rank = num2word.get(rank, rank)
    rank = str(rank).title()
    rank = letter2rank.get(rank, rank)

    if suit == "Back":
        return unicodedata.lookup("Playing Card Back")
    elif suit == "Joker" or (suit == "Red" and rank == "Joker"):
        return unicodedata.lookup("Playing Card Red Joker")
    elif suit == "Black" and rank == "Joker":
        return unicodedata.lookup("Playing Card Black Joker")

    card_name = f"Playing Card {rank} of {suit}"
    try:
        card_code = unicodedata.lookup(card_name)
    except KeyError:
        warnings.warn(f"Could not find playing card `{card_name}`")
        card_code = "\u2205"

    return card_code


# Maybe? - emojis
# https://www.compart.com/en/unicode/block/U+1F600
