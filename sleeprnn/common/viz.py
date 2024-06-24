from IPython.core.display import display, HTML

from sleeprnn.common import constants

PALETTE = {
    constants.RED: "#c62828",
    constants.GREY: "#455a64",
    constants.BLUE: "#0277bd",
    constants.GREEN: "#43a047",
    constants.DARK: "#1b2631",
    constants.CYAN: "#00838F",
    constants.PURPLE: "#8E24AA",
}

GREY_COLORS = {
    0: "#fafafa",
    1: "#f5f5f5",
    2: "#eeeeee",
    3: "#e0e0e0",
    4: "#bdbdbd",
    5: "#9e9e9e",
    6: "#757575",
    7: "#616161",
    8: "#424242",
    9: "#212121",
}

COMPARISON_COLORS = {
    "model": PALETTE["cyan"],
    "expert": PALETTE["blue"],
    "baseline": GREY_COLORS[7],
}

BASELINES_LABEL_MARKER = {
    "2019_chambon": ("DOSED", "s"),
    "2019_lacourse": ("A7", "d"),
    "2017_lajnef": ("Spinky", "p"),
}

DPI = 200
FONTSIZE_TITLE = 9
FONTSIZE_GENERAL = 8
AXIS_COLOR = GREY_COLORS[8]
LINEWIDTH = 1.1
MARKERSIZE = 5
LEGEND_LABEL_SPACING = 1.1


def notebook_full_width():
    display(HTML("<style>.container { width:100% !important; }</style>"))
