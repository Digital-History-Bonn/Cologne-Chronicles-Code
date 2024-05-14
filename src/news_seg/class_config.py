"""Contains config for class names, labels and colors"""

cmap_10 = [
    (1.0, 0.0, 0.16),
    (1.0, 0.43843843843843844, 0.0),
    (0, 0.222, 0.222),
    (0.36036036036036045, 0.5, 0.5),
    (0.0, 1.0, 0.2389486260454002),
    (0.8363201911589008, 1.0, 0.0),
    (0.0, 0.5615942028985507, 1.0),
    (0.0422705314009658, 0.0, 1.0),
    (0.6461352657004831, 0.0, 1.0),
    (1.0, 0.0, 0.75),
]

cmap_12 = [
    (1.0, 0.0, 0.16),
    (1.0, 0.43843843843843844, 0.0),
    (0, 0.222, 0.222),
    (0.36036036036036045, 0.5, 0.5),
    (0.0, 1.0, 0.2389486260454002),
    (0.8363201911589008, 1.0, 0.0),
    (0.0, 0.5615942028985507, 1.0),
    (0.0422705314009658, 0.0, 1.0),
    (0.6461352657004831, 0.0, 1.0),
    (1.0, 0.0, 0.75),
    (1.0, 0.73, 0.98),
    (0.5, 0.0, 0.0),
]

TOLERANCE = [
    10.0,  # "UnknownRegion"
    5.0,  # "caption"
    5.0,  # "table"
    5.0,  # "article"
    1.0,  # "heading"
    10.0,  # "header"
    2.0,  # "separator_vertical"
    2.0,  # "separator_short"
    5.0,  # "separator_horizontal"
    5.0,  # "image",
    5.0  # "inverted_text"
]

# The order dictates the priority in the drawing process. Eg. "image": 10 assigns label 10 to image regions, but the
# drawn region will be overwritten by tables, which are further down the dictionary.
LABEL_ASSIGNMENTS = {
    "TextLine": 0,
    "advertisement": 0,
    "UnknownRegion": 1,
    "image": 10,
    "Image": 10,
    "inverted_text": 11,
    "caption": 2,
    "table": 3,
    "article": 4,
    "article_": 4,
    "heading": 5,
    "header": 6,
    "separator_fancy": 7,
    "separator_vertical": 7,
    "separator_short": 8,
    "separator_horizontal": 9,
}

LABEL_NAMES = [
    "UnknownRegion",
    "caption",
    "table",
    "article",
    "heading",
    "header",
    "separator_vertical",
    "separator_short",
    "separator_horizontal",
    "image",
    "inverted_text",
]

REGION_TYPES = {
    "UnknownRegion": "UnknownRegion",
    "caption": "TextRegion",
    "table": "TableRegion",
    "article": "TextRegion",
    "heading": "TextRegion",
    "header": "TextRegion",
    "separator_vertical": "SeparatorRegion",
    "separator_short": "SeparatorRegion",
    "separator_horizontal": "SeparatorRegion",
    "image": "GraphicRegion",
    "inverted_text": "TextRegion",
}

VALID_TAGS = [
    'TextLine',
    'advertisement',
    'article_',
    'article',
    'caption',
    'heading',
    'header',
    'separator_fancy',
    'Image',
    'Graphic',
    'table',
    'inverted_text',
    'separator_horizontal',
    'separator_vertical',
    'separator_short',
    'UnknownRegion'
]

# REDUCE_CLASSES = {
#     0: [1],
#     7: [8]
# }
#
# REDUCE_CLASSES = {
#     0: [1],
#     4: [2, 5, 6, 11],
#     7: [8, 9]
# }

REDUCE_CLASSES = {
    7: [8, 9]
}
