"""Known FlyWire root ids used for validation and, later, for I/O with the game.

Ids come from the Shiu et al. 2024 example notebook (FlyWire v630 ids). Most survive
unchanged in v783; use :meth:`Connectome.index_of_existing` to skip the ones that don't.
"""

# 21 sugar-sensing gustatory receptor neurons, right hemisphere.
SUGAR_GRN_RIGHT = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]

# Proboscis extension motor neuron (rostrum protractor).
MN9 = 720575940660219265
