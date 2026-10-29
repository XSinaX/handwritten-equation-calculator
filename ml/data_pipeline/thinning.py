"""
Zhang-Suen thinning: reduces strokes in a binary image to 1-pixel-wide lines.

Used to give symbols from different image datasets (thin pen vs thick marker)
the same starting thickness before a random pen width is applied.
"""

import numpy as np


def thin(mask):
    """
    Parameters
    ----------
    mask : np.ndarray (H, W)
        Non-zero where there is ink.

    Returns
    -------
    np.ndarray (H, W) uint8
        1 on the 1-pixel-wide centre line of each stroke, 0 elsewhere.
    """

    image = np.pad((mask > 0).astype(np.uint8), 1)

    while True:
        changed = False

        for step in (0, 1):
            # Neighbours, clockwise from the pixel above (P2) to above-left (P9)
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]

            centre = image[1:-1, 1:-1]

            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(n.astype(np.int16) for n in neighbours)

            # Number of 0 -> 1 changes going round the neighbours
            transitions = sum(
                ((neighbours[i] == 0) & (neighbours[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8)
            )

            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remove = (centre == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & side

            if remove.any():
                centre[remove] = 0
                changed = True

        if not changed:
            break

    return image[1:-1, 1:-1]
