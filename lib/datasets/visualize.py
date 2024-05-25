import cv2
from webcolors import name_to_rgb
import numpy as np


def visualize(self, image, keypoints, filename):
    """Visualize an image with its keypoints, and store the result into a file

    Args:
        image (PIL.Image):
        keypoints (torch.Tensor): keypoints in shape K x 2
        filename (str): the name of file to store
    """
    assert self.colored_skeleton is not None

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
    if keypoints is not None:
        for (_, (line, color)) in self.colored_skeleton.items():
            color = name_to_rgb(color) if type(color) == str else color
            for i in range(len(line) - 1):
                start, end = keypoints[line[i]], keypoints[line[i + 1]]
                cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color=color,
                        thickness=3)
        for keypoint in keypoints:
            cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, name_to_rgb('black'), 1)
    cv2.imwrite(filename, image)