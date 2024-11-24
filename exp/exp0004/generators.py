import json
import numpy as np


class TrafficLightMaskGenerator:
    def __init__(
            self,
            image_size: list[int, int] | int | None = None,
            normalize: bool = False
        ):
        self.normalize = normalize
        self.original_size = (128, 64)  # (width, height)

        if image_size == None:
            self.image_size = self.original_size
        elif isinstance(image_size, list):
            self.image_size = (image_size[1], image_size[0])  # (width, height)
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)

        self.scale_x = self.image_size[0] / self.original_size[0]
        self.scale_y = self.image_size[1] / self.original_size[1]

        self.traffic_light_classes = [
            'green', 'yellow', 'red', 'straight',
            'left', 'right', 'empty', 'other'
        ]
        self.class_to_index = {
            cls: idx for idx, cls in enumerate(self.traffic_light_classes)
        }

    def scale_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        scaled_bbox = [
            x1 * self.scale_x,  # x1
            y1 * self.scale_y,  # y1
            x2 * self.scale_x,  # x2
            y2 * self.scale_y   # y2
        ]
        return scaled_bbox

    def generate_masks(self, traffic_lights_json):
        if isinstance(traffic_lights_json, str):
            with open(traffic_lights_json, "r") as f:
                traffic_lights = json.load(f)
        else:
            traffic_lights = traffic_lights_json

        # クラス数分のチャンネルを持つ空の配列を作成
        masks = np.zeros((self.image_size[1], self.image_size[0], 
                         len(self.traffic_light_classes)), dtype=np.float32)

        count_per_class = np.zeros(len(self.traffic_light_classes))
        # 各信号機に対してマスクを生成
        for light in traffic_lights:
            class_idx = self.class_to_index[light['class']]
            scaled_bbox = self.scale_bbox(light['bbox'])

            x1, y1, x2, y2 = map(self._convert_coordinate, scaled_bbox)

            x1 = max(0, min(x1, self.image_size[0]-1))
            x2 = max(0, min(x2, self.image_size[0]-1))
            y1 = max(0, min(y1, self.image_size[1]-1))
            y2 = max(0, min(y2, self.image_size[1]-1))

            masks[y1:y2+1, x1:x2+1, class_idx] = 1.0

            count_per_class[class_idx] += 1

        # 標準化
        if self.normalize:
            for i in range(len(self.traffic_light_classes)):
                if count_per_class[i] == 0:
                    continue
                masks[..., i] = (masks[..., i] - masks[..., i].min()) / (masks[..., i].max() - masks[..., i].min())

        return masks

    def _convert_coordinate(self, coord):
        return int(round(coord))