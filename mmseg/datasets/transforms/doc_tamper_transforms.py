import os
import pickle
import random
import tempfile
from pathlib import Path

import jpegio
import mmcv
import numpy as np
import six
from mmcv.transforms import BaseTransform, TRANSFORMS
from PIL import Image
import copy


@TRANSFORMS.register_module()
class RandomJpegCompressAndLoadInfo(BaseTransform):
    def __init__(self, jpeg_compress_time=(1, 2, 3), course=False, quality_lower=75, compress_pk=None, load_info=True,
                 return_rgb=False):
        super().__init__()
        self.jpeg_compress_time = jpeg_compress_time
        self.course = course
        self.quality_lower = quality_lower
        self.compress_pk = compress_pk
        if self.compress_pk is not None:
            assert os.path.exists(self.compress_pk), f"{self.compress_pk} not exists"
            self.compress_pk = pickle.load(open(self.compress_pk, 'rb'))
            # List of [q2, q1, q]

        self.load_info = load_info
        if course:
            raise NotImplementedError
        self.return_rgb = return_rgb

    def transform(self, results: dict) -> dict:
        img: np.ndarray = results['img']

        if self.course:
            raise NotImplementedError
        else:
            quality_lower = self.quality_lower

        if self.compress_pk is None:
            jpeg_compress_time = random.choice(self.jpeg_compress_time)
            compress_quality = np.random.randint(quality_lower, 101, jpeg_compress_time)
        else:
            image_path = results['img_path']
            index = int(Path(image_path).stem)
            compress_quality = self.compress_pk[index]

        im = Image.fromarray(img)

        if self.return_rgb:
            im_ = im.copy()
        else:
            im_ = im.convert("L")

        compress_quality = [compress_quality[0]]  # keep the same as the author,
        for q in compress_quality:
            buffer = six.BytesIO()
            im_.save(buffer, format="JPEG", quality=int(q))
            im_ = Image.open(buffer)

        if self.load_info:
            jpg = jpegio.read(buffer.getvalue())

            dct = copy.deepcopy(jpg.coef_arrays[0])
            use_qtb = copy.deepcopy(jpg.quant_tables[0]).astype(np.uint8)

            results['dct'] = np.clip(np.abs(dct), 0, 20)
            results['qtb'] = np.expand_dims(np.clip(use_qtb, 0, 63).astype(np.int32), 0)

        im = im_.convert('RGB')
        results['img'] = np.array(im)

        return results
