
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pdb

val_ref = json.load(open('../caption_eval/val_ref.json', 'r'))['annotations']

diffs = []
for i in range(30000):
    caps = [len(x['caption']) for x in val_ref[i * 5:(i + 1) * 5] if len(x['caption'].split(' ')) > 0]
    # diff = np.asarray(caps, dtype='float32').mean()
    diffs.extend(caps)
    # pdb.set_trace()

diffs = np.asarray(diffs, dtype='float32')
print(diffs.mean(), diffs.std())
