"""Launch the authors' own Von server (`von serve`, pip von-sdk==1.2.3 == github wfzyx/von src/von @ fb6e7a9),
unchanged except for one guard: a packed request longer than the model's 8,192-token window (ModernBERT-large
max_position_embeddings) is rejected with ValueError, which the authors' server answers with HTTP 422, instead of
being run past the window. Weights: the authors' default local checkpoint dir `checkpoints/von-1.2` (relative to
cwd servers/src/von), a symlink to the HF snapshot wfzyx/von@5df8185 (option_marker.pt + marker_calibration.json).
Not a serve.py adapter module.

    cd servers/src/von && ../../venvs/d-von/bin/python ../../adapters/von_guarded.py --port 8145
"""
import sys

import von.models.option_marker as om
from von.cli import main

WINDOW = 8192
_forward = om.OptionMarkerModel.forward


def guarded_forward(self, input_ids, attention_mask, mask_positions, independent_options=False):
    n = int(input_ids.shape[1])
    if n > WINDOW:
        raise ValueError(f"maximum context length is {WINDOW} tokens, the packed request has {n}")
    return _forward(self, input_ids, attention_mask, mask_positions, independent_options=independent_options)


om.OptionMarkerModel.forward = guarded_forward

if __name__ == "__main__":
    main(["serve", "--host", "127.0.0.1", "--model", "von-1.2", "--device", "cuda", *sys.argv[1:]])
