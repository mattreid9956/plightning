import os
import re
import logging
import pandas as pd
import parser

import pytorch_lightning as pl

_logger = logging.getLogger(__name__)


def find_model_path(cpcb: pl.callbacks.model_checkpoint.ModelCheckpoint, mode: str = "epoch") -> str:
    """Returns the path to the best or the last epoch run in the directory
    !EXPECT SPECIFIC filename format, we reverse engineer based on ModelCheckpoint.filename pattern
     ASSUMES YOU ALWAYS HAVE "epoch" name in your checkpoint filename!
    """
    mode = mode.lower()
    assert mode in ['epoch', 'best'], "mode must be either 'epoch' or 'best'!"
    
    files = pd.Series([f for f in os.listdir(cpcb.dirpath) if f.endswith(cpcb.FILE_EXTENSION)], name='filename')
    pattern = cpcb.filename
    if not pattern:
        pattern = "{epoch}" + cpcb.CHECKPOINT_JOIN_CHAR + "{step}"
        _logger.warning(f"Assuming default file pattern: {pattern}")

    groups = re.findall(r"(\{.*?)[:\}]", pattern)
    for group in groups:
        name = group[1:]
        pattern = pattern.replace(group, name + "={" + name)
   
    res = []
    for f in files:
        res.append(search(pattern, f).named)
    res = pd.DataFrame(res, index=cpcb.dirpath + "/" + files)
    _logger.info(f"Parsed {len(res)} checkpoint files:\n{res}")
    
    if mode == "best":
        output = res.sort_values(cpcb.monitor, ascending=True if cpcb.mode=="min" else False).iloc[[0]].index[0]
    else:
        output = res.sort_values("epoch", ascending=False).iloc[[0]].index[0]
    _logger.info(f"Found criteria={mode} gave file: {output}")
    return output
    
