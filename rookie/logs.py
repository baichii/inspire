"""
@创建日期 ：2022/5/5
@修改日期 ：2022/5/5
@作者 ：jzj
@功能 ：自定义日志模块
"""


import os
import time
import logging
from typing import List


class MyLogger:
    """
    自定义日志
    """
    def __init__(
            self,
            verbose=0,
            log_path=None,
            level=logging.INFO,
    ):
        """
        Args:
            verbose: console 打印
            log_path: 日志保存路径，如果为None，则不保存
            level: 日志等级，不知道改动了会有什么后果，所以尽量不要动
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        if verbose:
            sh = logging.StreamHandler()
            sh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            sh.setFormatter(sh_formatter)
            self.logger.addHandler(sh)

        if log_path:
            log_name = time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + ".log"
            fh = logging.FileHandler(os.path.join(log_path, log_name), "w")
            self.logger.addHandler(fh)

    def update(self, info):
        """info以字典传递"""
        message = " ".join([k + ": " + str(v) for k, v in info.items() if v != ""])
        self.logger.info(message)
