#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: sebastientourbier
# @Date:   2019-09-06 13:39:38
import sys
import os.path as op


def main():
    sys.path.insert(0, op.abspath('.'))
    from pymialsrtk.info import __version__
    print(__version__)


if __name__ == '__main__':
    main()
