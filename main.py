#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def main() -> int:
    x = os.path.join('data', 'raw', 'hello')
    print(x)
    return 0


if __name__ == '__main__':
    main()
