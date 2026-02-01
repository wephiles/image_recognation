#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.data_preparation import DataPreparer

from sklearn.model_selection import train_test_split


def main() -> int:
    preparer = DataPreparer()
    X_train, X_val, X_test, y_train, y_val, y_test = preparer.prepare_data()
    preparer.visualize_distribution(y_train + y_val + y_test)
    return 0


if __name__ == '__main__':
    main()
