import numpy as np
import pandas as pd


def mark_labels(labels):
    """
    对原始的label进行转换，转换为[1, 0], [0,1]这样的表示概率的格式
    :param labels: 未经转换的标签列表，类型为list()
    :return: 经过转换之后的列表，类型为np.ndarray
    """
    result = []
    for x in labels:
        if x == 0:
            result.append([1.0, 0.0])
        else:
            result.append([0.0, 1.0])
    return np.array(result)


def read_data(file_name):
    df = pd.read_csv(file_name,
                      header=0,
                      names=["age", "job", "marital", "education", "default", "balance", "housing", "loan",
                             "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
                             "y"])
    df.sample(frac=1.0)
    data = df.values
    data_x = data[:, :-1]
    data_x = (data_x - np.min(data_x)) / (np.max(data_x) - np.min(data))
    data_y = mark_labels(data[:, -1])
    # 取前50%为训练数据
    length = int(0.5 * len(data))
    train_x = data_x[:length]
    train_y = data_y[:length]
    test_x = data_x[length:]
    test_y = data_y[length:]
    return train_x, train_y, test_x, test_y