# -*-coding:utf-8-*-
import os
import sys
sys.path.append(os.getcwd())
import onnxruntime
import numpy as np
import pandas as pd

class ONNXModel():
    def __init__(self, onnx_path, gpu_cfg=False):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        if gpu_cfg:
            self.onnx_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
        self.output_len = 96
        self.batch_size = 32
        self.features= 7
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, data_n):
        all_predictions = []
            # 确保输入数据为 float32 类型
        data_numpy = data_n.astype('float32').values
        if data_numpy.ndim == 2:
            data_numpy = data_numpy.reshape((self.batch_size,self.output_len,self.features))
        input_feed = self.get_input_feed(self.input_name, data_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        all_predictions.append(output)
        return all_predictions


def main(onnx_path, data_path, N=96, batch_size=32,):
    # Load the time series data
    data = pd.read_csv(data_path)
    # 确保'date'列被排除，并且正确地定义了features变量
    data = data.drop(columns=['date'])  # 假设'date'列不用于预测
    data_96 = data.iloc[-(N*batch_size):,:]
    # Initialize the ONNX model
    model = ONNXModel(onnx_path, gpu_cfg=True)

    # Predict with batches and process predictions as before
    all_predictions = model.forward(data_96)  # 确保这一步骤与修改后的forward方法兼容

    return all_predictions



if __name__ == "__main__":
    # onnx_root_path = r'F:\Transformer\LTSF_Linear_main\LTSF_Linear_main\checkpoints\ETTm1_96_96_DLinear_custom_ftM_sl96_ll96_pl96_dm32_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
    onnx_path = r'checkpoint.onnx'
    # onnx_source_path = os.path.join(onnx_root_path, onnx_path)
    # model = ONNXModel(source_path)
    # output = model.forward(img_ndarray.astype('float32'))[0][0]

    # data_root_path = r"F:\Transformer\LTSF_Linear_main\LTSF_Linear_main\dataset\ett"
    data_path = r'ETTm1.csv'
    # data_source_path = os.path.join(data_root_path, data_path)
    # Call main function
    predictions = main(onnx_path, data_path, N=96)
    features = 7
    # Print or process predictions
    print(len(predictions))

    # 假设 predictions 是你的多维列表或数组
    # 首先将所有批次的预测结果连接起来
    # all_predictions = np.concatenate([batch[0] for batch in predictions])
    all_predictions = np.concatenate(predictions).reshape(-1, features)
    # 创建预测结果DataFrame

    # 加载ETTm1数据集以获取日期信息
    ettm1_df = pd.read_csv(data_source_path)
    # 注意：这里需要根据ETTm1数据集的实际列名进行调整
    # 获取DataFrame的列名（除去第一列）
    columns = list(ettm1_df.columns)[1:]
    predictions_df = pd.DataFrame(all_predictions, columns=columns)
    # 假设你想要将预测结果与ETTm1中的最后N行匹配
    # N是平铺后预测结果的长度
    last_n_dates = ettm1_df['date'].tail(len(all_predictions)).reset_index(drop=True)

    # 将日期信息添加到预测结果DataFrame中
    predictions_df.insert(0, 'date', last_n_dates)

    # 将整合后的DataFrame保存为CSV文件
    predictions_df.to_csv('predictions.csv', index=False)