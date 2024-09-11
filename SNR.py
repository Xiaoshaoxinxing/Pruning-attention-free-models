import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM

class EstimatorPruningSNR:
    """
    计算的SNR
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def model_weights_dict(self, model, fp16_conversion=False):
        res_dict = {}   # key：模块/层 名称，vlaue: 对应的矩阵参数
        for n, m in model.named_modules():      # 遍历整个模型，取出需要计算SNR的参数
            if isinstance(m, torch.nn.Linear):          # 剪枝实际上是对矩阵进行操作的，Linear层就是。过滤掉非Linear的层
                res_dict[n] = m.weight.to(device=self.device)

        if fp16_conversion:
            res_dict_half = {k: v.half() for k, v in res_dict.items()}
            return res_dict_half
        else:
            return res_dict

    def estimate_snr(self, E_w, mse):
        # 计算SNR
        if mse.item() > 0.0:
            pruning_snr = 10 * np.log10(E_w.item() / mse.item())
        else:
            # 当mse=0，说明两个组参数是一样的，并没有剪枝
            pruning_snr = np.Inf
        return pruning_snr

    
    def estimate_mse(self, tensor_base, tensor_pruning):
        # 计算两个tensor的MSE
        mse = torch.mean((tensor_base - tensor_pruning) ** 2)   # 计算当前模块MSE,即E[(W-F(W)}^2]
        num_elements = tensor_base.numel()  # 统计tensor的元素个数
        
        E_w = torch.mean((tensor_base) ** 2)    # 计算base模型当前模块参数中的平均期望，即，E(W^2)
        
        return mse, E_w, num_elements
        

    def estimate(self, base_model, pruning_model, every_layer=True):
        '''
        base_model: 基准模型，这里指未剪枝模型
        pruning_model：待评价模型，指剪枝后的模型
        '''
        base_model_dict = self.model_weights_dict(base_model)   # 获取base模型的Linear层的参数
        pruning_model_dict = self.model_weights_dict(pruning_model)   # 获取pruning模型的Linear层的参数
        
        mes_dict = {}
        for name, weight_base in base_model_dict.items():    # 遍历每个Linear层的参数
            weight_pruning =  pruning_model_dict[name]       # 取出剪枝模型的对应位置的权重参数
            mse_item, E_w_item, num_elements = self.estimate_mse(weight_base, weight_pruning)  # 计算出一组Linear参数的MSE、E(W^2)
            mes_dict[name] = {
                "mse": mse_item,
                "E_w": E_w_item,
                "num_elements": num_elements
            }
            if every_layer:
                SNR_i = self.estimate_snr(E_w_item, mse_item)
                print(f"{name}: {SNR_i}")
        
        # 计算总的MSE,和E(W^2)
        mse_all = 0
        E_w_all = 0
        num_elements_all = 0
        for name, mes_info in mes_dict.items():
            mse_all += mes_info["mse"] * mes_info["num_elements"]
            E_w_all += mes_info["E_w"] * mes_info["num_elements"]
            num_elements_all += mes_info["num_elements"]
        mse = mse_all / num_elements_all
        E_w = E_w_all / num_elements_all
        
        SNR_db = self.estimate_snr(E_w, mse)
        
        print(f"总的SNR： {SNR_db}")
        return SNR_db

class MetricAnalyzer:
    def __init__(self, base_model_path=None, device=None, every_layer=False) -> None:
        self.base_model_path = base_model_path
        self.pruning_model_path = None
        if device is None:          # 不指定GPU，则使用默认
            self.device = 'auto'
        self.device = device
        
        if self.base_model_path:
            self.base_model = self.load_model(self.base_model_path, self.device)
        else:
            self.base_model = None
            
        self.pruning_model = None
        self.every_layer = every_layer  # 是否打印每一层的SNR结果，默认Fasle
        self.estimator_snr = EstimatorPruningSNR()
        
    def load_model(self, model_path, torch_dtype=torch.float16, device='auto'):
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            device_map=device
        )
        model.seqlen = model.config.max_position_embeddings 
        return model
    
    def analyzer(self, pruning_model_path, base_model_path=None):
        if self.base_model is None:     # 初始化若没有给base_model,则这个位置必须要指定
            assert base_model_path is not None, "请指定base_model的路径"
            self.base_model_path = base_model_path
            self.base_model = self.load_model(self.base_model_path, self.device)
            
        self.pruning_model_path = pruning_model_path
        self.pruning_model = self.load_model(self.pruning_model_path, self.device)   # 加载待评测的模型
        
        base_model_name = os.path.basename(os.path.normpath(self.base_model_path))  # 获取模型名称（以目录名）
        pruning_model_name = os.path.basename(os.path.normpath(self.pruning_model_path))    # 获取模型名称（以目录名）
        
        print(f"============  {base_model_name} vs {pruning_model_name}  ============")
        self.estimator_snr.estimate(self.base_model, self.pruning_model, every_layer=self.every_layer)
        print(f"------------  end  ------------\n\n")
        

if __name__ == "__main__":
    base_model_path = "./various-sized models/RWKV_rwkv-4-1b5-pile"     # 剪枝前的模型路径
    pruning_model_path = "./wanda-main/out/1b5/unstructured_sparsegpt_0.5"  # 剪枝后的模型路径
    every_layer = False     # 是否输出每一层的SNR，默认Fasle
    metricanalyzer = MetricAnalyzer(base_model_path, every_layer=every_layer)
    metricanalyzer.analyzer(pruning_model_path)

    # 下面是批量计算多个剪枝模型的方法
    # pruning_model_path_1 = "./wanda-main/out/1b5"   # 剪枝模型的目录
    # model_list = os.listdir(pruning_model_path_1)
    # model_list.sort()
    # print(model_list)
    # for model_name in model_list:
    #     pruning_model_path = os.path.join(pruning_model_path_1, model_name)
    #     metricanalyzer.analyzer(pruning_model_path)
        
        
            
