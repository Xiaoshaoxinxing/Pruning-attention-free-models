import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM

class EstimatorPruningSNR:
    """
    Calculate the SNR
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def model_weights_dict(self, model, fp16_conversion=False):
        res_dict = {}   # key: module/layer name, value: corresponding matrix parameter
        for n, m in model.named_modules():      # Traverse the entire model, extract parameters that need SNR calculation
            if isinstance(m, torch.nn.Linear):          # Pruning is essentially an operation on matrices, and Linear layers are such. Filter out non-Linear layers
                res_dict[n] = m.weight.to(device=self.device)

        if fp16_conversion:
            res_dict_half = {k: v.half() for k, v in res_dict.items()}
            return res_dict_half
        else:
            return res_dict

    def estimate_snr(self, E_w, mse):
        # Calculate SNR
        if mse.item() > 0.0:
            pruning_snr = 10 * np.log10(E_w.item() / mse.item())
        else:
            # When mse=0, it means the two sets of parameters are the same, and no pruning has occurred
            pruning_snr = np.Inf
        return pruning_snr

    
    def estimate_mse(self, tensor_base, tensor_pruning):
        # Calculate the MSE between two tensors
        mse = torch.mean((tensor_base - tensor_pruning) ** 2)   # Calculate the current module's MSE, i.e., E[(W-F(W)]^2
        num_elements = tensor_base.numel()  # Count the number of elements in the tensor
        
        E_w = torch.mean((tensor_base) ** 2)    # Calculate the mean expectation of the current module parameters in the base model, i.e., E(W^2)
        
        return mse, E_w, num_elements
        

    def estimate(self, base_model, pruning_model, every_layer=True):
        '''
        base_model: Reference model, here referring to the non-pruned model
        pruning_model: Model to be evaluated, referring to the pruned model
        '''
        base_model_dict = self.model_weights_dict(base_model)   # Get the Linear layer parameters of the base model
        pruning_model_dict = self.model_weights_dict(pruning_model)   # Get the Linear layer parameters of the pruning model
        
        mes_dict = {}
        for name, weight_base in base_model_dict.items():    # Iterate over the parameters of each Linear layer
            weight_pruning =  pruning_model_dict[name]       # Extract the corresponding weight parameters in the pruned model
            mse_item, E_w_item, num_elements = self.estimate_mse(weight_base, weight_pruning)  # Calculate the MSE and E(W^2) for a set of Linear parameters
            mes_dict[name] = {
                "mse": mse_item,
                "E_w": E_w_item,
                "num_elements": num_elements
            }
            if every_layer:
                SNR_i = self.estimate_snr(E_w_item, mse_item)
                print(f"{name}: {SNR_i}")
        
        # Calculate the total MSE and E(W^2)
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
        
        print(f"Total SNR: {SNR_db}")
        return SNR_db

class MetricAnalyzer:
    def __init__(self, base_model_path=None, device=None, every_layer=False) -> None:
        self.base_model_path = base_model_path
        self.pruning_model_path = None
        if device is None:          # If GPU is not specified, use default
            self.device = 'auto'
        self.device = device
        
        if self.base_model_path:
            self.base_model = self.load_model(self.base_model_path, self.device)
        else:
            self.base_model = None
            
        self.pruning_model = None
        self.every_layer = every_layer  # Whether to print the SNR results for each layer, default is False
        self.estimator_snr = EstimatorPruningSNR()
        
    def load_model(self, model_path, torch_dtype=torch.float16, device='auto'):
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            device_map=device
        )
        model.seqlen = model.config.max_position_embeddings 
        return model
    
    def analyzer(self, pruning_model_path, base_model_path=None):
        if self.base_model is None:     # If base_model is not provided during initialization, it must be specified here
            assert base_model_path is not None, "Please specify the path for base_model"
            self.base_model_path = base_model_path
            self.base_model = self.load_model(self.base_model_path, self.device)
            
        self.pruning_model_path = pruning_model_path
        self.pruning_model = self.load_model(self.pruning_model_path, self.device)   # Load the model to be evaluated
        
        base_model_name = os.path.basename(os.path.normpath(self.base_model_path))  # Get the model name (as directory name)
        pruning_model_name = os.path.basename(os.path.normpath(self.pruning_model_path))    # Get the model name (as directory name)
        
        print(f"============  {base_model_name} vs {pruning_model_name}  ============")
        self.estimator_snr.estimate(self.base_model, self.pruning_model, every_layer=self.every_layer)
        print(f"------------  end  ------------\n\n")
        

if __name__ == "__main__":
    base_model_path = "./various-sized models/RWKV_rwkv-4-1b5-pile"     # Path to the model before pruning
    pruning_model_path = "./wanda-main/out/1b5/unstructured_sparsegpt_0.5"  # Path to the pruned model
    every_layer = False     # Whether to output the SNR for each layer, default is False
    metricanalyzer = MetricAnalyzer(base_model_path, every_layer=every_layer)
    metricanalyzer.analyzer(pruning_model_path)

    # Below is the method for calculating multiple pruned models in batches
    # pruning_model_path_1 = "./wanda-main/out/1b5"   # Directory of pruned models
    # model_list = os.listdir(pruning_model_path_1)
    # model_list.sort()
    # print(model_list)
    # for model_name in model_list:
    #     pruning_model_path = os.path.join(pruning_model_path_1, model_name)
    #     metricanalyzer.analyzer(pruning_model_path)

        
        
            
