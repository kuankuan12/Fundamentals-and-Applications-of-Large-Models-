import torch
from torch import optim, nn


# 定义HiRA网络结构
class HiRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # HiRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x, original_weight):
        """
        HiRA前向传播：计算 (W₀ ⊙ (B·A))x
        其中 W₀ 是原始权重，B·A 是低秩矩阵，⊙ 是Hadamard积（逐元素相乘）
        """
        # 计算 BA = B.weight @ A.weight，形状为 [out_features, in_features]
        # B.weight: [out_features, rank], A.weight: [rank, in_features]
        BA = torch.matmul(self.B.weight, self.A.weight)  # [out_features, rank] @ [rank, in_features] -> [out_features, in_features]
        
        # 计算 W₀ ⊙ (BA)，Hadamard积（逐元素相乘）
        # original_weight: [out_features, in_features]
        W0_BA = original_weight * BA  # [out_features, in_features] ⊙ [out_features, in_features]
        
        # 计算 (W₀ ⊙ (BA))x = x @ (W₀ ⊙ (BA))^T
        # x: [..., in_features], W0_BA: [out_features, in_features]
        return torch.matmul(x, W0_BA.t())  # [..., in_features] @ [in_features, out_features] -> [..., out_features]


def apply_hira(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            hira = HiRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(module.weight.device)
            setattr(module, "hira", hira)
            original_forward = module.forward

            # 显式绑定，在forward中直接使用module.weight（原始权重，被冻结）
            def forward_with_hira(x, layer1=original_forward, layer2=hira, orig_weight=module.weight):
                # HiRA: y = W₀x + (W₀ ⊙ (BA))x
                return layer1(x) + layer2(x, orig_weight)

            module.forward = forward_with_hira


def load_hira(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'hira'):
            hira_state = {k.replace(f'{name}.hira.', ''): v for k, v in state_dict.items() if f'{name}.hira.' in k}
            module.hira.load_state_dict(hira_state)


def save_hira(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'hira'):
            hira_state = {f'{name}.hira.{k}': v for k, v in module.hira.state_dict().items()}
            state_dict.update(hira_state)
    torch.save(state_dict, path)

