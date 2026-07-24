%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_96x96_square_forced(model, config):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. 입력 텐서 생성 (기울기 추적 활성화)
    if hasattr(config, 'channels'):
        channel = config.channels
    elif hasattr(config, 'enc_in'):
        channel = config.enc_in 
    input_tensor = torch.randn(1, config.seq_len, channel).to(device).requires_grad_(True)
    
    # 2. 모델 실행
    output_tensor = model(input_tensor)
    if isinstance(output_tensor, tuple): output_tensor = output_tensor[0]
    full_weights = []
    
    # 3. 영향력 계산 (Gradient 방식)
    print("가중치 계산 중...", end="")
    for i in range(config.pred_len):
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
            
        grad_output = torch.zeros_like(output_tensor).to(device)
        grad_output[0, i, 0] = 1
        
        output_tensor.backward(gradient=grad_output, retain_graph=True)
        
        if input_tensor.grad is not None:
            influence = input_tensor.grad[0, :, 0].abs().detach().cpu().numpy()
            full_weights.append(influence)
        else:
            full_weights.append(np.zeros(config.seq_len))
            
    full_matrix = np.array(full_weights) # (96, 720)

    # 4. 최근 96개 시점 슬라이싱 (96x96 데이터 만들기)
    square_matrix = full_matrix[-96:, -96:] 

    # 5. 시각화 (정사각형 강제 적용)
    plt.figure(figsize=(8, 8)) # 캔버스 크기 정사각형 설정
    
    # [핵심 수정] square=True 옵션 추가
    # 이 옵션이 각 셀을 정사각형으로 만들어 전체 비율을 고정합니다.
    ax = sns.heatmap(square_matrix, cmap='Reds', 
                     cbar_kws={'label': 'Importance (Abs Gradient)', 'shrink': 0.8},
                     square=True)
    
    plt.title("Model Overall Focus (Recent 96 Steps)", fontsize=14, pad=20)
    plt.xlabel("Input Time Steps (Recent 96)", fontsize=12)
    plt.ylabel("Output Time Steps (Future 96)", fontsize=12)
    
    # 여백 자동 조정
    plt.tight_layout()
    plt.show()