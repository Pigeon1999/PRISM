from models import DMSLinear, SegTSF, PRISM
from utils.metric import metric
from utils.tools import EarlyStopping, adjust_learning_rate

import os, time, gc
import torch, numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from ptflops import get_model_complexity_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = os.path.join('./checkpoints/', 'ETT_hour_linear.pth')
if not os.path.exists(path):
    os.makedirs(path)
    
def get_fits_macs(model, config):
    # 채널 수 가져오기
    if hasattr(config, 'channels'):
        channels = config.channels
    elif hasattr(config, 'enc_in'):
        channels = config.enc_in 
    else:
        channels = 1 # 기본값 예외 처리
    
    # --- 기존의 엄밀한 계산(FFT, IFFT, 복소수 4배) 모두 삭제 ---
    # 논문식 꼼수 계산법 적용: (총 파라미터 수) * (채널 수)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_macs = total_params * channels
    
    # ptflops와 비슷한 형태의 문자열(String)로 반환
    if total_macs >= 1e9: return f"{total_macs / 1e9:.2f} GMac"
    elif total_macs >= 1e6: return f"{total_macs / 1e6:.2f} MMac"
    elif total_macs >= 1e3: return f"{total_macs / 1e3:.2f} KMac"
    else: return f"{total_macs:.0f} Mac"

def get_params_str(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e6: return f"{total_params / 1e6:.2f} M"
    elif total_params >= 1e3: return f"{total_params / 1e3:.2f} k"
    else: return str(total_params)
    
def vali(model, vali_loader, criterion, device, pred_len, label_len, config):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs[:, -pred_len:, :]
            batch_y = batch_y[:, -pred_len:, :].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
            
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def train(config, train_loader, vali_loader, test_loader):
    model_dict = {
        'DMSLinear': DMSLinear,
        'SegTSF' : SegTSF,
        'PRISM' : PRISM,
    }
    model = model_dict[config.model].Model(config).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('Model total parameters:', total_params)
    gc.collect()

    is_fits = (getattr(config, 'model', '') == 'FITS') or hasattr(model, 'freq_upsampler')
    is_MixLinear = (getattr(config, 'model', '') == 'MixLinear') or hasattr(model, 'mix_linear')
    
    # 💡 FEDformer인지 확인하는 조건 추가 (config.model 이름이 'FEDformer'인 경우)
    is_fedformer = (getattr(config, 'model', '') == 'FEDformer')
    is_autoformer = (getattr(config, 'model', '') == 'Autoformer')

    if is_fits:
        # --- 1. FITS 모델일 경우: 커스텀 수식 사용 ---
        macs = get_fits_macs(model, config)
        params = get_params_str(model)
    elif is_fedformer or is_autoformer:
        # --- 2. FEDformer 또는 Autoformer 모델일 경우: 다중 입력(input_constructor) 사용 ---
        if hasattr(config, 'channels'):
            channel = config.channels
        elif hasattr(config, 'enc_in'):
            channel = config.enc_in 
            
        try:
            def prepare_fedformer_input(resolution):
                device = next(model.parameters()).device 

                batch_size = 1
                seq_len = config.seq_len
                label_len = getattr(config, 'label_len', 0)
                pred_len = config.pred_len 
                dec_seq_len = label_len + pred_len
                mark_dim = 4

                return {
                    'x_enc': torch.randn(batch_size, seq_len, channel).to(device),
                    'x_mark_enc': torch.randn(batch_size, seq_len, mark_dim).to(device),
                    'x_dec': torch.randn(batch_size, dec_seq_len, channel).to(device),
                    'x_mark_dec': torch.randn(batch_size, dec_seq_len, mark_dim).to(device)
                }

            macs, params = get_model_complexity_info(
                model,
                (1,), # input_constructor를 쓰므로 해상도는 더미값 전달
                input_constructor=prepare_fedformer_input,
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
        except Exception as e:
            macs = "Error"
            params = "Error"
            print(f"\n⚠️ FEDformer ptflops 에러 발생: {e}")

    else:
        # --- 3. 그 외 모델일 경우: 기존 ptflops (단일 입력) 사용 ---
        if hasattr(config, 'channels'):
            channel = config.channels
        elif hasattr(config, 'enc_in'):
            channel = config.enc_in 
            
        try:
            macs, params = get_model_complexity_info(
                model,
                (config.seq_len, channel),
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
        except Exception as e:
            macs = "Error"
            params = "Error"
            print(f"\n⚠️ 일반 모델 ptflops 에러 발생: {e}")

    # 최종 출력
    print('Computational complexity: ' + str(macs))
    print('Number of parameters: ' + str(params))
    
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

    if config.criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    elif config.criterion == 'MAE':
        criterion = torch.nn.L1Loss()
    elif config.criterion == 'Huber':
        criterion = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
                                        steps_per_epoch = train_steps,
                                        pct_start = 0.3,
                                        epochs = config.epochs,
                                        max_lr = config.learning_rate,)

    for epoch in range(config.epochs):
        iter_count = 0
        train_loss = []
        
        model.train()
        epoch_time = time.time()
        max_memory = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            iter_count += 1
            optimizer.zero_grad()
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs[:, -config.pred_len:, :]
            batch_y = batch_y[:, -config.pred_len:, :].to(device)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            if (i + 1) % 100 == 0:
                time_now = time.time()
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((config.epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
            
            loss.backward()
            optimizer.step()
            
            current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            max_memory = max(max_memory, current_memory)
            
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_loader, criterion, device, config.pred_len, getattr(config, 'label_len', 0), config)
        test_loss = vali(model, test_loader, criterion, device, config.pred_len, getattr(config, 'label_len', 0), config)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, scheduler, epoch + 1, config)
        print('=========================================')
        print(f"Max Memory (MB): {max_memory}")
    
    return model

def test(config, test_dataset, test_loader, model):
    model.to(device).eval()
    os.makedirs("./img", exist_ok=True)
    # folder_path = f'./results/{time.time():.0f}/'
    # os.makedirs(folder_path, exist_ok=True)

    # 누적 대신 합/개수 방식(예: MSE/MAE)
    n_samples = 0
    sum_abs = 0.0
    sum_sq  = 0.0

    # 필요 시 상관계수 등 위해서만 소규모 샘플 보관
    sample_plot_done = False

    begin_time = time.time()
    with torch.inference_mode():
        for i, (bx, by, bxm, bym) in enumerate(test_loader):
            bx = bx.float().to(device, non_blocking=True)
            by = by.float().to(device, non_blocking=True)

            out = model(bx)
            # 출력 길이 방어적 슬라이스
            if isinstance(out, tuple): out = out[0]
            pred_len = min(config.pred_len, out.shape[1], by.shape[1])
            out = out[:, -pred_len:, :]
            tgt = by[:, -pred_len:, :]

            # 메트릭 즉시 업데이트(여기서는 MSE/MAE 예시)
            diff = (out - tgt).detach()
            sum_abs += diff.abs().sum().item()
            sum_sq  += (diff ** 2).sum().item()
            n_samples += diff.numel()

            # 필요하면 샘플 1회만 플롯
            if (i % 20 == 0) and (not sample_plot_done):
                x_cpu  = bx.detach().cpu()
                y_cpu  = tgt.detach().cpu()
                o_cpu  = out.detach().cpu()
                # 채널 존재/인덱스 방어
                ch = min(x_cpu.shape[-1]-1, 0)
                # 입력 구간 + 예측 구간 이어 붙이기 (1D)
                gt = np.concatenate((x_cpu[0, :, ch].numpy(), y_cpu[0, :, ch].numpy()), axis=0)
                pd = np.concatenate((x_cpu[0, :, ch].numpy(), o_cpu[0, :, ch].numpy()), axis=0)
                plt.figure(figsize=(12, 5))
                plt.plot(gt[config.seq_len:], label='true')
                plt.plot(pd[config.seq_len:], label='pred')
                plt.legend()
                plt.savefig(f'./img/{config.model}.png')
                plt.close()
                sample_plot_done = True

            # 루프 말미 GC(가볍게)
            del bx, by, bxm, bym, out, tgt, diff
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # ← 루프 내 상시호출은 지양. 필요하면 "간헐적"으로만
        # end for

    elapsed = time.time() - begin_time
    mae = sum_abs / n_samples
    mse = sum_sq  / n_samples
    rmse = np.sqrt(mse)

    # 결과 출력
    print("==================================================")
    print(f"mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, "
          f"ms/sample:{(elapsed*1000)/len(test_dataset):.3f}")
    print(f"inference time: {elapsed:.3f}s")

    # 마지막에만 캐시 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()