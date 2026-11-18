"""
CNNによる画像分類タスク

VGG11
(32*32サイズに合わせた調整あり)
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

import time
# ==========================================
# デバイス確認 (GPUが使用可能か自動判定)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# ==========================
# 入力データの読み込み(CIFAR-10)
# ==========================

import torchvision.datasets as dset
import torchvision.transforms as transforms

DATAROOT = 'data/cifar-10' #データ保存先



# データセットの読み込み（ない場合はダウンロード）
train_data = dset.CIFAR10(root=DATAROOT,
                           train=True,
                           download=True)
                        #    transform=transform)


test_data = dset.CIFAR10(root=DATAROOT,
                           train=False,
                           download=True)
                        #    transform=transform)


print("train_data数", len(train_data))
img, label = train_data[0]
img_np = np.asarray(img) # PIL Image を NumPy 配列に変換
print("train_data.shape=", img_np.shape)


# ====画像の確認（5×5のグリッド表示）=====
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for ax in axes.flat:
    # 0 ～ 49999 の範囲でランダムにインデックスを取得
    ind = random.randint(0, len(train_data) - 1)
    img, label = train_data[ind]

    # 数値ラベルを文字に変換
    label_name = train_data.classes[label]

    # サブプロット（各マス目）に描画
    ax.imshow(img)
    ax.set_title(label_name, fontsize=10)
    ax.axis('off')  # 軸を消す

plt.tight_layout()  # タイトルと画像が重ならないように自動調整
plt.show()


# transform
# スケーリング用の平均と標準偏差
mean_value = (0.5, 0.5, 0.5)
std_value = (0.5, 0.5, 0.5)

transform = transforms.Compose([
                                # transforms.Resize((224, 224)), # 必要に応じて（32x32のまま使う場合は不要）
                                transforms.ToTensor(), #Tensor化（0-1正規化、次元入れ替え、.tensor()が行われている）
                                transforms.Normalize(mean=mean_value, std=std_value) # 標準化
                                ])

train_data.transform = transform
test_data.transform = transform

print("after_transform")
print("train_data数", len(train_data))
img, label = train_data[0]
img_np = np.asarray(img) # PIL Image を NumPy 配列に
print("train_data.shape(tensor)=", img_np.shape)

batch_size = 256

# ミニバッチに分ける（ジェネレータとして使う）。shuffle=True：1epochごとにデータシャッフル
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=False)

# 画面表示用の全学習データ評価用（シャッフルなし）
train_eval_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)



start_time=time.perf_counter() #開始時間


# ==========================================
# CNNモデルの定義
# ==========================================

# シーケンシャル形式での書き方
class VGG11Net(nn.Module):
    def __init__(self, input_dim=[3, 32, 32], output_size=10,
                 dropout_rate = 0.4):

        super(VGG11Net, self).__init__()
        
               
        # -------------------------------------------------------------
        # レイヤー構造の定義
        # -------------------------------------------------------------

        VGG11 = [64,128,256,256,512,512,512,512]

        self.convlayer = nn.Sequential(
                                        nn.Conv2d(in_channels=input_dim[0], out_channels=VGG11[0], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[0]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=VGG11[0], out_channels=VGG11[1], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[1]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=VGG11[1], out_channels=VGG11[2], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[2]),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=VGG11[2], out_channels=VGG11[3], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[3]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=VGG11[3], out_channels=VGG11[4], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[4]),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=VGG11[4], out_channels=VGG11[5], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[5]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=VGG11[5], out_channels=VGG11[6], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[6]),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=VGG11[6], out_channels=VGG11[7], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(VGG11[7]),
                                        nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=2, stride=2), #32*32画像ではこれ以上不可

                                        # 代わりにGAP 1x1 の解像度に圧縮
                                        nn.AdaptiveAvgPool2d((1, 1))
                                        )

        # ダミーデータを通して node_num を取得
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            node_num = self.convlayer(dummy_input).numel()

        self.classifier = nn.Sequential(
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(node_num, 512),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(256, output_size)
        )

        
    def forward(self, x):

        x = self.convlayer(x)
                
        # 全結合層へ渡すために2次元に平坦化
        x = x.view(x.size(0), -1) 
        
        # 全結合層
        x = self.classifier(x)

        return x


img_size = img_np.shape # [C,H,W] = [3, 32, 32]

CNN_net = VGG11Net(input_dim=img_size, output_size=10,
                  dropout_rate = 0.4).to(device)

# モデル構造の確認
print(CNN_net)

# ==========================================
# 損失関数と最適化アルゴリズム（Optimizer）
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(CNN_net.parameters(), lr=0.001)

# ==========================================
# 3. 学習ループ（Training Loop）の実行
# ==========================================

EPOCHS = 20

# ログ保存用のリスト
train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []


print("--- 学習開始 ---")
for epoch in range(EPOCHS):

    CNN_net.train() # モデルを「学習モード」に設定
    train_loss = 0.0

     # # 前のステップで理解した DataLoader からミニバッチを高速供給！
        # for batch_X, batch_y in train_loader:

    # 3. 学習ループ内での「正解ラベル」取り出し
    for images, labels in train_loader:
        # データを GPU（または CPU）に送る
        images = images.to(device)
        labels = labels.to(device)
    
        # print(images.shape) #torch.Size([256, 3, 32, 32])

        # ① 勾配（計算履歴）の初期化
        optimizer.zero_grad()
        
        # ② 予測の計算（順伝播）
        outputs = CNN_net(images)
        
        # ③ 誤差の計算
        # 損失関数は最後、modelの出力(logit)にかける
        loss = criterion(outputs, labels)
        
        # ④ 逆伝播（自動微分で各パラメーターの勾配を計算）
        loss.backward()
        
        # ⑤ パラメーターの更新（一歩賢くなる）
        optimizer.step()


    # ================= 評価フェーズ (精度計算) =================
    CNN_net.eval() #　BatchNorm と Dropout が評価モードになる
        
    
    # 評価関数を定義しておくと綺麗に書けます
    # 評価関数（Loss と Accuracy の両方を返す）
    def evaluate(loader, data_length):
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # 勾配計算を無効化（メモリ節約・高速化）
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # 順伝播
                outputs = CNN_net(images)
                
                # 1. 損失（Loss）の計算と集計
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0) # バッチ内の総Lossを加算
                
                # 2. 精度（Accuracy）の計算と集集
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        
                
        # 全体での平均 Loss と Accuracy(%) を算出
        avg_loss = running_loss / data_length
        accuracy = correct / total
        
        return avg_loss, accuracy

    # 学習データ全体とテストデータ全体の精度を計算
    train_loss, train_acc = evaluate(train_eval_loader, len(train_data))
    test_loss, test_acc = evaluate(test_loader, len(test_data))

    # ログをリストに保存
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # 5エポックごとにログを表示
    if (epoch + 1) % 5 == 0:
        end_time=time.perf_counter()
        elapsed_time=end_time-start_time

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f} - 経過時間：{elapsed_time:.1f}s"
        )

print("--- 学習完了 ---\n")

end_time=time.perf_counter() #終了時間
elapsed_time=end_time-start_time
print(f"処理時間：{elapsed_time}s")

# ==========================================
# 4. 学習曲線の描画
# ==========================================
# lossとaccuracyをデータフレームにまとめる
df_log = pd.DataFrame({"train_loss":train_loss_list,
                       "test_loss":test_loss_list,
                       "train_accuracy":train_acc_list,
                       "test_accuracy":test_acc_list})

# 学習曲線の描画
df_log.plot(style=['r-', 'r--', 'b-', 'b--'])
# 軸ラベルの設定
plt.ylabel("Accuracy or loss")
plt.xlabel("epochs")
plt.show()

# ==========================================
# 5. 最終精度の評価
# ==========================================
# 最終的な正解率
print(f"final test accuracy: {test_acc_list[-1]}")

