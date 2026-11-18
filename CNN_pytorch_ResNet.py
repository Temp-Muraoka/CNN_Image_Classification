"""
CNNによる画像分類タスク

ResNet
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

test_data = dset.CIFAR10(root=DATAROOT,
                           train=False,
                           download=True)

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

# -------------------------------------------------------------
# 残差ブロック（1つの Skip Connection を持つ最小単位）
# -------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主経路（Conv -> BN -> ReLU -> Conv -> BN）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 迂回路（スキップ接続）
        # 解像度（stride > 1）やチャネル数が変わる時だけ 1x1 Conv でサイズを合わせる
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels: #ここはサイズが変化するときだけ
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # 元の入力を取っておく

        # メインの計算
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 元の入力(identity)をそのまま足し算
        out += self.shortcut(identity)
        out = self.relu(out)

        return out

# -------------------------------------------------------------
# ResNet 本体（8つの Block を順番に並べた構造）
# -------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, input_dim=[3, 32, 32], output_size=10):
        super(ResNet, self).__init__()

        # 【入り口】通常の畳み込み (32x32x3 -> 32x32x64)
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 【Stage 1】 チャネル数: 64, 解像度: 32x32
        self.block1_1 = BasicBlock(in_channels=64, out_channels=64, stride=1)
        self.block1_2 = BasicBlock(in_channels=64, out_channels=64, stride=1)

        # 【Stage 2】 チャネル数: 128, 解像度: 16x16 (ここで画像サイズ半分)
        self.block2_1 = BasicBlock(in_channels=64,  out_channels=128, stride=2)
        self.block2_2 = BasicBlock(in_channels=128, out_channels=128, stride=1)

        # 【Stage 3】 チャネル数: 256, 解像度: 8x8 (ここで画像サイズ半分)
        self.block3_1 = BasicBlock(in_channels=128, out_channels=256, stride=2)
        self.block3_2 = BasicBlock(in_channels=256, out_channels=256, stride=1)

        # 【Stage 4】 チャネル数: 512, 解像度: 4x4 (ここで画像サイズ半分)
        self.block4_1 = BasicBlock(in_channels=256, out_channels=512, stride=2)
        self.block4_2 = BasicBlock(in_channels=512, out_channels=512, stride=1)

        # 【全結合の前処理】 4x4 の解像度を 1x1 に圧縮
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 【全結合層】 512 個の特徴量から 10 クラスへ分類
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        # 1. 入力画像 (3, 32, 32)
        x = self.stem(x)       # -> (64, 32, 32)

        # 2. Stage 1
        x = self.block1_1(x)   # -> (64, 32, 32)
        x = self.block1_2(x)   # -> (64, 32, 32)

        # 3. Stage 2
        x = self.block2_1(x)   # -> (128, 16, 16) ※半分にダウンサンプル
        x = self.block2_2(x)   # -> (128, 16, 16)

        # 4. Stage 3
        x = self.block3_1(x)   # -> (256, 8, 8)   ※半分にダウンサンプル
        x = self.block3_2(x)   # -> (256, 8, 8)

        # 5. Stage 4
        x = self.block4_1(x)   # -> (512, 4, 4)   ※半分にダウンサンプル
        x = self.block4_2(x)   # -> (512, 4, 4)

        # 6. 出力層へ
        x = self.avg_pool(x)   # -> (512, 1, 1)
        x = torch.flatten(x, 1)# -> (512)
        x = self.fc(x)         # -> (10)

        return x

img_size = img_np.shape # [C,H,W] = [3, 32, 32]

CNN_net = ResNet(input_dim=img_size, output_size=10,
                 ).to(device)

# モデル構造の確認
print(CNN_net)

# ==========================================
# 損失関数と最適化アルゴリズム（Optimizer）
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(CNN_net.parameters(), lr=0.001)

# ==========================================
# 学習ループ（Training Loop）の実行
# ==========================================

EPOCHS = 20

# ログ保存用のリスト
train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []


print("--- 学習開始 ---")
for epoch in range(EPOCHS):

    CNN_net.train() # モデルを「学習モード」に設定

    train_loss = 0.0
    train_correct = 0

    # 3. 学習ループ内での「正解ラベル」取り出し
    for images, labels in train_loader:

        # データを GPU（または CPU）に送る
        images = images.to(device)
        labels = labels.to(device)

        # 勾配（計算履歴）の初期化
        optimizer.zero_grad()
        
        # 予測の計算（順伝播）
        outputs = CNN_net(images)
        
        # 誤差計算
        # 損失関数は最後、modelの出力(logit)にかける
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメーターの更新
        optimizer.step()

        train_loss += loss.item() * images.size(0) # バッチ内の総Lossを加算
                    
        # 精度（Accuracy）の計算と集集
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        
    # 平均 Loss と Accuracy(%) 
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)


    # ================= 評価フェーズ =================
    CNN_net.eval() #　評価モード BatchNorm と Dropoutの動作変化
        
    test_loss = 0.0
    test_correct = 0
    
    with torch.no_grad(): # 勾配計算を無効化

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = CNN_net(images)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0) # バッチ内の総Lossを加算
            
            # 精度（Accuracy）の計算と集集
            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
           
    # 平均 Loss と Accuracy(%) 
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)


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
# 学習曲線の描画
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
# 最終精度の評価
# ==========================================
# 最終的な正解率
print(f"final test accuracy: {test_acc_list[-1]}")



# ==========================================
# テストデータからランダムに25枚抽出し、予測結果を5x5で表示
# ==========================================
CNN_net.eval() # 評価モード

# テストデータからランダムに25個のインデックスを重複なしで選択
indices = random.sample(range(len(test_data)), 25)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))

# 正規化を元に戻すための逆変換（表示を綺麗にするため: pixel = val * 0.5 + 0.5）
# ※transformで mean=0.5, std=0.5 を使っているため
def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor.clamp(0, 1)

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        # test_data[idx] は (tensor_img, label) を返す
        img_tensor, true_label = test_data[idx]
        
        # モデルに入力するためにバッチ次元を追加してデバイスへ送る [1, 3, 32, 32]
        x = img_tensor.unsqueeze(0).to(device)
        outputs = CNN_net(x)
        pred_label = outputs.argmax(dim=1).item()
        
        # クラス名の取得
        true_name = test_data.classes[true_label]
        pred_name = test_data.classes[pred_label]
        
        # 表示用に Tensor を NumPy (H, W, C) に変換
        img_display = unnormalize(img_tensor).permute(1, 2, 0).numpy()
        
        ax.imshow(img_display)
        
        # 正解なら緑、不正解なら赤でタイトルを表示
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9, color=color)
        ax.axis('off')

plt.tight_layout()
plt.show()
