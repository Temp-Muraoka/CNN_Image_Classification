"""
CNNによる画像分類タスク

学習済みResNetのファインチューニング
MVTec ADの 'screw(ねじ)'の異常画像識別
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset

from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random

import time


# ==========================================
# デバイス確認 (GPUが使用可能か自動判定)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

# ==========================================
# 1. 入力データの読み込み (MVTec AD)
# ==========================================
# MVTec ADの 'screw'
# ※MVTec ADは 'test' フォルダ配下にクラス別フォルダ（good, scratch, bent 等）が並んでいます
# trainデータには正常画像しかないため、testデータを使って学習
DATA_DIR = 'data/screw/test'

# 事前学習済みResNet (ImageNet) に合わせた標準の正規化パラメータ
mean_value = (0.485, 0.456, 0.406)
std_value  = (0.229, 0.224, 0.225)

# 学習用（データ拡張あり）と評価用（リサイズと正規化のみ）のTransform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 左右反転（P:確率）
    transforms.RandomVerticalFlip(p=0.5),   # 上下反転
    transforms.RandomRotation(degrees=15),  # 回転
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # シフト
    # transforms.ColorJitter(brightness=0.2, contrast=0.2), #明るさ、コントラスト
    # transforms.GaussianBlur(kernel_size=(3, 3))、# ぼかし
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_value, std=std_value)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_value, std=std_value)
])

# ImageFolderでフォルダ構造から自動ラベリング（まずはTransformなしで読み込み）
base_dataset = dset.ImageFolder(root=DATA_DIR)
class_names = base_dataset.classes
num_classes = len(class_names)

print(f"総データ数: {len(base_dataset)}枚")
print(f"検出されたクラス ({num_classes}種類): {class_names}")

# ==== 画像の確認（5×5のグリッド表示：学習前の生画像） =====
num_preview = min(25, len(base_dataset))
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
preview_indices = random.sample(range(len(base_dataset)), num_preview)

for i, ax in enumerate(axes.flat):
    if i < num_preview:
        img, label = base_dataset[preview_indices[i]]
        ax.imshow(img)
        ax.set_title(class_names[label], fontsize=10)
    ax.axis('off')

plt.suptitle("MVTec AD Sample Images (Before Transform)", fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 2. Train / Test 分割 と DataLoader の作成
# ==========================================
# クラス比率を保ったまま（stratify）、学習用80% : テスト用20%に分割
targets = base_dataset.targets
train_idx, test_idx = train_test_split(
    list(range(len(base_dataset))),
    test_size=0.2,
    random_state=42,
    stratify=targets
)

# それぞれ別のTransformを適用するためDatasetを分けてからSubset化
train_full_dataset = dset.ImageFolder(root=DATA_DIR, transform=train_transform)
test_full_dataset  = dset.ImageFolder(root=DATA_DIR, transform=test_transform)

train_data = Subset(train_full_dataset, train_idx)
test_data  = Subset(test_full_dataset, test_idx)

print(f"学習データ数: {len(train_data)}枚 / テストデータ数: {len(test_data)}枚")

# MVTec ADはデータ数が少ないため、バッチサイズは小さめ（16など）に設定
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

start_time = time.perf_counter()


# ==========================================
# 3. 既存ResNetの読み込みとファインチューニング設定
# ==========================================
import torchvision.models as models

# ImageNet事前学習済みの ResNet18 をロード
weights = models.ResNet18_Weights.DEFAULT
CNN_net = models.resnet18(weights=weights)

print("--- 読み込んだモデルの構造 ---")
print(CNN_net)  # デフォルトでは fc 層の出力が 1000クラス（ImageNet用）になっている


# 最後の全結合層（fc）を MVTec AD のクラス数（num_classes）に付け替え
in_features = CNN_net.fc.in_features # 全結合層の入力ノード数
CNN_net.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)

CNN_net = CNN_net.to(device)
print("--- 変更後モデルの構造 ---")
print(CNN_net)

# ==========================================
# 重みの固定（Freeze）または全層微調整の設定
# ==========================================
# パターンA：特徴抽出部を固定して、最後の層だけをサクッと学習させたい場合
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# パターンB：全体を微調整（Fine-Tuning）したい場合
for param in CNN_net.parameters():
    param.requires_grad = True

# ==========================================
# 損失関数と最適化アルゴリズム（Optimizer）
# ==========================================

criterion = nn.CrossEntropyLoss()

# ファインチューニングでは、事前学習済みの重みを壊さないよう学習率（lr）を小さめに（1e-4）
optimizer = optim.Adam(CNN_net.parameters(), lr=0.0001)

# ==========================================
# 学習ループ（Training Loop）の実行
# ==========================================

EPOCHS = 30

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

# テストデータが25枚未満の場合もエラーにならないよう調整
num_samples = min(25, len(test_data))
indices = random.sample(range(len(test_data)), num_samples)

fig, axes = plt.subplots(5, 5, figsize=(12, 12))

# ImageNetの標準化を元に戻す逆変換関数
def unnormalize(tensor):
    mean = torch.tensor(mean_value).view(3, 1, 1)
    std = torch.tensor(std_value).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            idx = indices[i]
            img_tensor, true_label = test_data[idx]

            x = img_tensor.unsqueeze(0).to(device)
            outputs = CNN_net(x)
            pred_label = outputs.argmax(dim=1).item()

            true_name = class_names[true_label]
            pred_name = class_names[pred_label]

            img_display = unnormalize(img_tensor).permute(1, 2, 0).numpy()
            ax.imshow(img_display)

            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9, color=color)
        ax.axis('off')

plt.suptitle("Test Predictions (Green: Correct, Red: Incorrect)", fontsize=14)
plt.tight_layout()
plt.show()