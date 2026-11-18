# CNN Image Classification with PyTorch

## Overview

PyTorchを用いたCNN（畳み込みニューラルネットワーク）による画像分類の実装です。

CIFAR-10を用いたCNNモデルの構築、および事前学習済みモデルを利用したファインチューニングを実施しています。

CNNの基本構造から代表的なネットワークアーキテクチャまでを実装・検証し、モデル構造や学習方法の理解を深めることを目的としています。

## Objectives

- CNNの基本構造と画像分類モデルの実装
- VGG、ResNetなどの代表的なCNNアーキテクチャの理解
- 事前学習済みモデルを用いた転移学習・ファインチューニング
- 異なるモデル構造による画像分類の実装・評価

## Implementations

### 1. CNN models trained on CIFAR-10

CIFAR-10を用いて、CNNモデルを構築・学習します。

| File | Model | Description |
|---|---|---|
| `CNN_pytorch_VGG11.py` | VGG11 | VGG11のネットワーク構造を用いた画像分類 |
| `CNN_pytorch_ResNet.py` | ResNet | Residual Blockを用いたCNNを実装 |


ResNetでは、Residual BlockやSkip Connectionを含むネットワーク構造を実装しています。

### 2. Transfer Learning / Fine-tuning

事前学習済みモデルを利用し、別の画像データセットに対するファインチューニングを実施します。

| File | Model | Dataset / Task |
|---|---|---|
| `CNN_pytorch_ResNet_finetune.py` | ResNet | MVTec AD（screw）の画像分類 |
| `CNN_pytorch_MobileNet_finetune.py` | MobileNetV2 | MVTec AD（screw）の画像分類 |

主な実装内容：

- ImageNet事前学習済みモデルの読み込み
- 分類対象に合わせた最終全結合層の変更
- データ拡張（画像反転、回転、平行移動など）
- 学習済みパラメータの固定・ファインチューニング
- 学習および評価

## Dataset

### CIFAR-10

- 10クラスのカラー画像データセット
- 画像サイズ：32 × 32 pixels
- torchvisionから自動ダウンロード

### MVTec AD

- 製造業における外観検査・異常検知を想定した画像データセット
- 本プロジェクトでは`screw`カテゴリを使用
- `test`フォルダ内の画像をImageFolderで読み込み、クラス分類タスクとして学習・評価

※本実装はMVTec ADの標準的な異常検知ベンチマーク手順とは異なり、test画像を学習・評価用に分割して行う探索的な画像分類実験です。標準的な異常検知性能を示すものではありません。

## Environment

- Python
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- Matplotlib

※ライブラリのバージョンは今後整理・追記予定です。

## How to Run

リポジトリをクローンします。

```bash
git clone https://github.com/Temp-Muraoka/CNN_Image_Classification.git
cd CNN_Image_Classification
```

必要なライブラリをインストールします。

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

実行例：

```bash
python CNN_pytorch_ResNet.py
python CNN_pytorch_VGG11.py
python CNN_pytorch_ResNet_finetune.py
python CNN_pytorch_MobileNet_finetune.py
```

CIFAR-10はtorchvisionによって自動ダウンロードされます。

MVTec ADを使用するスクリプトでは、データセットを取得し、コードで指定されたディレクトリ構成に配置する必要があります。

```text
data/
└── screw/
    └── test/
        ├── good/
        ├── thread_top/
        ├── thread_side/
        └── ...
```

※実際のフォルダ構成は使用するMVTec ADデータセットに合わせてください。

## Results

各モデルの学習・評価結果については、コード内で以下を出力

- 学習曲線（Training / Validation Loss）
- 分類精度（Accuracy）
- Confusion Matrix


## Notes

本リポジトリは個人学習・自主開発を目的としたものです。

既存ライブラリおよび事前学習済みモデルを利用した実装と、独自に定義したネットワーク構造を区別して公開しています。