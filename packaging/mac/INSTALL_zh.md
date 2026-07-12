# VoxCPM2 macOS 安装指南

> 完整中文文档已合并至仓库根目录 [README_zh.md](../../README_zh.md#macos-安装包推荐)。本文档保留简要步骤供 DMG 内引用。

## 1. 下载

在 GitHub [Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases) 下载 `VoxCPM2.dmg`。

## 2. 安装

1. 双击打开 DMG  
2. 将 **VoxCPM2.app** 拖到「应用程序」文件夹  

## 3. 首次打开

应用**未经过 Apple 付费开发者签名**。若提示无法打开：

1. 在「应用程序」中找到 **VoxCPM2.app**  
2. **右键** → **打开**  
3. 在对话框中再次点击 **打开**  

仅需操作一次。

## 4. 准备模型

主模型需自行下载，**不包含在安装包内**：

- 模型：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)  
- 示例：`huggingface-cli download openbmb/VoxCPM2 --local-dir ~/Models/VoxCPM2`  
- 在应用**设置**中选择含 `config.json` 的模型文件夹  

## 5. 使用

启动后浏览器会自动打开 Gradio 页面。

## 系统要求

- macOS **13+**  
- **Apple Silicon（M 系列）** 推荐  
- 模型权重约 **5 GB+** 磁盘空间  

开发者安装请参阅 [README_zh.md](../../README_zh.md#本地使用开发者)。
