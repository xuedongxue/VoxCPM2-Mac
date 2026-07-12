# VoxCPM2 macOS 安装指南

面向通过 **DMG 安装包** 使用的用户（无需自行安装 Python）。

## 1. 下载

在 GitHub [Releases](https://github.com/xuedongxue/VoxCPM2-Mac/releases) 下载 `VoxCPM2.dmg`。

## 2. 安装

1. 双击打开 DMG  
2. 将 **VoxCPM2.app** 拖到「应用程序」文件夹  

## 3. 首次打开

应用**未经过 Apple 付费开发者签名**（开源公益项目）。若提示无法打开：

1. 在「应用程序」中找到 **VoxCPM2.app**  
2. **右键** → **打开**  
3. 在对话框中再次点击 **打开**  

仅需操作一次；之后可正常双击启动。

## 4. 准备模型

主模型需自行下载，**不包含在安装包内**：

- 模型地址：[openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)  
- 下载整个仓库到本地文件夹（须含 `config.json`）  
- 示例：`huggingface-cli download openbmb/VoxCPM2 --local-dir ~/Models/VoxCPM2`

在应用**设置**中选择该文件夹路径。ASR、降噪等辅助模型可在首次使用时按需联网下载。

## 5. 使用

启动后浏览器会自动打开 Gradio 页面，按界面提示进行语音合成即可。

## 系统要求

- macOS **13+**  
- **Apple Silicon（M 系列）** 推荐（arm64 安装包）  
- 为模型预留约 **5 GB+** 磁盘空间  

## 开发者安装

若需从源码运行或参与开发，请参阅仓库根目录 [README.md](../../README.md) 中的「本地使用（开发者）」章节。
