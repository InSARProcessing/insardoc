### 1. `conda` 安装 `dolphin`
```bash
conda install -c dolphin
# GPU 配置 - 在已安装12.版本cuda的情况下
pip install --upgrade "jax[cuda12]"
```
如果没有安装 `cuda-toolkit`, 则需要手动安装。
```bash
conda install -c nvidia cuda-toolkit=12.8 #或者其他适合显卡配置的版本
```
---
>注意：`cuda-toolkit` 和 `jax` 版本可能会与系统驱动冲突，建议安装与系统驱动版本匹配的 `cuda-toolkit`。使用`nvidia-smi | grep -i version`查看系统驱动版本。

### 2. 源码安装
```bash
git clone https://github.com/isce-framework/dolphin.git && cd dolphin
conda env create --file conda-env.yml
conda activate dolphin-env
python -m pip install .
```


### Issues: 

安装dolphin时，容易出现gpu模块报错, 输入`TF_CPP_MIN_LOG_LEVEL=0 jax_platforms=cuda python -c "import jax; print('Devices:', jax.devices())"`来验证。如果报错，例如cusparse报错，需要手动建立软链接。

```bash
# 确保你在环境的 lib 目录下
cd ${CONDA_PREFIX}/lib

# 建立软链接，确保系统既能看到带版本号的，也能看到不带版本号的
ln -sf libcusparse.so.12 libcusparse.so
ln -sf libcublas.so.12 libcublas.so
ln -sf libcudnn.so.9 libcudnn.so
ln -sf libcusolver.so.11 libcusolver.so
ln -sf libnvjitlink.so.12 libnvjitlink.so

# 刷新当前的 shell 环境变量（确保没有拼写错误）
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
```