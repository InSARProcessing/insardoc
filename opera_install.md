## `Conda` 安装
`
conda create --name opera
conda activate opera
conda install -c conda-forge isce3-cuda compass 
conda install -c dolphin
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
`

## 可选：源码安装 `COMPASS`和`S1-READER`
1. 安装 `COMPASS`
```
git clone https://github.com/opera-adt/COMPASS.git && cd COMPASS
conda env update --name opera --file environment.yml
python -m pip install -e .
```

2. 安装 `S1-READER`
```
git clone https://github.com/opera-adt/s1-reader.git
cd s1-reader
conda env update --name opera --file environment.yml
python -m pip install -e .
```

## 可选：`CMAKE` 安装 `ISCE3`

**1. 准备源码**
```
cd ~/tools/isce3
mkdir -p src && cd src
git clone https://github.com/isce-framework/isce3.git
```

**2. 环境配置**
```
cd isce3
conda env update --name opera --file environment.yml
```

**3. 编译构建**
```
# 为避免系统库冲突，在conda下安装gcc等库
mamba install gcc_linux-64=12 gxx_linux-64=12 gfortran_linux-64=12 -y

# 然后回到isce3目录
cd ../..

# 输出系统变量
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$CXX

# 配置 CMake 项目
mkdir -p {install, build}
sudo apt install ninja-build build-essential
cd build
# 如果安装过，先清理旧缓存
rm -rf * 

cmake ../src/isce3 \
    -G "Ninja" \
    -DWITH_CUDA=ON \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DWITH_CUDA=ON \
    -DCMAKE_INSTALL_PREFIX=../install
# ninja 安装
ninja -j$(nproc)
ninja install
```
需要注意的是, 如果`ninja install` 失败, 且出现`compute_120`相关的错误，需删除`build\`下的所有文件，然后重新在cmake后，执行以下 sed 脚本
```
sed -i 's/compute_120/compute_86/g' build.ninja
sed -i 's/sm_120/sm_86/g' build.ninja
```

**4. 配置测试环境 (关键！)**

在`~/.bashrc`中加入以下内容：
```
export CUDAHOSTCXX=$CXX
export CUDACXX=$(which nvcc)
alias load_isce3='conda activate isce3 && \
                 export ISCE3_INSTALL=~/tools/isce3/install && \
                 export PATH=$ISCE3_INSTALL/bin:$ISCE3_INSTALL/packages/nisar/workflows:$PATH && \
                 export LD_LIBRARY_PATH=$ISCE3_INSTALL/lib:$ISCE3_INSTALL/lib64:$LD_LIBRARY_PATH && \
                 export PYTHONPATH=$ISCE3_INSTALL/packages:$PYTHONPATH'
```

**5. 运行测试**
```
ctest --output-on-failure
```
