## **Doris**

>doris安装前提：**需要安装fftw3**

**步骤**： 参考[APRILAB百科](https://aprilab-nwpu.feishu.cn/wiki/Q9z4wyKVJikLdVk1XPYcfWwlnGg)
* 下载doris源码 
```bash
git clone https://github.com/aprilab-dev/doris.git
```
* 在源码路径下执行：
```bash
cd doris/src
ls -l ./configure # 查看文件属性
chmod +x ./configure # 赋予权限

# 开始配置
./configure

# 安装
make -j 16
make install
```
----
按照以上的教程安装时，以下**注意事项**:
1. 无法运行 ./configure， 则需要安装`tcsh`
```bash
sudo apt update
sudo apt install tcsh
```
2. 在运行`./configure`时，需要确定系统中相关库的地址：
```bash
# 确认相关库
pkg-config --variable libdir fftw3 #确认fftw3安装路径
sudo find /usr -name "fftw3.h" 2>/dev/null # 确认fftw3.h安装路径
dpkg -l | grep -i lapack  #确认 lapack安装路径
sudo find /usr -name "liblapack.a" 2>/dev/null  #确认liblapack.a安装路径
$ lscpu | grep -i "byte order" # 确认是否是Little Endian系统
# 输出是：（确认是Little Endian系统）
# Byte Order:                           Little Endian
```
在运行`./configure`时按照提示输入，生成CMAKE文件：
``` bash
 Creating Makefile for:
 compiler:       g++
 fftw:           y
 FFTW LIB DIR:     /usr/lib/x86_64-linux-gnu
 FFTW INCLUDE DIR: /usr/include
 veclib:         n
 lapack:         y
 LAPACK dir:     /usr/lib/x86_64-linux-gnu/lapack
 Little endian:  y
 DEBUG version:  n
 Install in dir: ~/tools/doris/install
```
然后再运行`make`和`make install`

## **teresa**

* 下载源码
* 创建虚拟环境
* pip install
----
**注意**
安装完成后，需要找到该路径下的.py文件：`teresa/teresa/processor/dorisProcessor.py`
修改：
```bash
# 原版：
    def _doris(self, arg):
        _DORIS = os.getenv('STACK_BUILDER_DORIS', '/home/junjun/doris/doris/src/doris')
        with open("doris.log", 'w') as log_file:
            return subprocess.call([_DORIS, arg], stdout=log_file, stderr=log_file)
#
# 改为：
#
    def _doris(self, arg):
        _DORIS = os.getenv('STACK_BUILDER_DORIS', 'PATH/TO/YOUR/DORIS/EXE') # 例如'/usr/local/bin/doris'
        with open("doris.log", 'w') as log_file:
            return subprocess.call([_DORIS, arg], stdout=log_file, stderr=log_file)
```
---
**使用**

详细查看`teresa\README.md`以及`teresa\templates\doris_params.md`查看参数说明；

`teresa/teresa/utils/plotTools.py`为画图工具，以查看结果干涉图；

（更多细节有待补充）

## **s1_coregistration**

**注意事项**
<!-- 
SNAP安装在windows下的情况下，可以在~/.bashrc中加入以下输入来直接使用gpt命令：
```bash
# SNAP
export PATH=$PATH:/mnt/d/SNAP/esa-snap/bin/ # SNAP安装路径
alias gpt='/mnt/d/SNAP/esa-snap/bin/gpt.exe' # 修改引用参数名称
``` -->
* SNAP必须安装在Linux系统中，否则会出现路径冲突

* 使用前，python依赖库还需要额外安装：
```bash
pip install "GeoAlchemy2<0.14.0" packaging progressbar2 psycopg2 pyyaml "SQLAlchemy>=1.4,<2.0" SQLAlchemy-Utils>=0.37 spatialist>=0.12.1
pip install numpy scipy matplotlib rasterio gdal shapely fiona pyproj Pillow
```

* 另外需要在`s1_coregistration/image_ingestion/snap_processor.py`中修改gpt路径：
```bash
def __init__(self, snap_command='/PATH/TO/YOUR/PATH', cache_size=8, nr_proc=12, use_geoid=False):
```

* 以及如果wsl内部存储不足，在`s1_coregistration/image_ingestion/coregister_assembly.py`中修改tmp临时工作文件夹路径：
```bash
c = Coregistrator(cache_size=cache_size, nr_proc=nr_proc, output_dir=output_dir, temp_dir='/mnt/d/tmp')
```
如果使用WSL，注意使用内存情况 --> `free -h`查看WSL系统内存，`nproc`查看CPU核心数：

设置JVM使用内存为：`export _JAVA_OPTIONS="-Xmx16G"`

**注意**运行gpt时使用内存`cache_size`和JAVA使用内存加起来**不能超过**WSL系统内存

## **depsi**
>需要安装**MATLAB in Linux**（不适用MATLAB2025b）

**安装：**

1. 下载源码；
2. 使用`pdepsi/examples/`下的脚本，修改对应路径和基本配置信息即可。 

**说明：**

* 适用于处理涪城SAR数据：
    - `pdepsi/examples/pdepsi_fucheng_test.m`

* 适用于处理S1数据： 

    - `pdepsi/examples/S1/depsi_local_project_S1.m`  
    - `pdepsi/examples/S1/depsi_network_project_S1.m`
    - `pdepsi/examples/S1/depsi_spatialunwrap_project_S1.m` --> SBAS处理S1数据（不推荐， 版本比较老旧）；

* 适用于处理TerraSAR的数据：
    - `pdepsi/examples/other_sensors/`下的所有脚本

* `/home/wang_xuanxuan/tools/pdepsi/doc/` 内包含所有使用说明；

* 所有`@ *`的为对象，内部含有可使用函数，为封装完成的p文件；

* 关于输入文件：**teresa**生成的所有文件中 `YYMMDD/slave.res`和`YYMMDD/cint.minrefdem.raw`将作为被pdepsi读取的文件

* 所有m文件适用于在终端中无图形界面运行（防止JVM占用内存）：
```bash
# example
cd /home/wang_xuanxuan/tools/pdepsi/examples
matlab -nodesktop -nosplash -r pdepsi_fucheng_test
# 如果脚本存放地址不在包含数据和结果文件的项目文件夹下，脚本中路径建议修改为绝对路径
```
