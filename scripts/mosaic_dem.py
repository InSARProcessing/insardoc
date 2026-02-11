#!/usr/bin/env python3
"""
DEM Stitcher - Fixed version with proper byte order handling
Supports SRTM and Copernicus DEM sources
Converts between orthometric (EGM96/EGM2008) and ellipsoidal (WGS84) heights
"""

import os
import re
import sys
import zipfile
import argparse
import requests
import pathlib
import subprocess
import numpy as np
import ctypes
from math import floor
from osgeo import gdal, osr
from ctypes import c_char_p, c_int, POINTER, byref

# --------------- 辅助文件 ----------------
file_path = os.path.abspath(__file__)
_DEM_STITCHER_LIB = os.path.join(os.path.dirname(file_path), 'contrib', 'demStitch.so')
EGM2008_TIFF = os.path.join(os.path.dirname(file_path), 'utils', 'egm_model', 'egm2008.tif')
EGM96_TIFF = os.path.join(os.path.dirname(file_path), 'utils', 'egm_model', 'egm96_global.tif')


# --------------- 辅助读取 DEM tiles 函数----------------

def ensure_hgt_extracted(zip_path):
    hgt_path = zip_path.replace(".SRTMGL1.hgt.zip", ".hgt").replace(".hgt.zip", ".hgt")
    if not os.path.exists(hgt_path):
        with zipfile.ZipFile(zip_path) as zf:
            hgt_name = [f for f in zf.namelist() if f.endswith('.hgt')][0]
            with zf.open(hgt_name) as src, open(hgt_path, 'wb') as dst:
                dst.write(src.read())
    return hgt_path


def lat_lon_to_cop_tile(lat, lon):
    lat_deg = int(floor(lat))
    lon_deg = int(floor(lon))
    lat_p = 'N' if lat >= 0 else 'S'
    lon_p = 'E' if lon >= 0 else 'W'
    return f"Copernicus_DSM_COG_10_{lat_p}{abs(lat_deg):02d}_00_{lon_p}{abs(lon_deg):03d}_00_DEM"  # 默认 30m 分辨率


def get_bounds(tile_path):
    fname = os.path.basename(tile_path)
    match = re.match(r'([NS])(\d{2})([EW])(\d{3})\.hgt', fname)
    if match:
        lat = float(match.group(2)) if match.group(1) == 'N' else -float(match.group(2))
        lon = float(match.group(4)) if match.group(3) == 'E' else -float(match.group(4))
        return [lon, lat, lon + 1, lat + 1]
    try:
        ds = gdal.Open(tile_path)
        if ds:
            gt = ds.GetGeoTransform()
            xsize, ysize = ds.RasterXSize, ds.RasterYSize
            return [gt[0], gt[3] + ysize * gt[5], gt[0] + xsize * gt[1], gt[3]]
    except:
        pass
    return None

# --------------- 按照 dem 类型查找 tiles ----------------


def find_srtm_tiles(latlim, lonlim, dem_dir):
    """查找SRTM tiles - 按照从北到南、从西到东的顺序组织

    Args:
        latlim: [max_lat, min_lat] 纬度范围
        lonlim: [min_lon, max_lon] 经度范围
        dem_dir: DEM文件目录

    Returns:
        tiles列表，按照ISCE2期望的顺序(从北到南，从西到东)
    """
    tiles = []
    # 修复: 从北到南遍历 (从大到小)
    # latlim[0] = max_lat, latlim[1] = min_lat
    # range参数: (start, stop, step)
    # 需要从 ceil(max_lat)-1 到 floor(min_lat) (包含)，步长-1
    for lat in range(int(np.ceil(latlim[0])) - 1, int(np.floor(latlim[1])) - 1, -1):
        # 从西到东遍历 (从小到大)
        for lon in range(int(np.floor(lonlim[0])), int(np.ceil(lonlim[1]))):
            ns = 'N' if lat >= 0 else 'S'
            ew = 'E' if lon >= 0 else 'W'
            base = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            for ext in [".hgt", ".SRTMGL1.hgt.zip", ".hgt.zip"]:
                p = os.path.join(dem_dir, base + ext)
                if os.path.exists(p):
                    if p.endswith('.zip'):
                        p = ensure_hgt_extracted(p)
                    tiles.append(p)
                    break
    return tiles


def find_copernicus_tiles(latlim, lonlim, dem_dir, allow_download=True):
    """查找Copernicus tiles - 按照从北到南、从西到东的顺序组织

    Args:
        latlim: [max_lat, min_lat] 纬度范围
        lonlim: [min_lon, max_lon] 经度范围
        dem_dir: DEM文件目录
        allow_download: 是否允许下载缺失的tiles

    Returns:
        tiles列表，按照ISCE2期望的顺序(从北到南，从西到东)
    """
    tiles = []
    needed = []
    # 使用numpy.arange，步长为-1
    for lat in np.arange(np.ceil(latlim[0]) - 1, np.floor(latlim[1]) - 1, -1):
        # 从西到东遍历 (从小到大)
        for lon in np.arange(np.floor(lonlim[0]), np.ceil(lonlim[1])):
            tile_name = lat_lon_to_cop_tile(lat, lon) + ".tif"
            p = os.path.join(dem_dir, tile_name)
            if os.path.exists(p):
                try:
                    ds = gdal.Open(p)
                    if ds:
                        tiles.append(p)
                        ds = None
                        continue
                except:
                    pass
            needed.append((lat, lon))

    if allow_download:
        for lat, lon in needed:
            dl = download_cop30(lat, lon, dem_dir)
            if dl:
                tiles.append(dl)
    else:
        if needed:
            print(f"Warning: {len(needed)} Copernicus tiles missing, but --local-only is activated。")
    return tiles


def download_cop30(lat, lon, out_dir, overwrite=False):
    base_name = lat_lon_to_cop_tile(lat, lon)          # e.g., Copernicus_DSM_COG_10_N34_00_E116_00_DEM
    file_name = base_name + ".tif"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{base_name}/{file_name}"
    out_path = os.path.join(out_dir, file_name)

    if os.path.exists(out_path) and not overwrite:
        print(f"Already existed: {out_path}")
        return out_path

    print(f"Download: {url}")
    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"Save to: {out_path}")
            return out_path
        else:
            print(f"HTTP {r.status_code} Download fails: {file_name}")
            return None
    except Exception as e:
        print(f"Download error {file_name}: {e}")
        return None


# --------------- 主流程 ----------------
def prepare_dem():
    parser = argparse.ArgumentParser(
        description="DEM preparation: choose SRTM or Copernicus (mutually exclusive)."
    )
    parser.add_argument(
        'bbox',
        nargs=4,
        type=float,
        metavar='CROP_BOUND_BOX',
        help="Bounding box: LAT_MIN LAT_MAX LON_MIN LON_MAX (e.g., 34 36 116 118)"
    )
    parser.add_argument(
        '--source',
        choices=['srtm', 'copernicus'],
        required=True,
        help="Choose DEM source: 'srtm' (uses EGM96) or 'copernicus' (uses EGM2008)"
    )
    parser.add_argument(
        '-d', '--dem-dir',
        default='./dem_tiles',
        type=str,
        help="Local DEM directory (default: ./dem_tiles)"
    )
    parser.add_argument(
        '--local-only',
        action='store_true',
        help="For Copernicus: skip downloading missing tiles"
    )
    parser.add_argument(
        '-o', '--output',
        default='dem',
        type=str,
        help="Output prefix (default: dem)"
    )
    parser.add_argument(
        '--height',
        choices=['orthometric', 'ellipsoidal'],
        default='orthometric',
        help="Output height type for Doris: 'orthometric' (default) or 'ellipsoidal'"
    )
    parser.add_argument(
        '--sample',
        default=3601,
        type=int,
        help="Sample per DEM tile (default: 3601)"
    )
    args = parser.parse_args()

    # 调用核心函数
    prepare_dem_core(
        bbox=args.bbox,
        source=args.source,
        dem_dir=args.dem_dir,
        local_only=args.local_only,
        output=args.output,
        height=args.height,
        sample=args.sample
    )


def prepare_dem_core(
    bbox,
    source,
    dem_dir='./dem_tiles',
    local_only=False,
    output='dem',
    height='orthometric',
    sample=3601
):
    """
    Core DEM preparation logic, accepting explicit parameters.

    Parameters:
        bbox (list/tuple): [lat_min, lat_max, lon_min, lon_max]
        source (str): 'srtm' or 'copernicus'
        dem_dir (str): local DEM directory
        local_only (bool): skip download for Copernicus
        output (str): output prefix
        height (str): 'orthometric' or 'ellipsoidal'
        sample (int): samples per tile (3601 for 1 arcsec, 1201 for 3 arcsec)
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    latlim = [max(lat_min, lat_max), min(lat_min, lat_max)]
    lonlim = [min(lon_min, lon_max), max(lon_min, lon_max)]

    print(f"Region: Latitude {latlim[0]}°N–{latlim[1]}°N, Longitude {lonlim[0]}°E–{lonlim[1]}°E")
    print(f"DEM source: {source.upper()}")
    print(f"Directory: {dem_dir} {'(local only)' if local_only and source == 'copernicus' else ''}")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

    if source == 'srtm':
        tiles = find_srtm_tiles(latlim, lonlim, dem_dir)
        egm_source = 'egm96'
    else:  # copernicus
        tiles = find_copernicus_tiles(latlim, lonlim, dem_dir, allow_download=not local_only)
        egm_source = 'egm2008'

    if not tiles:
        raise FileNotFoundError(f"Cannot find any {source.upper()} tiles, please check the region or the directory。")

    print(f"Found {len(tiles)} tiles:")
    for t in tiles:
        print(f"  {os.path.basename(t)}")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

    ortho_tif = f"{output}_orthometric.tif"
    output_file = f"{output}.dem"

    # 拼接tiles
    data, h, w = mosaic_tiles(tiles, latlim, lonlim, output_file, sample_size=sample)
    if data is None or np.all(data == -32768):
        raise ValueError("No valid mosaic data!")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

    # 保存正高数据的GeoTIFF和XML
    save_geotiff(data, latlim, lonlim, ortho_tif)
    xml_filename = f"{output}.xml"
    generate_isce2_xml(xml_filename, output_file, data.shape, latlim, lonlim, egm_source, data_type='short')

    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

    # 处理椭球高转换
    ellip_tif = None
    xml_filename_ellip = None
    if height == 'ellipsoidal':
        # 转换为椭球高
        ellip_data = convert_egm_to_wgs84(data, latlim, lonlim, source=egm_source)
        ellip_tif = f"{output}_ellipsoidal.tif"
        save_geotiff(ellip_data, latlim, lonlim, ellip_tif)

        # 保存椭球高DEM文件 - 使用int16和大端序以保持与正高数据一致
        ellip_int16 = ellip_data.astype('>i2')  # 大端序 int16
        ellip_int16.tofile(output_file+'.wgs84')

        # 生成XML元数据
        xml_filename_ellip = f"{output}.wgs84.xml"
        generate_isce2_xml(xml_filename_ellip, output_file+'.wgs84', ellip_data.shape,
                           latlim, lonlim, egm_source, data_type='short')
        print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

    print(f"\n[Success] Use {egm_source.upper()} geoid")
    print(f"DEM height system: {height}")
    print("\nOutput file:")
    print(f"  Orthometric GeoTIFF: {ortho_tif}")
    print(f"  Orthometric DEM: {output_file}")
    print(f"  Orthometric XML: {xml_filename}")
    if ellip_tif:
        print(f"  Ellipsoidal GeoTIFF: {ellip_tif}")
        print(f"  Ellipsoidal DEM: {output_file+'.wgs84'}")
        print(f"  Ellipsoidal XML: {xml_filename_ellip}")
    print("* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */\n")

# ------------------- 拼接和保存 DEM 函数 -------------------


def mosaic_tiles(tiles, latlim, lonlim, output_file: str, sample_size: int = 3601):
    """根据ISCE2的逻辑拼接DEM tiles - 修复版本，正确处理字节序

    Args:
        tiles: DEM文件列表，必须按从北到南、从西到东的顺序
        latlim: [max_lat, min_lat] 纬度范围
        lonlim: [min_lon, max_lon] 经度范围
        output_file: 输出文件路径
        sample_size: 每个tile的采样数 (3601 for 1 arcsec, 1201 for 3 arcsec)

    Returns:
        (data, height, width) 元组，如果失败则返回 (None, 0, 0)
    """
    sample_size = int(sample_size)  # 确保sample_size是整数类型
    # 计算网格尺寸
    nlat = int(np.ceil(latlim[0]) - np.floor(latlim[1]))  # 纬度方向tile数
    nlon = int(np.ceil(lonlim[1]) - np.floor(lonlim[0]))  # 经度方向tile数

    print(f"Mosaicing {len(tiles)} tiles into {nlat}x{nlon} grid")

    # 🔍 调试输出: 显示文件顺序
    if len(tiles) <= 10:  # 只在tile数量较少时显示
        print("Tile order (should be north-to-south, west-to-east):")
        for i, tile in enumerate(tiles):
            print(f"  {i}: {os.path.basename(tile)}")

    # 使用C库进行拼接，模拟ISCE2的实现
    try:
        # 初始化拼接器
        stitcher = DEMStitcherPythonWrapper(_DEM_STITCHER_LIB)

        # 执行拼接
        success = stitcher.stitch_dem_tiles(
            input_files=tiles,
            output_file=output_file,
            nlat=nlat,
            nlon=nlon,
            samples_per_tile=sample_size,
            swap_bytes=True
        )

        if success and os.path.exists(output_file):
            # 读取拼接结果
            with open(output_file, 'rb') as f:
                raw_data = f.read()

            # ISCE2的concatenateDem在swap_bytes=1时会将数据转换为大端序
            data = np.frombuffer(raw_data, dtype='>i2')  # 大端序 int16

            # 计算正确的尺寸
            # DEM tile拼接时，相邻tile的边缘会重叠，所以实际尺寸是 (sample_size-1) × n
            height = (sample_size - 1) * nlat
            width = (sample_size - 1) * nlon

            if len(data) == height * width:
                data = data.reshape((height, width))
                print(f"Successfully mosaiced DEM with shape {data.shape}")
            else:
                print(f"Warning: Data size mismatch. Expected {height*width}, got {len(data)}")
                # 尝试其他可能的形状
                if len(data) % width == 0:
                    height = len(data) // width
                    data = data.reshape((height, width))
                    print(f"Reshaped to {data.shape}")
                else:
                    print(f"Error: Cannot determine proper dimensions")
                    return None, 0, 0

            return data, data.shape[0], data.shape[1]
        else:
            print("Failed to stitch tiles using C library")
            return None, 0, 0

    except Exception as e:
        print(f"Error during mosaicing: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0


class DEMStitcherPythonWrapper:
    def __init__(self, lib_path):
        """初始化C库拼接器"""
        self.lib = ctypes.CDLL(lib_path)

        # 定义函数签名
        self.lib.concatenateDem.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # filenamesIn
            ctypes.POINTER(ctypes.c_int),     # numFilesV
            ctypes.c_char_p,                  # filenameOut
            ctypes.POINTER(ctypes.c_int),     # samples
            ctypes.POINTER(ctypes.c_int)      # swap
        ]
        self.lib.concatenateDem.restype = ctypes.c_int

    def stitch_dem_tiles(self, input_files, output_file, nlat, nlon, samples_per_tile=1201, swap_bytes=False):
        """
        拼接DEM tiles

        参数:
        - input_files: 输入文件列表
        - output_file: 输出文件名
        - nlat: 纬度方向的tile数量
        - nlon: 经度方向的tile数量
        - samples_per_tile: 每个tile的样本数 (1201 for 3arcsec, 3601 for 1arcsec)
        - swap_bytes: 是否交换字节序
        """
        # 确保参数为整数类型
        nlat = int(nlat)
        nlon = int(nlon)
        samples_per_tile = int(samples_per_tile)
        swap_flag = 1 if swap_bytes else 0

        # 准备参数
        num_files = [nlat, nlon]

        # 转换为C类型
        file_list = [f.encode('utf-8') for f in input_files]  # 使用encode而不是bytes
        file_array = (ctypes.c_char_p * len(file_list))(*file_list)  # 直接初始化

        num_files_array = (ctypes.c_int * len(num_files))(*num_files)  # 直接初始化

        output_file_bytes = output_file.encode('utf-8')
        samples_c = ctypes.c_int(samples_per_tile)
        swap_c = ctypes.c_int(swap_flag)

        # 调用C函数
        result = self.lib.concatenateDem(
            file_array,
            num_files_array,
            output_file_bytes,
            ctypes.byref(samples_c),
            ctypes.byref(swap_c)
        )

        return result == 0  # 返回True表示成功


def save_geotiff(data, latlim, lonlim, filename):
    """保存为GeoTIFF格式"""
    height, width = data.shape

    # 计算地理变换参数
    pixel_width = (lonlim[1] - lonlim[0]) / width
    pixel_height = (latlim[1] - latlim[0]) / height

    # 创建GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, width, height, 1, gdal.GDT_Int16)

    # 设置地理变换
    dataset.SetGeoTransform([
        lonlim[0],  # 左上角经度
        pixel_width,  # 像素宽度
        0,  # 旋转参数
        latlim[1],  # 左上角纬度
        0,  # 旋转参数
        pixel_height  # 像素高度（负值表示从上到下）
    ])

    # 设置投影系统 (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    # 写入数据
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(-32768)

    # 设置统计信息
    band.ComputeStatistics(False)

    # 关闭数据集
    dataset = None
    print(f"Saved GeoTIFF: {filename}")


def generate_isce2_xml(xml_filename, hgt_filename, shape, latlim, lonlim, egm_source, data_type='short'):
    """生成ISCE2标准格式的XML元数据文件

    Args:
        xml_filename: 输出XML文件名
        hgt_filename: DEM数据文件名
        shape: 数据形状 (height, width)
        latlim: 纬度范围 [max_lat, min_lat]
        lonlim: 经度范围 [min_lon, max_lon]
        egm_source: EGM模型来源 ('egm96' 或 'egm2008')
        data_type: 数据类型 ('short' for int16, 'float' for float32)
    """
    height, width = shape

    # 计算坐标信息
    x_step = (lonlim[1] - lonlim[0]) / width
    y_step = (latlim[1] - latlim[0]) / height

    # 计算起始值和结束值
    x_first = lonlim[0]
    y_first = latlim[0]
    x_last = lonlim[1] - x_step
    y_last = latlim[1] - y_step

    with open(xml_filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<imageFile>\n')

        # ISCE版本信息
        f.write('    <property name="ISCE_VERSION">\n')
        f.write('        <value>Release: 2.6.3, svn-, 20230418. Current: svn-.</value>\n')
        f.write('    </property>\n')

        # 访问模式
        f.write('    <property name="access_mode">\n')
        f.write('        <value>READ</value>\n')
        f.write('        <doc>Image access mode.</doc>\n')
        f.write('    </property>\n')

        # 因为C库在swap_bytes=1时写入的是大端序数据
        f.write('    <property name="byte_order">\n')
        f.write('        <value>1</value>\n')  # 'b' for big-endian
        f.write('        <doc>Endianness of the image.</doc>\n')
        f.write('    </property>\n')

        # 第一个坐标组件 (X轴 - 经度)
        f.write('    <component name="coordinate1">\n')
        f.write('        <factorymodule>isceobj.Image</factorymodule>\n')
        f.write('        <factoryname>createCoordinate</factoryname>\n')
        f.write('        <doc>First coordinate of a 2D image (width).</doc>\n')
        f.write(f'        <property name="delta">\n')
        f.write(f'            <value>{x_step}</value>\n')
        f.write('            <doc>Coordinate quantization.</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="endingvalue">\n')
        f.write(f'            <value>{x_last}</value>\n')
        f.write('            <doc>Ending value of the coordinate.</doc>\n')
        f.write('        </property>\n')
        f.write('        <property name="family">\n')
        f.write('            <value>imagecoordinate</value>\n')
        f.write('            <doc>Instance family name</doc>\n')
        f.write('        </property>\n')
        f.write('        <property name="name">\n')
        f.write('            <value>imagecoordinate_name</value>\n')
        f.write('            <doc>Instance name</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="size">\n')
        f.write(f'            <value>{width}</value>\n')
        f.write('            <doc>Coordinate size.</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="startingvalue">\n')
        f.write(f'            <value>{x_first}</value>\n')
        f.write('            <doc>Starting value of the coordinate.</doc>\n')
        f.write('        </property>\n')
        f.write('    </component>\n')

        # 第二个坐标组件 (Y轴 - 纬度)
        f.write('    <component name="coordinate2">\n')
        f.write('        <factorymodule>isceobj.Image</factorymodule>\n')
        f.write('        <factoryname>createCoordinate</factoryname>\n')
        f.write('        <doc>Second coordinate of a 2D image (length).</doc>\n')
        f.write(f'        <property name="delta">\n')
        f.write(f'            <value>{-y_step}</value>\n')  # Y方向通常为负值
        f.write('            <doc>Coordinate quantization.</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="endingvalue">\n')
        f.write(f'            <value>{y_last}</value>\n')
        f.write('            <doc>Ending value of the coordinate.</doc>\n')
        f.write('        </property>\n')
        f.write('        <property name="family">\n')
        f.write('            <value>imagecoordinate</value>\n')
        f.write('            <doc>Instance family name</doc>\n')
        f.write('        </property>\n')
        f.write('        <property name="name">\n')
        f.write('            <value>imagecoordinate_name</value>\n')
        f.write('            <doc>Instance name</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="size">\n')
        f.write(f'            <value>{height}</value>\n')
        f.write('            <doc>Coordinate size.</doc>\n')
        f.write('        </property>\n')
        f.write(f'        <property name="startingvalue">\n')
        f.write(f'            <value>{y_first}</value>\n')
        f.write('            <doc>Starting value of the coordinate.</doc>\n')
        f.write('        </property>\n')
        f.write('    </component>\n')

        # 数据类型
        f.write('    <property name="data_type">\n')
        f.write(f'        <value>{data_type}</value>\n')
        f.write('        <doc>Image data type.</doc>\n')
        f.write('    </property>\n')

        # 额外文件名
        f.write('    <property name="extra_file_name">\n')
        f.write(f'        <value>{hgt_filename}.vrt</value>\n')
        f.write('        <doc>For example name of vrt metadata.</doc>\n')
        f.write('    </property>\n')

        # 族名称
        f.write('    <property name="family">\n')
        f.write('        <value>demimage</value>\n')
        f.write('        <doc>Instance family name</doc>\n')
        f.write('    </property>\n')

        # 文件名
        f.write('    <property name="file_name">\n')
        f.write(f'        <value>{hgt_filename}</value>\n')
        f.write('        <doc>Name of the image file.</doc>\n')
        f.write('    </property>\n')

        # 图像类型
        f.write('    <property name="image_type">\n')
        f.write('        <value>dem</value>\n')
        f.write('        <doc>Image type used for displaying.</doc>\n')
        f.write('    </property>\n')

        # 图像尺寸
        f.write('    <property name="length">\n')
        f.write(f'        <value>{height}</value>\n')
        f.write('        <doc>Image length</doc>\n')
        f.write('    </property>\n')

        # 元数据位置
        f.write('    <property name="metadata_location">\n')
        f.write(f'        <value>{xml_filename}</value>\n')
        f.write('        <doc>Location of the metadata file where the instance was defined</doc>\n')
        f.write('    </property>\n')

        # 名称
        f.write('    <property name="name">\n')
        f.write('        <value>demimage_name</value>\n')
        f.write('        <doc>Instance name</doc>\n')
        f.write('    </property>\n')

        # 波段数
        f.write('    <property name="number_bands">\n')
        f.write('        <value>1</value>\n')
        f.write('        <doc>Number of image bands.</doc>\n')
        f.write('    </property>\n')

        # 参考基准
        f.write('    <property name="reference">\n')
        f.write(f'        <value>{egm_source.upper()}</value>\n')
        f.write('        <doc>Geodetic datum</doc>\n')
        f.write('    </property>\n')

        # 存储方案
        f.write('    <property name="scheme">\n')
        f.write('        <value>BIP</value>\n')
        f.write('        <doc>Interleaving scheme of the image.</doc>\n')
        f.write('    </property>\n')

        # 宽度
        f.write('    <property name="width">\n')
        f.write(f'        <value>{width}</value>\n')
        f.write('        <doc>Image width</doc>\n')
        f.write('    </property>\n')

        # 范围值
        f.write('    <property name="xmax">\n')
        f.write(f'        <value>{x_last}</value>\n')
        f.write('        <doc>Maximum range value</doc>\n')
        f.write('    </property>\n')

        f.write('    <property name="xmin">\n')
        f.write(f'        <value>{x_first}</value>\n')
        f.write('        <doc>Minimum range value</doc>\n')
        f.write('    </property>\n')

        f.write('</imageFile>\n')

    print(f"Generated ISCE2 XML metadata: {xml_filename}")


# ------------------- 高程系统转换函数 -------------------

def convert_egm_to_wgs84(data, latlim, lonlim, source='egm96'):
    """将基于EGM的高程转换为WGS84椭球高
    Args:
        data: 输入的DEM数据数组
        latlim: 纬度范围 [max_lat, min_lat]
        lonlim: 经度范围 [min_lon, max_lon]
        source: EGM模型来源 ('egm96' 或 'egm2008')
    Returns:
        转换后的DEM数据数组
    """
    print(f"Converting heights from {source.upper()} to WGS84 ellipsoidal heights")

    # 生成经纬度网格用于插值
    height, width = data.shape
    lats = np.linspace(latlim[0], latlim[1], height)
    lons = np.linspace(lonlim[0], lonlim[1], width)

    # 获取geoid高度
    geoid_heights = get_geoid_heights(lats, lons, source)

    # 转换：椭球高 = 正高 + geoid高
    corrected_data = data.astype(np.float32)
    valid_mask = (data != -32768) & (~np.isnan(data)) & (~np.isnan(geoid_heights))
    corrected_data[valid_mask] = data[valid_mask].astype(np.float32) + geoid_heights[valid_mask]

    # 设置无效值
    corrected_data[~valid_mask] = -32768

    return corrected_data


def get_geoid_heights(lats, lons, source='egm96'):
    """获取指定位置的geoid高度
    Args:
        lats: 纬度数组
        lons: 经度数组
        source: EGM模型来源 ('egm96' 或 'egm2008')
    Returns:
        geoid高度数组
    """
    # 选择对应的EGM文件
    if source.lower() == 'egm2008':
        egm_tiff = EGM2008_TIFF
    else:  # egm96
        egm_tiff = EGM96_TIFF

    if not os.path.exists(egm_tiff):
        print(f"Warning: {source.upper()} geoid model file not found: {egm_tiff}")
        print("Using simplified approximation instead.")
        return np.zeros((len(lats), len(lons)))

    # 读取geoid模型并进行插值
    try:
        ds = gdal.Open(egm_tiff)
        if ds is None:
            print(f"Warning: Could not open {egm_tiff}, using simplified approximation")
            return np.zeros((len(lats), len(lons)))

        # 获取geoid模型的地理变换参数
        transform = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)

        # 创建输出网格
        geoid_heights = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

        # 对每个点进行插值
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # 将经纬度转换为像素坐标
                px = int((lon - transform[0]) / transform[1])
                py = int((lat - transform[3]) / transform[5])

                # 检查边界
                if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                    # 读取geoid高度
                    geoid_val = band.ReadAsArray(px, py, 1, 1)[0, 0]
                    if geoid_val != band.GetNoDataValue():
                        geoid_heights[i, j] = geoid_val

        ds = None

        # 使用双线性插值填补NaN值
        geoid_heights = interpolate_nans(geoid_heights)

        return geoid_heights

    except Exception as e:
        print(f"Warning: Error reading geoid model {egm_tiff}: {e}")
        print("Using simplified approximation instead.")
        return np.zeros((len(lats), len(lons)))


def interpolate_nans(grid):
    """使用双线性插值填补数组中的NaN值"""
    from scipy.interpolate import griddata

    height, width = grid.shape

    # 找到有效值的位置
    valid_mask = ~np.isnan(grid)
    valid_points = np.column_stack(np.where(valid_mask))
    valid_values = grid[valid_mask]

    # 如果没有有效值，返回零数组
    if len(valid_values) == 0:
        return np.zeros_like(grid)

    # 找到需要插值的位置
    nan_points = np.column_stack(np.where(~valid_mask))

    if len(nan_points) == 0:
        return grid

    # 执行插值
    interpolated_values = griddata(
        valid_points,
        valid_values,
        nan_points,
        method='linear',
        fill_value=0  # 如果无法插值，使用0填充
    )

    # 创建输出数组并填入插值结果
    result = grid.copy()
    result[tuple(nan_points.T)] = interpolated_values

    return result


# 使用示例
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(
            "Usage: python dem_stitcher_python_fixed.py LAT_MIN LAT_MAX LON_MIN LON_MAX --source srtm/copernicus [options]")
        print("\nExample:")
        print("  python dem_stitcher_python_fixed.py 34 36 116 118 --source srtm")
        print("  python dem_stitcher_python_fixed.py 34 36 116 118 --source copernicus --height ellipsoidal")
    else:
        prepare_dem()
