# dem_stitcher_python.py
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
    tiles = []
    for lat in range(int(np.floor(latlim[1])), int(np.ceil(latlim[0]))):
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
    tiles = []
    needed = []
    for lat in np.arange(np.floor(latlim[1]), np.ceil(latlim[0])):
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
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    latlim = [max(lat_min, lat_max), min(lat_min, lat_max)]
    lonlim = [min(lon_min, lon_max), max(lon_min, lon_max)]

    # 后续逻辑保持不变（略去重复代码）
    print(f"Region: Latitude {latlim[0]}°N–{latlim[1]}°N, Longitude {lonlim[0]}°E–{lonlim[1]}°E")
    print(f"DEM source: {source.upper()}")
    print(f"Directory: {dem_dir} {'(local only)' if local_only and source == 'copernicus' else ''}")

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

    ortho_tif = f"{output}_orthometric.tif"
    data, h, w = mosaic_tiles(tiles, latlim, lonlim, sample_size=sample)
    if np.all(data == -32768):
        raise ValueError("No valid mosaic data!")

    # 处理高程系统转换
    if height == 'ellipsoidal':
        # 转换为椭球高
        ellip_data = convert_egm_to_wgs84(data, latlim, lonlim, source=egm_source)
        ellip_tif = f"{output}_ellipsoidal.tif"
        save_geotiff(ellip_data, latlim, lonlim, ellip_tif)
        # 为Doris准备椭球高数据
        doris_data = ellip_data

        # 同时保存正高数据
        save_geotiff(data, latlim, lonlim, ortho_tif)
    else:
        # 保持正高数据
        ellip_tif = None
        save_geotiff(data, latlim, lonlim, ortho_tif)
        # 为Doris准备正高数据
        doris_data = data.astype(np.float32)

    # 生成Doris格式的DEM文件 (原始二进制格式)
    doris_filename = f"{output}.dem"
    with open(doris_filename, 'wb') as f:
        doris_data.tofile(f)

    print(f"\n[Success] Use {egm_source.upper()} geoid")
    print(f"DEM height system: {height}")
    print("\nOutput file:")
    if ellip_tif:
        print(f"  Ellipsoidal: {ellip_tif}")
    print(f"  Orthometric:   {ortho_tif}")
    print(f"  Doris RAW: {doris_filename}")


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


# ------------------- 拼接和保存 DEM 函数 -------------------

def mosaic_tiles(tiles, latlim, lonlim, sample_size=3601):
    """根据ISCE2的逻辑拼接DEM tiles"""
    # 计算网格尺寸
    nlat = int(np.ceil(latlim[0]) - np.floor(latlim[1]))  # 纬度方向tile数
    nlon = int(np.ceil(lonlim[1]) - np.floor(lonlim[0]))  # 经度方向tile数

    print(f"Mosaicing {len(tiles)} tiles into {nlat}x{nlon} grid")

    # 创建临时输出文件
    temp_output = "temp_mosaic.hgt"

    # 使用C库进行拼接，模拟ISCE2的实现
    try:
        # 初始化拼接器
        stitcher = DEMStitcherPythonWrapper(_DEM_STITCHER_LIB)

        # 执行拼接
        success = stitcher.stitch_dem_tiles(
            input_files=tiles,
            output_file=temp_output,
            nlat=nlat,
            nlon=nlon,
            samples_per_tile=sample_size,
            swap_bytes=True  # ISCE2默认进行字节序转换
        )

        if success and os.path.exists(temp_output):
            # 读取拼接结果
            with open(temp_output, 'rb') as f:
                raw_data = f.read()

            # 计算理论数据尺寸
            expected_size = (sample_size - 1) * nlat * (sample_size - 1) * nlon

            # 解析二进制数据
            data = np.frombuffer(raw_data, dtype=np.int16)

            # 按照正确的尺寸重塑
            height = (sample_size - 1) * nlat
            width = (sample_size - 1) * nlon

            if len(data) == height * width:
                data = data.reshape((height, width))
            else:
                print(f"Warning: Data size mismatch. Expected {expected_size}, got {len(data)}")
                # 尝试其他可能的形状
                if len(data) % width == 0:
                    height = len(data) // width
                    data = data.reshape((height, width))
                else:
                    print(f"Error: Cannot determine proper dimensions")
                    return None, 0, 0

            # 清理临时文件
            os.remove(temp_output)

            print(f"Successfully mosaiced DEM with shape {data.shape}")
            return data, data.shape[0], data.shape[1]
        else:
            print("Failed to stitch tiles using C library")
            return None, 0, 0

    except Exception as e:
        print(f"Error during mosaicing: {e}")
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
        # 准备参数
        num_files = [nlat, nlon]
        swap_flag = 1 if swap_bytes else 0

        # 转换为C类型
        file_list = [bytes(f, 'utf-8') for f in input_files]
        file_array = (ctypes.c_char_p * len(file_list))()
        file_array[:] = file_list

        num_files_array = (ctypes.c_int * len(num_files))()
        num_files_array[:] = num_files

        output_file_bytes = bytes(output_file, 'utf-8')
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


# ------------------- GDAL 高程系统转化函数 -------------------
def gdal_calc(expression, a_file, b_file, outfile, NoDataValue=None,
              format='GTiff', output_type='Float32', creation_options=None):

    env_root = pathlib.Path(sys.executable).resolve().parent.parent
    proj_lib = env_root / "share" / "proj"
    # 2. 复制当前 shell 环境并追加 PROJ_LIB
    my_env = os.environ.copy()
    my_env["PROJ_LIB"] = str(proj_lib)

    cmd = [
        'gdal_calc.py',
        '-A', a_file,
        '-B', b_file,
        '--outfile', outfile,
        '--calc', expression,
        '--format', format,
        '--type', output_type,
        '--overwrite'
    ]
    if NoDataValue is not None:
        cmd += ['--NoDataValue', str(NoDataValue)]
    if creation_options:
        for opt in creation_options:
            cmd += ['--co', opt]
    print(f"Run {' '.join(cmd)}")
    subprocess.run(cmd, env=my_env, capture_output=True, text=True)


def process_egm_and_doris(orthometric_tif, out_prefix, egm_source='egm96', output_height='orthometric'):
    """
    处理 EGM 基准面校正和 Doris 输出
    :param orthometric_tif: 正高 DEM 文件路径
    :param out_prefix: 输出文件前缀
    :param egm_source: 'egm96' 或 'egm2008'
    """
    ds = gdal.Open(orthometric_tif)
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    min_lon = gt[0]
    max_lat = gt[3]
    max_lon = min_lon + w * gt[1]
    min_lat = max_lat + h * gt[5]
    ds = None

    # 根据源类型选择 EGM 文件
    if egm_source == 'egm2008':
        egm_tiff = EGM2008_TIFF
        print(f"Convert using EGM2008 geoid tiff")
    else:  # egm96
        egm_tiff = EGM96_TIFF
        print(f"Converting using EGM96 geoid tiff")

    if not os.path.exists(egm_tiff):
        raise FileNotFoundError(f"EGM geoid tiff doesn't exist: {egm_tiff}")

    egm_res = f"{out_prefix}_{egm_source}_resampled.tif"
    gdal.Warp(egm_res, egm_tiff,
              outputBounds=[min_lon, min_lat, max_lon, max_lat],
              width=w, height=h, resampleAlg='bilinear',
              dstSRS='EPSG:4326', outputType=gdal.GDT_Float32,
              format='GTiff', creationOptions=['COMPRESS=LZW'])

    ortho_f32 = f"{out_prefix}_orthometric_float.tif"
    gdal.Translate(ortho_f32, orthometric_tif, outputType=gdal.GDT_Float32, noData=-32768)

    ellip_tif = f"{out_prefix}_ellipsoidal.tif"
    gdal_calc("numpy.where(A == -32768, -32768, A + B)",
              ortho_f32, egm_res, ellip_tif,
              NoDataValue=-32768, output_type='Float32',
              creation_options=['COMPRESS=LZW', 'PREDICTOR=3'])

    for f in [egm_res, ortho_f32]:
        if os.path.exists(f):
            os.remove(f)

    raw_out = f"{out_prefix}.dem"
    # var_out = f"{out_prefix}.hdr"
    tmp_tif = f"{out_prefix}_doris.tif"

    # Create Float32 GeoTIFF
    if output_height == 'ellipsoidal':
        gdal.Translate(tmp_tif, ellip_tif, outputType=gdal.GDT_Float32, noData=-32768)
    elif output_height == 'orthometric':
        gdal.Translate(tmp_tif, orthometric_tif, outputType=gdal.GDT_Float32, noData=-32768)
    else:
        raise ValueError("output_height has to be input as 'ellipsoidal' or 'orthometric'")

    # Convert to ENVI format (BSQ with .hdr header)
    envi_out = f"{out_prefix}_envi.bin"
    gdal.Translate(envi_out, tmp_tif, format="ENVI", creationOptions=['INTERLEAVE=BSQ'])

    # Rename to Doris expected format
    if os.path.exists(envi_out):
        os.rename(envi_out, raw_out)
        # Also rename the header file if needed
        if os.path.exists(f"{out_prefix}_envi.hdr"):
            os.rename(f"{out_prefix}_envi.hdr", f"{out_prefix}.hdr")
    else:
        raise FileNotFoundError("ENVI conversion failed")

    for ext in ['.img', '.blw', '.tmp', tmp_tif]:
        p = f"{out_prefix}{ext}" if ext != tmp_tif else tmp_tif
        if os.path.exists(p):
            os.remove(p)

    return ellip_tif, orthometric_tif


# 使用示例
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prepare_dem.py --source srtm/copernicus --bbox LAT_MIN LAT_MAX LON_MIN LON_MAX")
    prepare_dem()
