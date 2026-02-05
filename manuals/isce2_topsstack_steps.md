以下是整理后的 Markdown 表格，清晰呈现 InSAR 处理流程中各脚本的功能与背景：

| 脚本名称 | 主要作用 | 背景说明 |
|----------|----------|----------|
| `run_01_unpack_topo_reference` | 解压或解包参考影像所需的地形数据（DEM）及相关辅助文件，并转换为后续处理所需格式或目录结构 | InSAR 处理需预先准备地形数据（如 SRTM）用于正射校正、相位模拟等几何处理 |
| `run_02_unpack_secondary_slc` | 解压次要（Secondary）SLC 影像数据（如多时相 Sentinel-1 SLC 产品） | Sentinel-1 TOPS 模式原始数据通常为 ZIP 格式，需解压并整理至指定目录供后续读取 |
| `run_03_average_baseline` | 计算各影像对（Master 与各 Secondary）的平均空间/时间基线 | 时序 InSAR 多基线处理中需掌握每对干涉影像的轨道几何差异，用于质量筛选与形变建模 |
| `run_04_extract_burst_overlaps` | 提取参考与次要影像在相同子波束（Subswath）和 Burst 段的重叠区域 | Sentinel-1 TOPS SLC 由多个 Subswath 和 Burst 拼接而成，干涉前需精确对齐对应 Burst |
| `run_05_overlap_geo2rdr` | 对重叠区域执行 geo2rdr 几何变换（地理坐标 → 雷达坐标） | InSAR 配准关键步骤，将地理坐标系下的数据重投影至雷达成像坐标系（方位-距离向） |
| `run_06_overlap_resample` | 对重叠区域进行重采样，使参考与次要影像在雷达坐标系中严格对齐 | 配准是 InSAR 核心，仅当两景影像像素级对齐后才能生成高质量干涉相位 |
| `run_07_pairs_misreg` | 估计或修正已配准干涉对的残余配准误差（mis-registration） | 几何校正后仍可能存在微量偏移，需通过相位或相关系数分析进行迭代修正 |
| `run_08_timeseries_misreg` | 评估多时相影像间的配准误差随时间或组合的变化 | 时序 InSAR 要求所有 Secondary 影像相对于参考影像高精度对齐，并在全时序上保持一致性 |
| `run_09_fullBurst_geo2rdr` | 对完整 Burst 范围的 SLC 执行 geo2rdr 变换（非仅重叠区） | 前期仅处理重叠区用于测试配准，此步对全影像统一几何变换，为干涉做准备 |
| `run_10_fullBurst_resample` | 对完整 Burst 的 SLC 进行重采样，实现全图雷达坐标对齐 | geo2rdr 后需将次要影像插值至参考影像网格，确保干涉图相位像素一一对应 |
| `run_11_extract_stack_valid_region` | 提取多景影像堆栈中所有时相均有效的共同区域，剔除无效区（如水域、畸变区） | 时序处理前需裁剪或掩膜，保留可用于形变反演的稳定有效像素范围 |
| `run_12_merge_reference_secondary_slc` | 将配准后的参考与各次要影像合并组织至统一目录或结构 | 为批量生成干涉对做准备，便于后续自动化干涉计算流程 |
| `run_13_generate_burst_igram` | 基于配准好的 Burst 逐个生成参考与次要影像间的干涉图 | TOPS 模式通常先在 Burst 级别生成干涉图，确保局部配准精度 |
| `run_14_merge_burst_igram` | 将单个 Burst 的干涉图拼接为整景连续干涉图 | 一景 Sentinel-1 包含多个相邻 Burst，需合并以获得完整相位覆盖 |
| `run_15_filter_coherence` | 计算相干系数并进行相位滤波（如 Goldstein 滤波） | 相干性评估相位可靠性；滤波可抑制噪声，提升后续相位解缠质量 |
| `run_16_unwrap` | 对干涉相位执行相位解缠（Phase Unwrapping） | 干涉相位在 [0, 2π) 范围内模糊，需解缠展开为连续相位，才能反演地表形变或高程 |

> 参考来源：[CSDN 博客](https://blog.csdn.net/kangkangluoluo/article/details/146350600)