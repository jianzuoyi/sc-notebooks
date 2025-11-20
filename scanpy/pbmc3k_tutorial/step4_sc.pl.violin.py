"""
violin 函数源代码（来自 scanpy.plotting._anndata）

此文件包含从 official/scanpy/src/scanpy/plotting/_anndata.py 中复制的 violin 函数代码，
并添加了详细的中文注释以便理解。

注意：此文件中的代码依赖于 scanpy 包的其他模块，不能独立运行。
实际使用时，应该通过 scanpy.pl.violin() 调用此函数。

依赖说明：
- sanitize_anndata: 清理和验证 AnnData 对象
- check_use_raw: 检查是否使用 raw 数据
- get.obs_df: 获取观察值数据框
- _utils.add_colors_for_categorical_sample_annotation: 为分类注释添加颜色
- _deprecated_scale: 处理已弃用的 scale 参数
- setup_axes: 设置坐标轴
- settings: 全局设置对象
- _utils.savefig_or_show: 保存或显示图形
  * 注意：此函数使用 matplotlib 的全局状态机制（plt.gcf()）来获取当前图形
  * 它内部调用 plt.savefig()，会自动保存当前活动的图形，无需显式传入图形对象
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Sequence
from types import NoneType
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import CategoricalDtype

# 类型检查相关的导入
if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from seaborn import FacetGrid
    from .._utils import Empty
    from ._utils import DensityNorm

# 注意：以下导入在实际的 scanpy 包中可用，这里仅作说明
# 在实际的 scanpy 包中，这些导入应该是：
# from .. import get
# from .. import logging as logg
# from .._compat import old_positionals, Empty as _empty
# from .._settings import settings
# from .._utils import sanitize_anndata, check_use_raw
# from . import _utils
# from ._docs import doc_show_save_ax
# from ._utils import _deprecated_scale, setup_axes


def violin(
    adata: AnnData,
    keys: str | Sequence[str],
    groupby: str | None = None,
    *,
    log: bool = False,
    use_raw: bool | None = None,
    stripplot: bool = True,
    jitter: float | bool = True,
    size: int = 1,
    layer: str | None = None,
    density_norm: DensityNorm = "width",
    order: Sequence[str] | None = None,
    multi_panel: bool | None = None,
    xlabel: str = "",
    ylabel: str | Sequence[str] | None = None,
    rotation: float | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    # deprecated
    save: bool | str | None = None,
    scale: DensityNorm | Empty = _empty,
    **kwds,
) -> Axes | FacetGrid | None:
    """
    小提琴图绘制函数
    
    此函数是对 seaborn.violinplot 的封装，用于绘制 AnnData 对象的小提琴图。
    小提琴图可以展示数据的分布情况，包括密度分布和实际数据点。
    
    参数说明
    ----------
    adata : AnnData
        注释数据矩阵（Annotated data matrix），包含单细胞数据
        
    keys : str | Sequence[str]
        要绘制的键（keys），可以是：
        - 字符串：单个键，如 'n_genes_by_counts'
        - 字符串序列：多个键，如 ['n_genes_by_counts', 'total_counts']
        
        这些键可以来自两个地方（函数会自动识别）：
        
        1. adata.obs 中的列名（观察值注释）
           - 例如：'n_genes_by_counts', 'total_counts', 'pct_counts_mt'
           - 这些是每个细胞（observation）的元数据
           - 数据直接从 adata.obs[keys] 获取
           
        2. adata.var_names 中的变量名（通常是基因名）
           - 例如：'CD79A', 'MS4A1', 'CD8A'
           - 这些是基因的表达值
           - 数据从表达矩阵 adata.X 或 adata.raw.X 中提取
           - 对于每个细胞，提取该基因的表达值
        
        注意：函数内部使用 get.obs_df() 来获取数据，它会：
        - 首先检查 keys 是否在 adata.obs.columns 中
        - 如果不在，则检查是否在 adata.var_names 中
        - 如果在 var_names 中，从表达矩阵中提取该基因在所有细胞中的表达值
        - 最终返回一个 DataFrame，行为细胞（obs_names），列为 keys
        
    groupby : str | None, 默认 None
        用于分组的观察值键（observation key）
        - 如果为 None：不进行分组，直接绘制 keys 的分布
        - 如果指定：按照该列对数据进行分组，每个组绘制一个小提琴图
        例如：groupby='leiden' 会按照聚类结果分组绘制
        
    log : bool, 默认 False
        是否使用对数坐标轴
        - True：y 轴使用对数刻度
        - False：y 轴使用线性刻度
        
    use_raw : bool | None, 默认 None
        是否使用 adata.raw 属性
        - None：如果 adata.raw 存在则使用，否则使用 adata.X
        - True：强制使用 adata.raw
        - False：使用 adata.X
        
    stripplot : bool, 默认 True
        是否在小提琴图上叠加散点图（stripplot）
        - True：在小提琴图上显示实际数据点
        - False：只显示小提琴图，不显示数据点
        
    jitter : float | bool, 默认 True
        散点图的抖动程度（仅当 stripplot=True 时有效）
        - True：使用默认抖动
        - False：不使用抖动
        - 浮点数：指定抖动的大小（如 0.4）
        
    size : int, 默认 1
        散点图中点的大小（仅当 stripplot=True 时有效）
        
    layer : str | None, 默认 None
        要使用的 AnnData 层名称
        - None：使用默认层（根据 use_raw 决定）
        - 字符串：使用指定的层（如 'counts', 'normalized'）
        注意：layer 的优先级高于 use_raw
        
    density_norm : str, 默认 "width"
        小提琴宽度归一化方法
        - "width"：所有小提琴具有相同的宽度（默认）
        - "area"：所有小提琴具有相同的面积
        - "count"：小提琴的宽度对应于观察值的数量
        
    order : Sequence[str] | None, 默认 None
        类别显示的顺序
        - None：使用默认顺序
        - 序列：指定类别的显示顺序（如 ['A', 'B', 'C']）
        
    multi_panel : bool | None, 默认 None
        是否在多面板中显示键（即使 groupby 不为 None）
        - True：每个键在单独的面板中显示
        - False 或 None：根据情况决定是否使用多面板
        
    xlabel : str, 默认 ""
        x 轴标签
        - 如果为空字符串且 groupby 不为 None 且 rotation 为 None，
          则自动使用 groupby 作为 x 轴标签
        
    ylabel : str | Sequence[str] | None, 默认 None
        y 轴标签
        - None 且 groupby 为 None：默认为 'value'
        - None 且 groupby 不为 None：默认为 keys
        - 字符串：单个 y 轴标签
        - 序列：多个 y 轴标签（当有多个 keys 时）
        
    rotation : float | None, 默认 None
        x 轴刻度标签的旋转角度（度）
        - None：不旋转
        - 浮点数：旋转角度（如 90 表示垂直）
        
    show : bool | None, 默认 None
        是否显示图形
        - None：使用全局设置
        - True：显示图形
        - False：不显示，返回 Axes 对象
        
    ax : Axes | None, 默认 None
        matplotlib 坐标轴对象
        - None：创建新的坐标轴
        - Axes 对象：在指定的坐标轴上绘制
        
    save : bool | str | None, 默认 None（已弃用）
        是否保存图形
        - None：不保存
        - True：保存为默认文件名
        - 字符串：保存为指定文件名
        
    **kwds
        传递给 seaborn.violinplot 的其他关键字参数
        
    返回值
    -------
    Axes | FacetGrid | None
        - 如果 show=True：返回 None
        - 如果 multi_panel=True 且 groupby=None 且 len(ys)==1：返回 FacetGrid
        - 如果只有一个坐标轴：返回 Axes 对象
        - 如果有多个坐标轴：返回 Axes 对象列表
        
    示例
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> 
    >>> # 绘制单个键的小提琴图
    >>> sc.pl.violin(adata, keys='S_score')
    >>> 
    >>> # 按类别分组绘制，并旋转 x 轴标签
    >>> sc.pl.violin(adata, keys='S_score', groupby='bulk_labels', rotation=90)
    >>> 
    >>> # 绘制多个键
    >>> sc.pl.violin(adata, keys=['n_genes_by_counts', 'total_counts'], 
    ...              jitter=0.4, multi_panel=True)
    >>> 
    >>> # 不显示散点图（适用于大数据集）
    >>> sc.pl.violin(adata, keys='S_score', stripplot=False)
    """
    import seaborn as sns  # 延迟导入，因为 seaborn 导入较慢

    # 清理和验证 AnnData 对象
    sanitize_anndata(adata)
    
    # 检查是否使用 raw 数据
    use_raw = check_use_raw(adata, use_raw)
    
    # 将 keys 转换为列表格式
    if isinstance(keys, str):
        keys = [keys]
    
    # 去除重复的键，同时保持顺序
    keys = list(OrderedDict.fromkeys(keys))
    
    # 处理已弃用的 scale 参数，将其转换为 density_norm
    density_norm = _deprecated_scale(density_norm, scale, default="width")
    del scale

    # 处理 ylabel 参数
    # 如果 ylabel 是字符串或 None，根据情况扩展为列表
    if isinstance(ylabel, str | NoneType):
        # 如果没有分组，需要 1 个 y 轴标签；如果有分组，需要 len(keys) 个标签
        ylabel = [ylabel] * (1 if groupby is None else len(keys))
    
    # 验证 ylabel 的数量是否正确
    if groupby is None:
        # 没有分组时，应该只有 1 个 y 轴标签
        if len(ylabel) != 1:
            msg = f"Expected number of y-labels to be `1`, found `{len(ylabel)}`."
            raise ValueError(msg)
    else:
        # 有分组时，y 轴标签数量应该等于 keys 的数量
        if len(ylabel) != len(keys):
            msg = f"Expected number of y-labels to be `{len(keys)}`, found `{len(ylabel)}`."
            raise ValueError(msg)

    # 根据是否有分组，准备不同的数据
    if groupby is not None:
        # 有分组的情况
        # 获取包含分组列和所有 keys 的数据框
        # 
        # 注意：get.obs_df() 函数会智能识别 keys 的来源：
        # - 如果 key 在 adata.obs.columns 中，直接从 adata.obs[key] 获取（如 'n_genes_by_counts'）
        # - 如果 key 在 adata.var_names 中，从表达矩阵 adata.X 中提取该基因的表达值（如 'CD79A'）
        # - 返回的 DataFrame 行为细胞（obs_names），列为 keys
        obs_df = get.obs_df(adata, keys=[groupby, *keys], layer=layer, use_raw=use_raw)
        
        # 如果没有指定调色板，自动生成
        if kwds.get("palette") is None:
            # 检查 groupby 列是否为分类类型
            if not isinstance(adata.obs[groupby].dtype, CategoricalDtype):
                msg = (
                    f"The column `adata.obs[{groupby!r}]` needs to be categorical, "
                    f"but is of dtype {adata.obs[groupby].dtype}."
                )
                raise ValueError(msg)
            
            # 为分类注释添加颜色
            _utils.add_colors_for_categorical_sample_annotation(adata, groupby)
            
            # 设置 hue 参数（用于分组）
            kwds["hue"] = groupby
            
            # 创建调色板字典，将类别映射到颜色
            kwds["palette"] = dict(
                zip(
                    obs_df[groupby].cat.categories,
                    adata.uns[f"{groupby}_colors"],
                    strict=True,
                )
            )
    else:
        # 没有分组的情况
        # 只获取 keys 对应的数据
        # 
        # 注意：get.obs_df() 会从 adata.obs 或 adata.var_names 中获取数据
        # 返回的 DataFrame 行为细胞，列为 keys
        obs_df = get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw)
    
    # 根据是否有分组，准备不同的数据结构
    if groupby is None:
        # 没有分组：将数据从宽格式转换为长格式
        # 例如：将多个列（keys）转换为 variable-value 对
        obs_tidy = pd.melt(obs_df, value_vars=keys)
        x = "variable"  # x 轴是变量名（即 keys）
        ys = ["value"]  # y 轴是值
    else:
        # 有分组：保持原格式
        obs_tidy = obs_df
        x = groupby  # x 轴是分组列
        ys = keys  # y 轴是各个 keys

    # 多面板模式：当 multi_panel=True 且没有分组且只有一个 y 变量时
    if multi_panel and groupby is None and len(ys) == 1:
        # 这是一种快速且简单的方式来适应多个键的尺度（当 groupby 为 None 时）
        y = ys[0]

        # 使用 seaborn 的 catplot 创建多面板小提琴图
        g: sns.axisgrid.FacetGrid = sns.catplot(
            y=y,  # y 轴变量
            data=obs_tidy,  # 数据
            kind="violin",  # 图形类型为小提琴图
            density_norm=density_norm,  # 密度归一化方法
            col=x,  # 按列（variable）分面板
            col_order=keys,  # 列的顺序
            sharey=False,  # 不共享 y 轴（每个面板可以有不同尺度）
            cut=0,  # 限制小提琴图的延伸范围
            inner=None,  # 不显示内部图形（如箱线图）
            **kwds,  # 其他参数
        )

        # 如果需要，在小提琴图上叠加散点图
        if stripplot:
            # 按 x 变量分组
            grouped_df = obs_tidy.groupby(x, observed=True)
            # 为每个面板添加散点图
            for ax_id, key in zip(range(g.axes.shape[1]), keys, strict=True):
                sns.stripplot(
                    y=y,  # y 轴变量
                    data=grouped_df.get_group(key),  # 该面板对应的数据
                    jitter=jitter,  # 抖动
                    size=size,  # 点的大小
                    color="black",  # 点的颜色
                    ax=g.axes[0, ax_id],  # 目标坐标轴
                )
        
        # 如果使用对数坐标
        if log:
            g.set(yscale="log")
        
        # 设置标题和 x 轴标签
        g.set_titles(col_template="{col_name}").set_xlabels("")
        
        # 如果指定了旋转角度，旋转 x 轴标签
        if rotation is not None:
            for ax_base in g.axes[0]:
                ax_base.tick_params(axis="x", labelrotation=rotation)
    else:
        # 标准模式：单面板或多面板（有分组时）
        # 设置默认参数
        # cut=0 限制小提琴图的延伸范围（参见 stacked_violin 代码了解更多信息）
        kwds.setdefault("cut", 0)
        kwds.setdefault("inner")

        # 创建或使用坐标轴
        if ax is None:
            # 创建新的坐标轴
            # 如果没有分组，创建一个面板；如果有分组，为每个 key 创建一个面板
            axs, _, _, _ = setup_axes(
                ax,
                panels=["x"] if groupby is None else keys,  # 面板列表
                show_ticks=True,  # 显示刻度
                right_margin=0.3,  # 右边距
            )
        else:
            # 使用提供的坐标轴
            axs = [ax]
        
        # 为每个坐标轴和 y 变量绘制图形
        for ax_base, y, ylab in zip(axs, ys, ylabel, strict=True):
            # 绘制小提琴图
            sns.violinplot(
                x=x,  # x 轴变量（分组列或 variable）
                y=y,  # y 轴变量（值或 key）
                data=obs_tidy,  # 数据
                order=order,  # 类别顺序
                orient="vertical",  # 垂直方向
                density_norm=density_norm,  # 密度归一化方法
                ax=ax_base,  # 目标坐标轴
                **kwds,  # 其他参数
            )
            
            # 如果需要，在小提琴图上叠加散点图
            if stripplot:
                sns.stripplot(
                    x=x,  # x 轴变量
                    y=y,  # y 轴变量
                    data=obs_tidy,  # 数据
                    order=order,  # 类别顺序
                    jitter=jitter,  # 抖动
                    color="black",  # 点的颜色
                    size=size,  # 点的大小
                    ax=ax_base,  # 目标坐标轴
                )
            
            # 设置 x 轴标签
            # 如果 xlabel 为空且 groupby 不为 None 且 rotation 为 None，
            # 则使用 groupby 作为 x 轴标签
            if xlabel == "" and groupby is not None and rotation is None:
                xlabel = groupby.replace("_", " ")  # 将下划线替换为空格
            ax_base.set_xlabel(xlabel)
            
            # 设置 y 轴标签
            if ylab is not None:
                ax_base.set_ylabel(ylab)
            
            # 如果使用对数坐标
            if log:
                ax_base.set_yscale("log")
            
            # 如果指定了旋转角度，旋转 x 轴标签
            if rotation is not None:
                ax_base.tick_params(axis="x", labelrotation=rotation)
    
    # 决定是否显示图形
    show = settings.autoshow if show is None else show
    
    # 保存或显示图形
    # 
    # 注意：savefig_or_show 函数没有显式传入图形对象，但它仍然可以保存图形。
    # 这是因为 matplotlib 使用全局状态机制：
    # 
    # 1. 在绘制过程中（如 sns.violinplot, sns.stripplot），seaborn 和 matplotlib 
    #    会自动创建图形对象并将其设置为"当前活动的图形"（current figure）
    # 
    # 2. matplotlib 维护一个全局的图形管理器，可以通过 plt.gcf() 获取当前图形
    # 
    # 3. savefig_or_show 内部调用 _savefig，而 _savefig 使用 plt.savefig(filename)
    #    plt.savefig() 会自动保存当前活动的图形，相当于 plt.gcf().savefig(filename)
    # 
    # 4. 因此，虽然函数签名中没有图形参数，但通过 matplotlib 的全局状态机制，
    #    可以自动获取并保存当前正在绘制的图形
    # 
    # 5. 如果保存了图形，函数最后会调用 plt.close() 来清理当前图形
    # 
    # 这种设计的优点：
    # - 简化了函数调用，不需要显式传递图形对象
    # - 符合 matplotlib 的常见使用模式
    # - 在大多数情况下都能正常工作
    # 
    # 潜在问题：
    # - 如果用户手动切换了当前图形，可能会保存错误的图形
    # - 在多线程环境中可能会有问题
    _utils.savefig_or_show("violin", show=show, save=save)
    
    # 根据情况返回不同的值
    if show:
        return None
    if multi_panel and groupby is None and len(ys) == 1:
        return g  # 返回 FacetGrid
    if len(axs) == 1:
        return axs[0]  # 返回单个 Axes
    return axs  # 返回 Axes 列表


# ============================================================================
# savefig_or_show 相关函数
# ============================================================================
# 以下代码来自 official/scanpy/src/scanpy/plotting/_utils.py
# 这些函数用于保存或显示 matplotlib 图形


def _savefig(writekey, dpi=None, ext=None):
    """
    保存当前图形到文件
    
    这是 savefig_or_show 调用的内部函数，负责实际的保存操作。
    
    参数说明
    ----------
    writekey : str
        写入键（write key），用于生成文件名
        文件名格式：{settings.figdir}/{writekey}{settings.plot_suffix}.{ext}
        例如：如果 writekey="violin"，可能生成 "figures/violin.pdf"
        
    dpi : int | None, 默认 None
        保存图形的分辨率（每英寸点数）
        - None：使用 matplotlib 的默认设置（rcParams["savefig.dpi"]）
        - 如果默认 dpi < 150，会发出警告建议使用更高分辨率
        - 整数：指定具体的 dpi 值
        
    ext : str | None, 默认 None
        文件扩展名（不含点号）
        - None：使用 settings.file_format_figs（默认格式，通常是 "pdf"）
        - 字符串：指定扩展名（如 "png", "svg", "pdf"）
        
    工作原理
    ----------
    1. 检查 dpi 设置，如果分辨率过低（<150）会发出警告
    2. 创建保存目录（如果不存在）
    3. 生成完整的文件路径
    4. 调用 plt.savefig() 保存当前活动的图形
       - 使用 bbox_inches="tight" 自动裁剪空白边距
       - 这是 matplotlib 的全局函数，会自动保存当前图形（plt.gcf()）
    
    注意
    -------
    - 此函数使用 matplotlib 的全局状态机制
    - plt.savefig() 会自动保存当前活动的图形，无需显式传入图形对象
    - 通过 plt.gcf()（get current figure）获取当前图形
    """
    # 注意：以下代码需要从 scanpy 包中导入：
    # from matplotlib import rcParams
    # from matplotlib import pyplot as plt
    # from .. import logging as logg
    # from .._settings import settings
    
    if dpi is None:
        # 我们需要这个检查，因为在 notebook 中，内部图形也会受到 'savefig.dpi' 的影响
        if (
            not isinstance(rcParams["savefig.dpi"], str)
            and rcParams["savefig.dpi"] < 150
        ):
            # 如果分辨率过低，发出警告
            if settings._low_resolution_warning:
                logg.warning(
                    "You are using a low resolution (dpi<150) for saving figures.\n"
                    "Consider running `set_figure_params(dpi_save=...)`, which will "
                    "adjust `matplotlib.rcParams['savefig.dpi']`"
                )
                settings._low_resolution_warning = False
        else:
            # 使用 matplotlib 的默认 dpi 设置
            dpi = rcParams["savefig.dpi"]
    
    # 创建保存目录（如果不存在）
    # settings.figdir 是 scanpy 设置的图形保存目录
    settings.figdir.mkdir(parents=True, exist_ok=True)
    
    # 确定文件扩展名
    if ext is None:
        ext = settings.file_format_figs  # 默认格式（通常是 "pdf"）
    
    # 生成完整的文件路径
    # 格式：{figdir}/{writekey}{plot_suffix}.{ext}
    # 例如：figures/violin.pdf 或 figures/violin_20240101.pdf
    filename = settings.figdir / f"{writekey}{settings.plot_suffix}.{ext}"
    
    # 在警告级别输出消息；这对用户来说真的很重要
    logg.warning(f"saving figure to file {filename}")
    
    # 保存当前活动的图形
    # plt.savefig() 是 matplotlib 的全局函数，会自动保存当前图形（plt.gcf()）
    # bbox_inches="tight" 会自动裁剪图形周围的空白边距，使图形更紧凑
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")


def savefig_or_show(
    writekey: str,
    *,
    show: bool | None = None,
    dpi: int | None = None,
    ext: str | None = None,
    save: bool | str | None = None,
):
    """
    保存或显示图形
    
    这是 scanpy 绘图函数中常用的辅助函数，用于统一处理图形的保存和显示。
    
    参数说明
    ----------
    writekey : str
        写入键，用于生成文件名（不含扩展名）
        例如："violin" 会生成 "violin.pdf"（取决于文件格式设置）
        
    show : bool | None, 默认 None
        是否显示图形
        - None：使用全局设置（settings.autoshow）
        - True：显示图形（调用 plt.show()）
        - False：不显示图形
        
    dpi : int | None, 默认 None
        保存图形的分辨率（每英寸点数）
        - None：使用默认设置
        - 整数：指定 dpi 值
        
    ext : str | None, 默认 None
        文件扩展名（不含点号）
        - None：使用默认格式（settings.file_format_figs）
        - 字符串：指定扩展名（如 "png", "svg", "pdf"）
        
    save : bool | str | None, 默认 None
        是否保存图形（已弃用，但为了向后兼容仍支持）
        - None：使用全局设置（settings.autosave）
        - True：保存图形
        - False：不保存图形
        - 字符串：保存图形，并将字符串追加到 writekey 后面
          例如：save="-custom" 会生成 "violin-custom.pdf"
          如果字符串包含扩展名（如 ".png"），会自动提取并设置 ext
        
    工作流程
    ----------
    1. 处理 save 参数：
       - 如果 save 是字符串，检查是否包含文件扩展名
       - 如果包含扩展名，提取并设置 ext，从 save 中移除扩展名
       - 将 save 字符串追加到 writekey 后面
       - 将 save 设置为 True
       
    2. 决定是否保存：
       - 如果 save 为 None，使用 settings.autosave
       - 否则使用 save 的值
       
    3. 如果保存：
       - 如果 save 参数被使用（非 None），发出弃用警告
       - 调用 _savefig() 保存当前图形
       
    4. 决定是否显示：
       - 如果 show 为 None，使用 settings.autoshow
       - 否则使用 show 的值
       - 如果显示，调用 plt.show()
       
    5. 如果保存了图形：
       - 调用 plt.close() 清理当前图形，释放内存
       
    示例
    --------
    >>> # 只显示，不保存
    >>> savefig_or_show("violin", show=True, save=False)
    
    >>> # 只保存，不显示
    >>> savefig_or_show("violin", show=False, save=True)
    
    >>> # 保存为自定义文件名
    >>> savefig_or_show("violin", save="-custom")  # 生成 violin-custom.pdf
    
    >>> # 保存为 PNG 格式
    >>> savefig_or_show("violin", save=".png")  # 生成 violin.png
    
    >>> # 使用全局设置
    >>> savefig_or_show("violin")  # 根据 settings.autosave 和 settings.autoshow 决定
    """
    # 注意：以下代码需要从 scanpy 包中导入：
    # from matplotlib import pyplot as plt
    # from .._settings import settings
    # from .._compat import warn
    
    # 处理 save 参数为字符串的情况
    if isinstance(save, str):
        # 检查 save 字符串是否包含文件扩展名
        if ext is None:
            # 尝试从 save 字符串中提取扩展名
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    # 找到扩展名，提取并设置 ext
                    ext = try_ext[1:]  # 移除点号
                    save = save.replace(try_ext, "")  # 从 save 中移除扩展名
                    break
        
        # 将 save 字符串追加到 writekey 后面
        # 例如：writekey="violin", save="-custom" -> "violin-custom"
        writekey += save
        save = True  # 将 save 设置为 True，表示要保存
    
    # 决定是否保存图形
    # 使用海象运算符（:=）同时赋值和判断
    # 如果 save 为 None，使用全局设置 settings.autosave
    # 否则使用 save 的值
    if do_save := settings.autosave if save is None else save:
        # 如果 save 参数被使用（非 None），发出弃用警告
        if save:  # `save=True | "some-str"` 参数已被使用
            msg = (
                "Argument `save` is deprecated and will be removed in a future version. "
                "Use `sc.pl.plot(show=False).figure.savefig()` instead."
            )
            warn(msg, FutureWarning)
        
        # 调用内部函数保存当前图形
        _savefig(writekey, dpi=dpi, ext=ext)
    
    # 决定是否显示图形
    # 如果 show 为 None，使用全局设置 settings.autoshow
    # 否则使用 show 的值
    if settings.autoshow if show is None else show:
        # 显示当前活动的图形
        # plt.show() 会显示当前图形（plt.gcf()）
        plt.show()
    
    # 如果保存了图形，清理当前图形以释放内存
    if do_save:
        plt.close()  # 清除当前图形


# ============================================================================
# 关于 violin 函数使用的数据来源
# ============================================================================
"""
violin 函数使用的数据来源说明：

violin 函数可以同时使用 adata.obs 和 adata.var 的数据，具体取决于 keys 参数：

1. 如果 keys 是 adata.obs 中的列名：
   - 例如：keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
   - 这些是细胞的元数据（metadata），每个细胞有一个值
   - 数据直接从 adata.obs[keys] 获取
   - 数据形状：n_obs（细胞数）行 × len(keys) 列

2. 如果 keys 是 adata.var_names 中的基因名：
   - 例如：keys=['CD79A', 'MS4A1', 'CD8A']
   - 这些是基因的表达值，需要从表达矩阵中提取
   - 数据从 adata.X 或 adata.raw.X 中提取（取决于 use_raw 参数）
   - 对于每个基因，提取它在所有细胞中的表达值
   - 数据形状：n_obs（细胞数）行 × len(keys) 列

3. 混合使用：
   - 可以同时使用 obs 列和 var_names
   - 例如：keys=['n_genes_by_counts', 'CD79A', 'total_counts']
   - 函数会自动识别每个 key 的来源

工作原理（通过 get.obs_df() 函数）：
1. 对于每个 key，首先检查是否在 adata.obs.columns 中
2. 如果在，直接从 adata.obs[key] 获取
3. 如果不在，检查是否在 adata.var_names 中
4. 如果在 var_names 中，从表达矩阵中提取该基因的表达值
5. 最终返回一个 DataFrame，行为细胞（obs_names），列为 keys

示例：
>>> import scanpy as sc
>>> adata = sc.datasets.pbmc68k_reduced()
>>> 
>>> # 使用 obs 中的列（细胞的元数据）
>>> sc.pl.violin(adata, keys=['n_genes_by_counts', 'total_counts'])
>>> 
>>> # 使用 var_names 中的基因名（基因表达值）
>>> sc.pl.violin(adata, keys=['CD79A', 'MS4A1'])
>>> 
>>> # 混合使用
>>> sc.pl.violin(adata, keys=['n_genes_by_counts', 'CD79A'])
>>> 
>>> # 按分组绘制
>>> sc.pl.violin(adata, keys='CD79A', groupby='bulk_labels')
"""


# ============================================================================
# 关于 plt.savefig() 的说明
# ============================================================================
"""
matplotlib.pyplot.savefig() 函数说明：

plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

这是 matplotlib 的全局函数，用于保存当前活动的图形。

关键特性：
1. 自动获取当前图形：
   - plt.savefig() 内部会调用 plt.gcf()（get current figure）获取当前图形
   - 不需要显式传入图形对象
   - 相当于：plt.gcf().savefig(fname, ...)

2. 常用参数：
   - fname: 文件路径或文件对象
   - dpi: 分辨率（每英寸点数），默认使用 rcParams["savefig.dpi"]
   - bbox_inches: 边界框设置
     * None: 使用默认边界框
     * "tight": 自动裁剪空白边距，使图形更紧凑
     * Bbox 对象: 自定义边界框
   - format: 文件格式（如 "png", "pdf", "svg"），通常从文件名推断
   - facecolor, edgecolor: 图形的背景色和边框色

3. 工作原理：
   - matplotlib 维护一个全局的图形管理器
   - 当前活动的图形可以通过 plt.gcf() 获取
   - 所有绘图操作（如 plt.plot(), sns.violinplot()）都会在当前图形上操作
   - plt.savefig() 保存的就是这个当前图形

4. 示例：
   >>> import matplotlib.pyplot as plt
   >>> plt.plot([1, 2, 3], [1, 4, 9])
   >>> plt.savefig("plot.png")  # 保存当前图形
   >>> 
   >>> # 等价于：
   >>> fig = plt.gcf()  # 获取当前图形
   >>> fig.savefig("plot.png")  # 保存该图形

5. 注意事项：
   - 如果创建了多个图形，需要确保保存的是正确的图形
   - 可以使用 plt.figure() 创建新图形并切换当前图形
   - 在多线程环境中，全局状态可能会有问题
"""

