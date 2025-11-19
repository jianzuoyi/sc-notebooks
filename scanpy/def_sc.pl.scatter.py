"""
scanpy.pl.scatter 函数的源码（带详细中文注释）

此文件包含 scanpy.pl.scatter 函数的完整源码及其依赖的辅助函数。
所有注释均为中文，用于帮助理解函数的工作原理。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from matplotlib import patheffects, rcParams
from matplotlib.colors import is_color_like
from pandas.api.types import CategoricalDtype

from .. import get
from .. import logging as logg
from .._compat import old_positionals
from .._settings import settings
from .._utils import (
    _doc_params,
    check_use_raw,
    get_literal_vals,
    sanitize_anndata,
)
from . import _utils
from ._docs import (
    doc_scatter_basic,
    doc_show_save_ax,
)
from ._utils import (
    scatter_base,
    scatter_group,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from cycler import Cycler
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap
    from numpy.typing import NDArray

    from ._utils import (
        ColorLike,
        _FontSize,
        _FontWeight,
        _LegendLoc,
    )

# 定义可视化基础类型的字面量类型
# 支持的可视化基础包括：PCA、t-SNE、UMAP、扩散图、Force-directed 图等
type _Basis = Literal["pca", "tsne", "umap", "diffmap", "draw_graph_fr"]


@old_positionals(
    "color",
    "use_raw",
    "layers",
    "sort_order",
    "alpha",
    "basis",
    "groups",
    "components",
    "projection",
    "legend_loc",
    "legend_fontsize",
    "legend_fontweight",
    "legend_fontoutline",
    "color_map",
    # 17 个位置参数足以保持向后兼容性
)
@_doc_params(scatter_temp=doc_scatter_basic, show_save_ax=doc_show_save_ax)
def scatter(  # noqa: PLR0913
    adata: AnnData,
    x: str | None = None,
    y: str | None = None,
    *,
    color: str | ColorLike | Collection[str | ColorLike] | None = None,
    use_raw: bool | None = None,
    layers: str | Collection[str] | None = None,
    sort_order: bool = True,
    alpha: float | None = None,
    basis: _Basis | None = None,
    groups: str | Iterable[str] | None = None,
    components: str | Collection[str] | None = None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc: _LegendLoc | None = "right margin",
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight | None = None,
    legend_fontoutline: float | None = None,
    color_map: str | Colormap | None = None,
    palette: Cycler | ListedColormap | ColorLike | Sequence[ColorLike] | None = None,
    frameon: bool | None = None,
    right_margin: float | None = None,
    left_margin: float | None = None,
    size: float | None = None,
    marker: str | Sequence[str] = ".",
    title: str | Collection[str] | None = None,
    show: bool | None = None,
    ax: Axes | None = None,
    # 已弃用的参数
    save: str | bool | None = None,
) -> Axes | list[Axes] | None:
    """
    沿着观测值（observations）或变量（variables）轴绘制散点图。
    
    使用观测值注释（`.obs`）、变量注释（`.var`）或基因表达值（`.var_names`）
    来为散点图着色。

    参数
    ----------
    adata : AnnData
        带注释的数据矩阵（Annotated data matrix）
    x : str | None, optional
        x 坐标。可以是观测值列名、变量名或基因名
    y : str | None, optional
        y 坐标。可以是观测值列名、变量名或基因名
    color : str | ColorLike | Collection[str | ColorLike] | None, optional
        用于着色的键。可以是：
        - 观测值/细胞的注释键（如 'ann1'）
        - 变量/基因的注释键
        - 十六进制颜色规范（如 '#fe57a1'）
        - 上述类型的集合（如 ['ann1', 'ann2']）
    use_raw : bool | None, optional
        是否使用 `adata` 的 `raw` 属性。如果 `.raw` 存在，默认为 `True`
    layers : str | Collection[str] | None, optional
        如果 `adata` 存在 `layers` 属性，则使用它：
        为 `x`、`y` 和 `color` 指定图层。
        如果 `layers` 是字符串，则扩展为 `(layers, layers, layers)`
    sort_order : bool, default=True
        是否对数据进行排序
    alpha : float | None, optional
        点的透明度（0-1 之间）
    basis : _Basis | None, optional
        表示计算坐标的绘图工具的字符串。
        可选值：'pca', 'tsne', 'umap', 'diffmap', 'draw_graph_fr' 等
    groups : str | Iterable[str] | None, optional
        要突出显示的组名（仅用于分类注释）
    components : str | Collection[str] | None, optional
        要绘制的组件。例如，对于 PCA 可以是 '1,2' 或 ['1', '2']
    projection : Literal["2d", "3d"], default="2d"
        投影类型：'2d' 或 '3d'
    legend_loc : _LegendLoc | None, default="right margin"
        图例位置。可选值包括：
        - 'right margin': 右侧边距
        - 'on data': 在数据上
        - 'bottom right': 右下角
        - 其他 matplotlib 图例位置
    legend_fontsize : float | _FontSize | None, optional
        图例字体大小
    legend_fontweight : int | _FontWeight | None, optional
        图例字体粗细
    legend_fontoutline : float | None, optional
        图例字体轮廓宽度（用于在数据上显示图例时）
    color_map : str | Colormap | None, optional
        颜色映射名称或 Colormap 对象
    palette : Cycler | ListedColormap | ColorLike | Sequence[ColorLike] | None, optional
        用于分类数据的调色板
    frameon : bool | None, optional
        是否显示坐标轴框架
    right_margin : float | None, optional
        右侧边距（用于放置图例）
    left_margin : float | None, optional
        左侧边距
    size : float | None, optional
        点的大小。如果为 None，则根据数据点数量自动计算
    marker : str | Sequence[str], default="."
        点的标记样式
    title : str | Collection[str] | None, optional
        图的标题
    show : bool | None, optional
        是否显示图形。如果为 False，则返回 Axes 对象
    ax : Axes | None, optional
        要绘制的 matplotlib Axes 对象。如果为 None，则创建新的图形
    save : str | bool | None, optional
        （已弃用）保存图形的路径

    返回
    -------
    Axes | list[Axes] | None
        如果 `show==False`，返回 :class:`~matplotlib.axes.Axes` 或 Axes 列表。
        如果 `show==True`，返回 None（图形已显示）

    示例
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> # 使用 UMAP 坐标绘制散点图，按细胞类型着色
    >>> sc.pl.scatter(adata, basis='umap', color='bulk_labels')
    >>> # 使用自定义 x 和 y 坐标
    >>> sc.pl.scatter(adata, x='gene1', y='gene2', color='cell_type')
    """
    # color 可以是观测值列名或 matplotlib 颜色规范（或它们的集合）
    # 如果 color 不为 None，将其转换为集合形式以便统一处理
    if color is not None:
        # 如果 color 是单个字符串或颜色规范，将其转换为列表
        # 否则保持为集合形式
        color = cast(
            "Collection[str | ColorLike]",
            [color] if isinstance(color, str) or is_color_like(color) else color,
        )
    
    # 保存所有局部变量（参数）到字典中，以便传递给内部函数
    args = locals()

    # 如果指定了 basis（如 'umap', 'pca' 等），则使用预计算的坐标
    if basis is not None:
        return _scatter_obs(**args)
    
    # 如果没有指定 basis，则必须提供 x 和 y 坐标
    if x is None or y is None:
        msg = "Either provide a `basis` or `x` and `y`."
        raise ValueError(msg)
    
    # 检查 x, y 和 color 是否都是观测值（obs）的注释
    # 如果是，则调用 _scatter_obs 函数
    if _check_if_annotations(adata, "obs", x=x, y=y, colors=color, use_raw=use_raw):
        return _scatter_obs(**args)
    
    # 检查 x, y 和 color 是否都是变量（var）的注释
    # 如果是，则转置数据后调用 _scatter_obs 函数
    if _check_if_annotations(adata, "var", x=x, y=y, colors=color, use_raw=use_raw):
        # 创建转置后的参数字典
        args_t = {**args, "adata": adata.T}
        # 在转置的数据上调用 _scatter_obs
        axs = _scatter_obs(**args_t)
        # 将添加到新 adata 对象的 .uns 注释存储回原始 adata 对象
        adata.uns = args_t["adata"].uns
        return axs
    
    # 如果既不是 obs 也不是 var 的注释，则抛出错误
    msg = (
        "`x`, `y`, and potential `color` inputs must all "
        "come from either `.obs` or `.var`"
    )
    raise ValueError(msg)


def _check_if_annotations(
    adata: AnnData,
    axis_name: Literal["obs", "var"],
    *,
    x: str | None = None,
    y: str | None = None,
    colors: Collection[str | ColorLike] | None = None,
    use_raw: bool | None = None,
) -> bool:
    """
    检查 `x`、`y` 和 `colors` 是否是 `adata` 的注释。
    
    在 `colors` 的情况下，也接受有效的 matplotlib 颜色。

    如果 `axis_name` 是 `obs`，则在 `adata.obs.columns` 和 `adata.var_names` 中检查。
    如果 `axis_name` 是 `var`，则在 `adata.var.columns` 和 `adata.obs_names` 中检查。

    参数
    ----------
    adata : AnnData
        带注释的数据矩阵
    axis_name : Literal["obs", "var"]
        要检查的轴名称：'obs' 或 'var'
    x : str | None, optional
        x 坐标的键
    y : str | None, optional
        y 坐标的键
    colors : Collection[str | ColorLike] | None, optional
        颜色键的集合
    use_raw : bool | None, optional
        是否使用 raw 数据

    返回
    -------
    bool
        如果所有输入都是有效的注释，返回 True；否则返回 False
    """
    # 获取指定轴的列名（注释名）
    annotations: pd.Index[str] = getattr(adata, axis_name).columns
    
    # 根据 axis_name 和 use_raw 确定要检查的对象
    # 如果 axis_name 是 'obs' 且使用 raw，则检查 raw.var_names
    # 否则检查 adata.var_names 或 adata.obs_names
    other_ax_obj = (
        adata.raw if check_use_raw(adata, use_raw) and axis_name == "obs" else adata
    )
    names: pd.Index[str] = getattr(
        other_ax_obj, "var" if axis_name == "obs" else "obs"
    ).index

    def is_annotation(needle: pd.Index) -> NDArray[np.bool_]:
        """
        检查 needle 中的每个值是否是有效的注释。
        
        有效注释包括：
        1. None（允许为空）
        2. 在 annotations（列名）中
        3. 在 names（索引名，如基因名或细胞名）中
        """
        return needle.isin({None}) | needle.isin(annotations) | needle.isin(names)

    # 检查 x 和 y 是否都是有效注释
    if not is_annotation(pd.Index([x, y])).all():
        return False

    # 检查 colors 中的每个值
    color_idx = pd.Index(colors if colors is not None else [])
    
    # 首先检查哪些是有效的 matplotlib 颜色
    color_valid: NDArray[np.bool_] = np.fromiter(
        map(is_color_like, color_idx), dtype=np.bool_, count=len(color_idx)
    )
    
    # 对于不是有效颜色的项，检查是否是有效注释
    color_valid[~color_valid] = is_annotation(color_idx[~color_valid])
    
    # 只有当所有颜色值都有效时，才返回 True
    return bool(color_valid.all())


def _scatter_obs(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    adata: AnnData,
    x: str | None = None,
    y: str | None = None,
    color: Collection[str | ColorLike] | None = None,
    use_raw: bool | None = None,
    layers: str | Collection[str] | None = None,
    sort_order: bool = True,
    alpha: float | None = None,
    basis: _Basis | None = None,
    groups: str | Iterable[str] | None = None,
    components: str | Collection[str] | None = None,
    projection: Literal["2d", "3d"] = "2d",
    legend_loc: _LegendLoc | None = "right margin",
    legend_fontsize: float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight | None = None,
    legend_fontoutline: float | None = None,
    color_map: str | Colormap | None = None,
    palette: Cycler | ListedColormap | ColorLike | Sequence[ColorLike] | None = None,
    frameon: bool | None = None,
    right_margin: float | None = None,
    left_margin: float | None = None,
    size: float | None = None,
    marker: str | Sequence[str] = ".",
    title: str | Collection[str] | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    ax: Axes | None = None,
) -> Axes | list[Axes] | None:
    """
    在观测值（observations）上绘制散点图的核心函数。
    
    这是 scatter 函数的主要实现，处理所有实际的绘图逻辑。

    参数
    ----------
    参数与 scatter 函数相同（见 scatter 函数的文档字符串）

    返回
    -------
    Axes | list[Axes] | None
        如果 `show==False`，返回 Axes 对象或 Axes 列表
    """
    # 清理和验证 AnnData 对象
    sanitize_anndata(adata)

    # 确定是否使用 raw 数据
    use_raw = check_use_raw(adata, use_raw)

    # ========== 处理 layers 参数 ==========
    # layers 用于指定从哪个数据层读取 x、y 和 color 数据
    # 如果 layers 是 "X" 或 None，或者是一个有效的层名，则扩展为三元组
    if layers in ["X", None] or (isinstance(layers, str) and layers in adata.layers):
        layers = (layers, layers, layers)  # (x_layer, y_layer, color_layer)
    # 如果 layers 是包含 3 个元素的集合，则转换为元组
    elif isinstance(layers, Collection) and len(layers) == 3:
        layers = tuple(layers)
        # 验证每个层是否有效
        for layer in layers:
            if layer not in adata.layers and layer not in ["X", None]:
                msg = (
                    "`layers` should have elements that are "
                    "either None or in adata.layers.keys()."
                )
                raise ValueError(msg)
    else:
        # layers 格式不正确
        msg = (
            "`layers` should be a string or a collection of strings "
            f"with length 3, had value '{layers}'"
        )
        raise ValueError(msg)
    
    # 如果使用 raw 数据，则不能同时使用 layers（除了 "X" 或 None）
    if use_raw and layers not in [("X", "X", "X"), (None, None, None)]:
        msg = "`use_raw` must be `False` if layers are used."
        raise ValueError(msg)

    # ========== 验证 legend_loc 参数 ==========
    if legend_loc not in (valid_legend_locs := get_literal_vals(_utils._LegendLoc)):
        msg = f"Invalid `legend_loc`, need to be one of: {valid_legend_locs}."
        raise ValueError(msg)
    
    # ========== 处理 components 参数 ==========
    # components 指定要绘制哪些组件（例如 PCA 的前两个主成分）
    if components is None:
        # 根据投影类型设置默认组件
        components = "1,2" if "2d" in projection else "1,2,3"
    # 如果 components 是字符串，则按逗号分割
    if isinstance(components, str):
        components = components.split(",")
    # 将组件索引转换为整数（从 0 开始，所以减 1）
    components = np.array(components).astype(int) - 1
    
    # ========== 处理 color 键 ==========
    # 如果没有指定 color，则使用灰色
    keys = ["grey"] if color is None else color
    
    # ========== 处理 title 参数 ==========
    # 如果 title 是字符串，则转换为列表
    if title is not None and isinstance(title, str):
        title = [title]
    
    # ========== 获取高亮数据 ==========
    # 从 adata.uns 中获取高亮数据（如果有）
    highlights = adata.uns.get("highlights", [])
    
    # ========== 获取坐标数据 ==========
    if basis is not None:
        # 如果指定了 basis（如 'umap', 'pca' 等），从 obsm 中获取预计算的坐标
        try:
            # 对于扩散图（diffmap），需要忽略第 0 个扩散分量
            if basis == "diffmap":
                components += 1
            # 从 obsm 中获取坐标数据（例如 'X_umap', 'X_pca' 等）
            xy = adata.obsm["X_" + basis][:, components]
            # 修正组件向量以便用于标签等
            if basis == "diffmap":
                components -= 1
        except KeyError:
            msg = f"compute coordinates using visualization tool {basis} first"
            raise KeyError(msg) from None
    elif x is not None and y is not None:
        # 如果没有指定 basis，则从 x 和 y 参数中获取坐标
        if use_raw:
            # 如果使用 raw 数据
            # 检查 x 是否在 obs 列中，如果是则从 obs 获取，否则从 raw 获取
            if x in adata.obs.columns:
                x_arr = adata.obs_vector(x)
            else:
                x_arr = adata.raw.obs_vector(x)
            # 对 y 做同样的处理
            if y in adata.obs.columns:
                y_arr = adata.obs_vector(y)
            else:
                y_arr = adata.raw.obs_vector(y)
        else:
            # 不使用 raw 数据，从指定的层获取数据
            x_arr = adata.obs_vector(x, layer=layers[0])
            y_arr = adata.obs_vector(y, layer=layers[1])
        # 将 x 和 y 数组合并为一个 N×2 的数组
        xy = np.c_[x_arr, y_arr]
    else:
        # 既没有 basis 也没有 x 和 y，抛出错误
        msg = "Either provide a `basis` or `x` and `y`."
        raise ValueError(msg)

    # ========== 计算点的大小 ==========
    if size is None:
        n = xy.shape[0]  # 数据点数量
        # 根据数据点数量自动计算点的大小
        # 公式：120000 / n，确保点的大小随数据量自动调整
        size = 120000 / n

    # ========== 设置图例字体大小 ==========
    if legend_fontsize is None:
        # 使用 matplotlib 的默认图例字体大小
        legend_fontsize = rcParams["legend.fontsize"]

    # ========== 处理调色板 ==========
    # 如果 palette 是序列且第一个元素不是颜色，则将其视为多个调色板
    if isinstance(palette, Sequence) and not isinstance(palette, str):
        palettes = palette if not is_color_like(palette[0]) else [palette]
    else:
        # 否则，为每个 key 使用相同的调色板
        palettes = [palette for _ in range(len(keys))]
    # 为每个调色板设置默认值
    palettes = [_utils.default_palette(palette) for palette in palettes]

    # ========== 确定组件名称和轴标签 ==========
    if basis is not None:
        # 根据 basis 类型确定组件名称（用于轴标签）
        component_name = (
            "DC" if basis == "diffmap"  # Diffusion Component
            else "tSNE" if basis == "tsne"
            else "UMAP" if basis == "umap"
            else "PC" if basis == "pca"  # Principal Component
            else "TriMap" if basis == "trimap"
            else basis.replace("draw_graph_", "").upper() if "draw_graph" in basis
            else basis
        )
    else:
        component_name = None
    
    # 如果没有组件名称，则使用 x 和 y 作为轴标签
    axis_labels = (x, y) if component_name is None else None
    # 只有在没有组件名称时才显示刻度
    show_ticks = component_name is None

    # ========== 生成颜色数据 ==========
    # 为每个 color key 生成颜色数组或颜色值
    color_ids: list[np.ndarray | ColorLike] = []
    categoricals = []  # 记录哪些 key 是分类数据
    colorbars = []  # 记录哪些 key 需要显示颜色条
    
    for ikey, key in enumerate(keys):
        c = "white"  # 默认颜色
        categorical = False  # 默认假设是连续数据或单一颜色
        colorbar = None
        
        # ========== 确定颜色数据的来源 ==========
        # 测试是分类注释还是连续注释
        if key in adata.obs:
            # key 在观测值列中
            if isinstance(adata.obs[key].dtype, CategoricalDtype):
                # 如果是分类数据类型，标记为分类
                categorical = True
            else:
                # 否则是连续数据，获取数值数组
                c = adata.obs[key].to_numpy()
        # 根据基因表达值着色
        elif use_raw and adata.raw is not None and key in adata.raw.var_names:
            # 从 raw 数据中获取基因表达值
            c = adata.raw.obs_vector(key)
        elif key in adata.var_names:
            # 从指定层中获取基因表达值
            c = adata.obs_vector(key, layer=layers[2])
        elif is_color_like(key):  # 单一颜色
            # key 本身就是一个颜色值
            c = key
            colorbar = False  # 单一颜色不需要颜色条
        else:
            # key 无效，抛出错误
            msg = (
                f"key {key!r} is invalid! pass valid observation annotation, "
                f"one of {adata.obs.columns.tolist()} or a gene name {adata.var_names}"
            )
            raise ValueError(msg)
        
        # 如果 colorbar 尚未确定，则根据是否为分类数据来决定
        if colorbar is None:
            colorbar = not categorical
        
        colorbars.append(colorbar)
        if categorical:
            categoricals.append(ikey)  # 记录分类数据的索引
        color_ids.append(c)  # 保存颜色数据

    # ========== 设置边距 ==========
    # 如果有分类数据且图例在右侧边距，则设置右侧边距
    if right_margin is None and len(categoricals) > 0 and legend_loc == "right margin":
        right_margin = 0.5
    
    # ========== 设置标题 ==========
    if title is None and keys[0] is not None:
        # 如果没有指定标题，则使用 key 的名称作为标题
        # 将下划线替换为空格，如果是颜色则使用空字符串
        title = [
            key.replace("_", " ") if not is_color_like(key) else "" for key in keys
        ]

    # ========== 调用底层绘图函数 ==========
    # scatter_base 是实际的绘图函数，处理所有绘图细节
    axs: list[Axes] = scatter_base(
        xy,  # 坐标数据
        title=title,
        alpha=alpha,
        component_name=component_name,  # 组件名称（如 'UMAP', 'PC' 等）
        axis_labels=axis_labels,  # 轴标签
        component_indexnames=components + 1,  # 组件索引名称（从 1 开始）
        projection=projection,  # 投影类型（'2d' 或 '3d'）
        colors=color_ids,  # 颜色数据列表
        highlights=highlights,  # 高亮数据
        colorbars=colorbars,  # 是否需要颜色条
        right_margin=right_margin,
        left_margin=left_margin,
        sizes=[size for _ in keys],  # 点的大小列表
        markers=marker,  # 标记样式
        color_map=color_map,  # 颜色映射
        show_ticks=show_ticks,  # 是否显示刻度
        ax=ax,  # 目标 Axes 对象
    )

    def add_centroid(centroids, name, xy, mask) -> None:
        """
        添加分类数据的质心位置。
        
        质心用于在图例位置为 'on data' 时在数据上显示标签。

        参数
        ----------
        centroids : dict
            存储质心位置的字典
        name : str
            分类名称
        xy : np.ndarray
            坐标数组
        mask : np.ndarray
            布尔掩码，指示哪些点属于该分类
        """
        xy_mask = xy[mask]  # 获取属于该分类的点
        if xy_mask.shape[0] == 0:
            return  # 如果没有点，则返回
        # 计算中位数位置
        median = np.median(xy_mask, axis=0)
        # 找到最接近中位数的点
        i = np.argmin(np.sum(np.abs(xy_mask - median), axis=1))
        # 将该点的位置作为质心
        centroids[name] = xy_mask[i]

    # ========== 处理分类数据 ==========
    # 遍历所有分类注释并绘制它们
    for ikey, pal in zip(categoricals, palettes, strict=False):
        key = keys[ikey]  # 获取分类数据的 key
        
        # 为分类样本注释添加颜色
        _utils.add_colors_for_categorical_sample_annotation(
            adata, key, palette=pal, force_update_colors=palette is not None
        )
        
        # 实际绘制各个组
        mask_remaining = np.ones(xy.shape[0], dtype=bool)  # 剩余未绘制的点的掩码
        centroids = {}  # 存储每个组的质心位置
        
        if groups is None:
            # 如果没有指定 groups，则绘制所有组
            for iname, name in enumerate(adata.obs[key].cat.categories):
                # 跳过在设置中标记为忽略的类别
                if name not in settings.categories_to_ignore:
                    # 绘制该组的数据点
                    mask = scatter_group(
                        axs[ikey],  # 目标 Axes
                        key,  # 分类 key
                        iname,  # 类别索引
                        adata,  # AnnData 对象
                        xy,  # 坐标
                        projection=projection,  # 投影类型
                        size=size,  # 点的大小
                        alpha=alpha,  # 透明度
                        marker=marker,  # 标记样式
                    )
                    mask_remaining[mask] = False  # 标记这些点已绘制
                    # 如果图例位置在数据上，则计算质心
                    if legend_loc.startswith("on data"):
                        add_centroid(centroids, name, xy, mask)
        else:
            # 如果指定了 groups，则只绘制指定的组
            groups = [groups] if isinstance(groups, str) else groups
            for name in groups:
                # 验证组名是否有效
                if name not in set(adata.obs[key].cat.categories):
                    msg = (
                        f"{name!r} is invalid! specify valid name, "
                        f"one of {adata.obs[key].cat.categories}"
                    )
                    raise ValueError(msg)
                else:
                    # 找到该组的索引
                    iname = np.flatnonzero(
                        adata.obs[key].cat.categories.values == name
                    )[0]
                    # 绘制该组的数据点
                    mask = scatter_group(
                        axs[ikey],
                        key,
                        iname,
                        adata,
                        xy,
                        projection=projection,
                        size=size,
                        alpha=alpha,
                        marker=marker,
                    )
                    # 如果图例位置在数据上，则计算质心
                    if legend_loc.startswith("on data"):
                        add_centroid(centroids, name, xy, mask)
                    mask_remaining[mask] = False  # 标记这些点已绘制
        
        # ========== 绘制剩余的点 ==========
        # 绘制未在 groups 中指定的点（用浅灰色）
        if mask_remaining.sum() > 0:
            data = [xy[mask_remaining, 0], xy[mask_remaining, 1]]
            if projection == "3d":
                data.append(xy[mask_remaining, 2])
            axs[ikey].scatter(
                *data,
                marker=marker,
                c="lightgrey",  # 浅灰色
                s=size,
                edgecolors="none",
                zorder=-1,  # 放在最底层
            )
        
        # ========== 添加图例 ==========
        legend = None
        if legend_loc.startswith("on data"):
            # 如果图例位置在数据上，则在每个组的质心位置添加文本标签
            if legend_fontweight is None:
                legend_fontweight = "bold"  # 默认使用粗体
            
            # 如果指定了字体轮廓，则添加轮廓效果
            if legend_fontoutline is not None:
                path_effect = [
                    patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
                ]
            else:
                path_effect = None
            
            # 在每个组的质心位置添加文本标签
            for name, pos in centroids.items():
                axs[ikey].text(
                    pos[0],  # x 坐标
                    pos[1],  # y 坐标
                    name,  # 标签文本
                    weight=legend_fontweight,  # 字体粗细
                    verticalalignment="center",  # 垂直对齐
                    horizontalalignment="center",  # 水平对齐
                    fontsize=legend_fontsize,  # 字体大小
                    path_effects=path_effect,  # 路径效果（轮廓）
                )

            # 如果图例位置是 'on data export'，则导出标签位置到 CSV 文件
            all_pos = np.zeros((len(adata.obs[key].cat.categories), 2))
            for iname, name in enumerate(adata.obs[key].cat.categories):
                all_pos[iname] = centroids.get(name, [np.nan, np.nan])
            if legend_loc == "on data export":
                filename = settings.writedir / "pos.csv"
                logg.warning(f"exporting label positions to {filename}")
                settings.writedir.mkdir(parents=True, exist_ok=True)
                np.savetxt(filename, all_pos, delimiter=",")
        elif legend_loc == "right margin":
            # 如果图例在右侧边距，则创建标准图例
            legend = axs[ikey].legend(
                frameon=False,  # 无框架
                loc="center left",  # 位置：左侧居中
                bbox_to_anchor=(1, 0.5),  # 锚点位置
                ncol=(
                    1 if len(adata.obs[key].cat.categories) <= 14  # 类别数 <= 14，单列
                    else 2 if len(adata.obs[key].cat.categories) <= 30  # 类别数 <= 30，两列
                    else 3  # 类别数 > 30，三列
                ),
                fontsize=legend_fontsize,
            )
        elif legend_loc != "none":
            # 其他图例位置（如 'bottom right', 'upper left' 等）
            legend = axs[ikey].legend(
                frameon=False, loc=legend_loc, fontsize=legend_fontsize
            )
        
        # 设置图例中标记的大小
        if legend is not None:
            for handle in legend.legend_handles:
                handle.set_sizes([300.0])

    # ========== 绘制坐标轴框架 ==========
    # frameon 控制是否显示坐标轴框架
    frameon = settings._frameon if frameon is None else frameon
    if not frameon and x is None and y is None:
        # 如果没有指定 x 和 y（即使用了 basis），则隐藏坐标轴
        for ax_ in axs:
            ax_.set_xlabel("")
            ax_.set_ylabel("")
            ax_.set_frame_on(False)

    # ========== 显示或保存图形 ==========
    show = settings.autoshow if show is None else show
    _utils.savefig_or_show("scatter" if basis is None else basis, show=show, save=save)
    
    # 根据 show 参数决定返回值
    if show:
        return None  # 如果显示了图形，则返回 None
    if len(keys) > 1:
        return axs  # 如果有多个 key，返回 Axes 列表
    return axs[0]  # 如果只有一个 key，返回单个 Axes 对象

