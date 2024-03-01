import os
import re
import pickle
import json
import torch
import glob
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


# Create custom legend for the horizontal lines
line_items = [
    Line2D([0], [0], color="red", linestyle="--", linewidth=1, alpha=0.7),
    Line2D([0], [0], color="cyan", linestyle="--", linewidth=1, alpha=0.7),
    Line2D([0], [0], color="yellow", linestyle="--", linewidth=1, alpha=0.7),
]

# Color/Name Maps
method_color_map = {
    "MI": "#A40122",
    "MIa": "#E20134",
    "MIc": "#E20134",
    "MI_D": "#C761E7",
    "MI_I": "#FFB2FD",
    "MI_DI": "#8400CD",
    "GE_D": "#FF6E3A",
    "GE": "#D4522A",
    "LE": "#FFC33B",
    "LE_I": "#ffe4a8",
    "-LE_I": "#FF5AAF",
    "-LE": "#FF8AD5",
    "-LEc": "#FF8AD5",
    "ELP": "#0070D1",
    "EPM": "#008DF9",
    "EMV": "#00C2F9",
    "-PPL": "#009F81",
    "-PPL_I": "#56ccb6",
    "PPL": "#1e6e34",
    "PPL_I": "#5f966f",
    "BEST": "grey",
    "MEAN": "grey",
    "WORST": "grey",
    "BEST_line": "red",
    "MEAN_line": "cyan",
    "WORST_line": "yellow",
    "BEST_line_filter": "red",
    "MEAN_line_filter": "#31d4d4",
    "WORST_line_filter": "#ff8c00",
}

task_color_map = {
    "imdb": "#8400CD",
    "sst2": "#C761E7",
    "agnews": "#FF5AAF",
    "glue-rte": "#FFB2FD",
    "newspop": "#7A3C3C",
    "tweet-irony": "#A40122",
    "tweet-emotion": "#E20134",
    "cb": "#FF6E3A",
    "sst5": "#FFC33B",
    "copa": "#192bc2",
    "piqa": "#00C2F9",
    "storycloze": "#009F81",
    "hellaswag": "#20bf55",
}

model_marker_map = {
    "gpt-neo": ".",
    "opt": "*",
    "gpt": "<",
    "bloom": ">",
    "gpt-j": "o",
}

method_marker_map = {
    "GE_D": "x",
    "LE": "x",
    "-LE_I": "x",
    "EPM": "x",
    "GE": "^",
    "MI_DI": "D",
    "MI_D": "D",
    "MI_I": "D",
    "MIa": "D",
    "MIc": "s",
    "-LEc": "s",
}


def get_model_marker(model):
    return model_marker_map[re.search(r"^([a-z\-]+)", model).group(1)]


def get_model_size(model):
    return np.log(float(re.search(r"(\d+(\.\d+)*)", model).group(1))) * 10


def latex_text(base, sub=None, sup=None):
    text = base
    if sub:
        text += f"$_\mathrm{{{sub}}}$"
    if sup:
        text += f"$^\mathrm{{{sup}}}$"
    return text


combn_hatch_map = {
    "CC": "xxx",
    "CN": "\\\\\\",
    "NC": "///",
    "NN": None,
}

method_hatch_map = defaultdict(lambda: None)
method_hatch_map["MIc"] = "xxx"
method_hatch_map["-LEc"] = "xxx"
method_hatch_map["BEST"] = "*"
method_hatch_map["MEAN"] = "o"
method_hatch_map["WORST"] = "..."

method_rename_map = {
    "MI": "MI",
    "MIc": latex_text("MI", "A", "(PA)"),
    "MIa": latex_text("MI", "A"),
    "MI_I": latex_text("MI", "AL"),
    "MI_D": latex_text("MI", "AG"),
    "MI_DI": latex_text("MI", "AGL"),
    "GE_D": "GE",
    "GE": latex_text("GE", "M"),
    "LE": "LE",
    "LE_I": latex_text("LE", "L"),
    "-LE_I": "MDL",
    "-LE": latex_text("MDL", "M"),
    "-LEc": latex_text("MDL", "M", "(PA)"),
    "ELP": "ZLP",
    "EPM": "ZPM",
    "EMV": "ZMV",
    "-PPL": "PPL",
    "-PPL_I": latex_text("PPL", "L"),
    "PPL": latex_text("PPL", "N"),
    "PPL_I": latex_text("PPL", "NL"),
    "BEST": "best prompt",
    "MEAN": "average of prompts",
    "WORST": "worst prompt",
}
task_rename_map = {
    "imdb": "imdb",
    "sst2": "g-sst2",
    "agnews": "agnews",
    "glue-rte": "g-rte",
    "newspop": "newspop",
    "tweet-irony": "t-irony",
    "tweet-emotion": "t-emo",
    "cb": "sg-cb",
    "sst5": "sst5",
    "copa": "copa",
    "piqa": "piqa",
    "storycloze": "story",
    "hellaswag": "hella",
    "U_copa": "copa w/ label bias",
    "U_piqa": "piqa w/ label bias",
    "U_storycloze": "story w/ label bias",
    "U_hellaswag": "hella w/ label bias",
    "avg": "avg",
    "balanced": "balanced",
    "unbalanced": "unbalanced",
    "dynamic": "dynamic",
    "udynamic": "dynamic w/ label bias",
    "task average": "task average",
}
setup_rename_map = {
    "CC": "calibration for Prompt selection & Answer selection (PA)",
    "CN": "calibration for Prompt selection (P)",
    "NC": "calibration for Answer selection (A)",
    "NN": "no calibration",
    "v1": "$|v|=1$",
    "v2": "$|v|=2$",
    "v12": "$1≤|v|≤2$",
    "few": "few-shot",
    "ours": "CBM",
    "cbu": "CC",
    "pmi": latex_text("PMI", "DC"),
    "mean": "mean",
    "softmax": "softmax",
    "acc": "Accuracy of Selected Prompt",
    "f1": "F1 of Selected Prompt",
    "acc_to_best": "Scaled Accuracy of Selected Prompt",
    "f1_to_best": "Scaled F1 of Selected Prompt",
    "acc_corr": "Corr(Selected Prompt Accuracy, Prompt Selection Score)",
    "f1_corr": "Corr(Selected Prompt F1, Prompt Selection Score)",
    "acc_BEST": "Accuracy of the Best Prompt",
    "f1_BEST": "F1 of the Best Prompt",
    "task": "Evaluation Dataset",
    "model": "Model",
    "category": "Evaluation Dataset Category",
    "method": "Prompt Selection Method",
    "acc_better": "Ratio of Prompts whose Accuracy Improved Through Calibration",
    "f1_better": "Ratio of Prompts whose F1 Improved Through Calibration",
    "anywhere": "all cases",
    "front": "$x$ before prompt",
    "middle": "$x$ in the middle",
    "end": "$x$ after prompt",
}
model_rename_map = {
    "gpt-neo1.3b": "GPT-Neo 1.3B",
    "opt1.3b": "OPT 1.3B",
    "gpt1.5b": "GPT2-XL 1.5B",
    "gpt-neo2.7b": "GPT-Neo 2.7B",
    "opt2.7b": "OPT 2.7B",
    "bloom3b": "BLOOM 3B",
    "gpt-j6b": "GPT-J 6B",
    "opt6.7b": "OPT 6.7B",
    "opt30b": "OPT 30B",
    "opt66b": "OPT 66B",
}
rename_map = method_rename_map | task_rename_map | setup_rename_map | model_rename_map


def get_combn_method_name(method, combn):
    simple_combn_rename_map = {
        "CC": "(PA)",
        "CN": "(P)",
        "NC": "(A)",
    }
    return (
        latex_text(rename_map[method], "", simple_combn_rename_map[combn])
        if combn != "NN"
        else rename_map[method]
    )


# Orders
base_method_order = ["MI", "GE_D", "LE", "-LE_I", "ELP", "EPM", "EMV", "-PPL"]
nomi_base_method_order = ["MIa", "GE_D", "LE", "-LE_I", "ELP", "EPM", "EMV", "-PPL"]
line_order = ["BEST", "MEAN", "WORST"]

token_method_order = ["MIa", "MI", "GE_D", "LE", "-LE_I", "ELP", "EPM", "EMV", "-PPL"]

transfer_method_order = [
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "MI",
    "GE_D",
    "GE",
    "LE",
    "-LE_I",
    "-LE",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]  # '-PPL_I', 'PPL', 'PPL_I']
nomi_transfer_method_order = [
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "GE_D",
    "GE",
    "LE",
    "-LE_I",
    "-LE",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]  # '-PPL_I', 'PPL', 'PPL_I']

corr_base_method_order = ["MI", "GE_D", "LE", "ELP", "EPM", "EMV", "-PPL"]
corr_transfer_method_order = [
    "MI_D",
    "MIa",
    "MI",
    "GE_D",
    "GE",
    "LE",
    "-LE",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]  # '-PPL_I', 'PPL', 'PPL_I']
nomi_corr_transfer_method_order = [
    "MI_D",
    "MIa",
    "GE_D",
    "GE",
    "LE",
    "-LE",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]  # '-PPL_I', 'PPL', 'PPL_I']

main_method_order = [
    "BEST",
    "MEAN",
    "WORST",
    "-LEc",
    "MIc",
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "MI",
    "GE_D",
    "LE",
    "-LE_I",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]
nomi_main_method_order = [
    "BEST",
    "MEAN",
    "WORST",
    "-LEc",
    "MIc",
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "GE_D",
    "LE",
    "-LE_I",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]

newmain_method_order = [
    "-LEc",
    "MIc",
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "MI",
    "GE_D",
    "LE",
    "-LE_I",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]
nomi_newmain_method_order = [
    "-LEc",
    "MIc",
    "MI_DI",
    "MI_D",
    "MI_I",
    "MIa",
    "GE_D",
    "LE",
    "-LE_I",
    "ELP",
    "EPM",
    "EMV",
    "-PPL",
]

setup_order = ["v1", "v12", "v2", "few"]

task_order = [
    "imdb",
    "sst2",
    "agnews",
    "glue-rte",
    "newspop",
    "tweet-irony",
    "tweet-emotion",
    "cb",
    "sst5",
    "copa",
    "piqa",
    "storycloze",
    "hellaswag",
]
task_order_rename = [
    "imdb",
    "sst2",
    "agnews",
    "rte",
    "newspop",
    "irony",
    "emotion",
    "cb",
    "sst5",
    "copa",
    "piqa",
    "story",
    "hella",
]

cls_task_order = [
    "imdb",
    "sst2",
    "agnews",
    "glue-rte",
    "newspop",
    "tweet-irony",
    "tweet-emotion",
    "cb",
    "sst5",
]
cls_task_order_rename = [
    "imdb",
    "sst2",
    "agnews",
    "rte",
    "newspop",
    "irony",
    "emotion",
    "cb",
    "sst5",
]

dyn_task_order = ["copa", "piqa", "storycloze", "hellaswag"]
udyn_task_order = ["U_copa", "U_piqa", "U_storycloze", "U_hellaswag"]
dyn_task_order_rename = ["copa", "piqa", "story", "hella"]

category_order = ["balanced", "unbalanced", "dynamic"]
cls_category_order = ["balanced", "unbalanced"]

combn_order = ["CC", "CN", "NC", "NN"]
cali_type_order = ["ours", "pmi", "cbu"]

porder = ["acc", "f1"]
ppl_task_order = [
    "imdb",
    "sst2",
    "agnews",
    "tweet-irony",
    "tweet-emotion",
    "sst5",
    "copa",
    "piqa",
    "hellaswag",
]

model_order = [
    "gpt-neo1.3b",
    "opt1.3b",
    "gpt1.5b",
    "gpt-neo2.7b",
    "opt2.7b",
    "bloom3b",
    "gpt-j6b",
    "opt6.7b",
    "opt30b",
    "opt66b",
]

filter_type_order = ["all", "filtered"]


# Utils
def filter_dataframe(
    df, options={}, filters=[], no_udynamic=False, only_nn=False, only_opt=False
):
    # Create a boolean mask with all True values (same shape as the DataFrame)
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)

    # Iterate through the options and update the mask based on the key-value pairs
    for key, value in options.items():
        if isinstance(value, (tuple, list)):
            mask = df[key].isin(value)
        else:
            mask = mask & (df[key] == value)

    for filter in filters:
        mask = mask & filter

    if no_udynamic:
        mask = mask & (df["category"] != "udynamic")

    if only_nn:
        mask = mask & (df["combn"] == "NN")

    if only_opt:
        mask = mask & (df["model"] == "opt2.7b")

    # Apply the mask to the DataFrame and return the result
    return df[mask]


def get_dict_key(option):
    return "__".join([f"{key}={value}" for key, value in option.items()])


def get_ps_dict_filename(**option):
    return get_dict_key(option) + ".pkl"


def get_ps_dict_filepath(**option):
    save_dir = "../ps_dicts"
    filepath = os.path.join(save_dir, get_ps_dict_filename(**option))
    print(filepath)
    return filepath


def fill_missing_ops(ops_df, ps_df):
    ps_df = ps_df[ps_df["model"].isin(ops_df["model"].unique())]

    ops_tasks = set(ops_df["task"].unique())
    ps_tasks = set(ps_df["task"].unique())
    to_add_tasks = ps_tasks - ops_tasks

    if to_add_tasks:
        rows = ps_df[
            ps_df["task"].isin(to_add_tasks) & (ps_df["method"] == "MIa")
        ].copy()
        rows.loc[:, "method"] = "MI"
        return pd.concat([ops_df, rows])

    return ops_df


def read_ps_dict(option):
    with open(get_ps_dict_filepath(**option), "rb") as f:
        ps_dict = pickle.load(f)

    # rule-based add-hoc exception handling: copy the balanced and unbalanced tasks of ps_df to ops_df
    if option["setup"] == "v1" and ps_dict["ops_df"] is not None:
        ps_dict["ops_df"] = fill_missing_ops(ps_dict["ops_df"], ps_dict["ps_df"])

    for dtype in ps_dict:
        for key, value in option.items():
            if ps_dict[dtype] is not None and key not in ps_dict[dtype]:
                if key == "do_filter":
                    ps_dict[dtype].loc[:, key] = "filtered" if value else "all"
                else:
                    ps_dict[dtype].loc[:, key] = value

    return ps_dict


def ps_combine_and_filter_model(ps_df, ips_df, base_df, ops_df=None):
    if ips_df is not None:
        assert set(ps_df["model"].unique()) == set(ips_df["model"].unique())
    comb_ps_df = pd.concat(
        [ps_df, ips_df, ops_df, base_df]
    )  # warning: this is base_df of only ps_df (not ops_df)

    if ops_df is not None:
        models = ops_df["model"].unique().tolist()
        comb_ps_df = comb_ps_df[comb_ps_df["model"].isin(models)].reset_index(drop=True)

    return comb_ps_df


def add_mic(ps_df):
    ps_df.loc[(ps_df["method"] == "MIa") & (ps_df["combn"] == "CC"), "method"] = "MIc"
    ps_df.loc[(ps_df["method"] == "MIc"), "combn"] = "NN"

    ps_df.loc[(ps_df["method"] == "-LE") & (ps_df["combn"] == "CC"), "method"] = "-LEc"
    ps_df.loc[(ps_df["method"] == "-LEc"), "combn"] = "NN"

    ps_df = filter_dataframe(ps_df, {"combn": "NN"})
    return ps_df.drop("combn", axis=1)


def do_add_task_average(df):
    groupby = df.select_dtypes(["object"]).columns.tolist()
    groupby.remove("task")
    groupby.remove("category")

    rows = []
    for name, subdf in df.groupby(groupby):
        row = {key: val for key, val in zip(groupby, name)} | {
            "task": "avg",
            "category": "task average",
        }
        s1 = pd.Series(row)
        s2 = subdf.mean(axis=0, numeric_only=True)
        rows.append(pd.concat([s1, s2]))

    return pd.concat([df, pd.DataFrame(rows)])


def leave_only_common_models(df):
    text_columns = df.select_dtypes(["object"]).columns.tolist()
    text_columns.remove("model")

    common_models = None
    for _, subdf in df.groupby(text_columns):
        models = set(subdf["model"].unique())
        common_models = common_models or models  # init
        commom_models = common_models.intersection(models)
    df = df[df["model"].isin(common_models)].reset_index(drop=True)
    print(common_models)
    return df


def get_for_baseline(
    ps_dict,
    method_order=None,
    task_order=None,
    no_udynamic=True,
    add_task_average=True,
    only_nn=True,
    filter={},
):
    ps_dict = ps_dict.copy()

    if "MI" not in method_order:
        ps_dict["ops_df"] = None

    df = ps_combine_and_filter_model(**ps_dict)

    df = leave_only_common_models(df)

    if no_udynamic:
        df = filter_dataframe(df, no_udynamic=True)

    if only_nn:
        df = filter_dataframe(df, only_nn=True)

    if "MIc" in method_order:
        df = add_mic(df)

    if method_order is not None:
        df = df[df["method"].isin(method_order)].reset_index(drop=True)

    if task_order is not None:
        df = df[df["task"].isin(task_order)].reset_index(drop=True)

    if filter:
        df = filter_dataframe(df, filter)

    if add_task_average:
        df = do_add_task_average(df)

    return df


def merge_columns(df, columns):
    if "," in columns:
        real_columns = columns.split(",")
        df[columns] = df.apply(
            lambda row: "_".join(str(row[c]) for c in real_columns), axis=1
        )
    return df


# generate latex table
def generate_latex_table(matrix):
    # Transpose the matrix if the columns are multi-indexed
    # if isinstance(matrix.index, pd.MultiIndex):
    #     matrix = matrix.T

    num_index_levels = matrix.index.nlevels
    num_columns = matrix.columns.nlevels + len(matrix.columns)

    column_format = "c" * (num_index_levels + num_columns)

    latex_table = matrix.to_latex(
        multirow=True,
        multicolumn=True,
        column_format=column_format,
        escape=False,
        bold_rows=False,
    )

    # Center-align multirows and multicolumns vertically
    latex_table = latex_table.replace("\\multirow{", "\\multirow[c]{")

    # Define a function to replace the matched multicolumn pattern
    def replace_multicolumn_center(match):
        return "\\multicolumn{{{}}}{{c}}{{ {} }}".format(match.group(1), match.group(2))

    # Replace \multicolumn with centered version
    latex_table = re.sub(
        r"\\multicolumn{(\d+)}{l}{([^{}]+)}", replace_multicolumn_center, latex_table
    )

    # Remove vertical lines and replace the beginning of the table
    latex_table = latex_table.replace("|", "")
    latex_table = latex_table.replace(
        "\\begin{tabular}",
        "\\begin{table*}[ht!]\n\\centering \\small\n\\caption{CAPTION}\n\\label{tab:LABEL_PLACEHOLDER}\n\\begin{tabular}",
    )

    # Add closing tag for table* environment
    latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}\n\\end{table*}")

    BACKSLASH = "BACKSLASH"

    def escape_latex(s):
        return s.replace("\\", BACKSLASH)

    def unescape_latex(s):
        return s.replace(BACKSLASH, "\\")

    # Boldface the column names
    for column_level in range(len(matrix.columns.names)):
        for col_name in sorted(
            matrix.columns.get_level_values(column_level).unique(), key=lambda x: len(x)
        ):
            escaped_col_name = escape_latex(col_name)
            pattern = re.compile(r"\b" + re.escape(escaped_col_name) + r"\b")
            latex_table = pattern.sub(
                "\\\\textbf{" + escaped_col_name + "}", latex_table
            )

    # Boldface the index names
    for index_level in range(len(matrix.index.names)):
        if index_level > 0:
            for idx_name in sorted(
                matrix.index.get_level_values(index_level).unique(),
                key=lambda x: len(x),
            ):
                escaped_idx_name = escape_latex(idx_name)
                pattern = re.compile(r"\b" + re.escape(escaped_idx_name) + r"\b")
                latex_table = pattern.sub(
                    "\\\\textbf{" + escaped_idx_name + "}", latex_table
                )

    # Remove the innermost index name from the LaTeX table
    if isinstance(matrix.index, pd.MultiIndex):
        inner_index_names = matrix.index.names[1:]
        for inner_index_name in inner_index_names:
            latex_table = latex_table.replace(
                "\\textbf{" + inner_index_name + "}", ""
            ).replace(inner_index_name, "")
    else:
        index_name = matrix.index.name
        latex_table = latex_table.replace("\\textbf{" + index_name + "}", "").replace(
            index_name, ""
        )

    # Identify the empty row (search for it)
    empty_row_pattern = r"\s*((&\s*|{\s*}\s*){2,}\\\\)"
    empty_row_search = re.search(empty_row_pattern, latex_table)

    if empty_row_search:
        # Remove the empty row
        latex_table = (
            latex_table[: empty_row_search.start()]
            + latex_table[empty_row_search.end() :]
        )

    return latex_table


# get matrix
def get_matrix(
    df,
    value,
    index="category",
    columns="method",
    index_order=None,
    column_order=None,
    *args,
    **kwargs,
):
    df = merge_columns(df, index)
    df = merge_columns(df, columns)

    num_models = 0
    for name, subdf in df.groupby([index, columns]):
        _num_models = len(subdf["model"].unique())
        num_models = num_models or _num_models  # init
        assert num_models == _num_models

    matrix = df.groupby([index, columns]).mean(numeric_only=True)[value].unstack()

    if column_order:
        matrix = matrix[column_order]
    if index_order:
        matrix = matrix.transpose()[index_order].transpose()

    return matrix


def highlight_numbers(matrix, compare_along="index"):
    if compare_along == "index":
        axis = 0
    else:
        axis = 1

    top1 = matrix.apply(lambda x: max([i for i in x if type(i) != str]), axis=axis)
    top2 = matrix.apply(
        lambda x: sorted([i for i in x if type(i) != str])[-2], axis=axis
    )

    fmatrix = matrix.copy()
    for j, col in enumerate(matrix.columns):
        for i, idx in enumerate(matrix.index):
            val = matrix.loc[idx, col]
            if type(val) != str:
                if val == (top1[col] if compare_along == "index" else top1[idx]):
                    rep = f"\\textbf{{{val:.4f}}}"
                elif val == (top2[col] if compare_along == "index" else top2[idx]):
                    rep = f"\\underline{{{val:.4f}}}"
                else:
                    rep = f"{val:.4f}"
                val = rep
            fmatrix.loc[idx, col] = val
    return fmatrix


def get_formatted_matrix(
    df,
    value,
    index="category",
    columns="method",
    index_order=category_order,
    column_order=base_method_order,
    hightlight=False,
    *args,
    **kwargs,
):
    matrix = get_matrix(df, value, index, columns, index_order, column_order)

    if column_order:
        col_mapper = {col: rename_map[col] for col in column_order}
        matrix = matrix.rename(col_mapper, axis=1)

    if index_order:
        index_mapper = {i: rename_map.get(i, i) for i in index_order}
        matrix = matrix.rename(index_mapper, axis=0)

    if hightlight:
        if index == "method":
            matrix = highlight_numbers(matrix, compare_along="index")
        elif columns == "method":
            matrix = highlight_numbers(matrix, compare_along="columns")

    return matrix


def get_metric_filename(**option):
    metric_type = option.pop("metric_type")
    prefix = {
        "metric": "",
        "ometric": "o",
        "imetric": "i",
    }[metric_type]

    return get_dict_key(option) + f".{prefix}metric.csv"


def get_line_matrix_dict(
    df,
    pname,
    get_matrix_fn=get_matrix,
    index="category",
    columns="method",
    index_order=None,
    column_order=None,
):
    matrix_dict = dict()
    for method in ["BEST", "MEAN", "WORST"]:
        matrix_dict[method] = get_matrix_fn(
            df, f"{pname}_{method}", index, columns, index_order, column_order
        )
    return matrix_dict


def get_add_lines_fn(color_fn, bar_width, method_axis, add_lines):
    def add_lines_fn(line_dict, ax, idx, col, bar_x):
        for line in line_order:
            color_col = line + "_line" if method_axis == "columns" else col
            color_idx = idx if method_axis == "columns" else line + "_line"

            ax.hlines(
                line_dict[line].loc[idx, col],
                bar_x - (bar_width * (1 / 3)),
                bar_x + (bar_width * (1 / 3)),
                colors=color_fn(color_idx, color_col),
                linewidth=1,
                alpha=0.7,
            )

    if add_lines:
        return add_lines_fn
    return lambda line_dict, ax, idx, col, bar_x: None


def merge_ps_dicts(*dicts):
    new_dict = defaultdict(list)
    for key in ["ps_df", "ips_df", "ops_df", "base_df"]:
        for ps_dict in dicts:
            if ps_dict[key] is not None:
                new_dict[key].append(ps_dict[key])

        if new_dict[key]:
            new_dict[key] = pd.concat(new_dict[key])
        else:
            new_dict[key] = None
    return new_dict


def read_ps_dicts(options):
    dicts = []
    for option in options:
        dicts.append(read_ps_dict(option))
    return merge_ps_dicts(*dicts)


# Simple bar
def draw_simple_bar(
    df,
    pname,
    bar_width,
    legend_width,
    index_order,
    column_order,
    index,
    columns,
    method_axis,
    color_fn,
    hatch_fn,
    xlabel,
    title,
    add_lines,
    figsize,
    line_dict=None,
    add_lines_fn=None,
    matrix=None,
    get_matrix_fn=get_matrix,
    rename_map=rename_map,
    ylim=(0, 0.8),
    figname=None,
    legend_rows=2,
    no_legend=False,
    title_suffix="",
    special=[],
    num_columns=5,
    std_matrix=None,
    std_line_dict=None,
):
    if special:
        start = special[0]
        end = special[1]

    ptitle = rename_map.get(pname, pname)
    # Filter the DataFrames
    if matrix is None:
        matrix_options = dict(
            index=index,
            columns=columns,
            index_order=index_order,
            column_order=column_order,
        )
        add_lines_fn = get_add_lines_fn(color_fn, bar_width, method_axis, add_lines)

        matrix = get_matrix_fn(df, pname, **matrix_options)
        line_dict = (
            get_line_matrix_dict(
                df, pname, get_matrix_fn=get_matrix_fn, **matrix_options
            )
            if add_lines
            else None
        )
    else:
        if add_lines:
            assert add_lines_fn is not None, "pass add_lines_fn as input"
            assert line_dict is not None, "pass line_dict as input"
        matrix = matrix[column_order].transpose()[index_order].transpose()

    num_categories = len(matrix.index)
    num_bars_per_category = len(matrix.columns)

    # Create the bar plot and customize the colors and patterns
    fig, ax = plt.subplots(figsize=figsize)
    num_categories = len(matrix.index)
    bars_per_category = len(matrix.columns)

    for i, idx in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            bar_x = i + j * bar_width
            bar_height = matrix.loc[idx, col]
            color = color_fn(idx, col)
            hatch = hatch_fn(idx, col)
            if special:
                if col == start:
                    x = bar_x - bar_width / 2
                    coords = [(x, bar_height + 0.03), (x, bar_height + 0.04)]
                elif col == end:
                    x = bar_x - bar_width / 2
                    coords.extend([(x, coords[1][1]), (x, coords[0][1])])
                    ax.plot(*zip(*coords), color="black")
                    ax.annotate(
                        "our work",
                        ((coords[0][0] + coords[-1][0]) / 2, coords[1][1] + 0.01),
                        ha="center",
                    )

            ax.bar(
                bar_x,
                bar_height,
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
                alpha=1.0,
                label=col if i == 0 else None,
            )
            if std_matrix is not None:
                ax.errorbar(
                    bar_x,
                    bar_height,
                    yerr=std_matrix.loc[idx, col],
                    fmt="-",
                    color="black",
                )
            if add_lines:
                if std_line_dict:
                    add_lines_fn(line_dict, std_line_dict, ax, idx, col, bar_x)
                else:
                    add_lines_fn(line_dict, ax, idx, col, bar_x)

    ax.set_xticks(np.arange(num_categories) + bars_per_category / 2 * bar_width)
    if len(index_order) == 1:
        ax.set_xticklabels([])
    elif index_order:
        ax.set_xticklabels([rename_map.get(i, i) for i in index_order], rotation=0)
    else:
        ax.set_xticklabels([i for i in matrix.index], rotation=90)

    ax.xaxis.grid(False)
    plt.ylabel(ptitle)

    plt.xlabel(rename_map.get(xlabel, xlabel))
    plt.title(rename_map.get(title, title) + title_suffix)

    if ylim:
        ax.set_ylim(ylim)

    legend_items = ax.get_legend_handles_labels()[0]
    legend_labels = [rename_map.get(m, m) for m in ax.get_legend_handles_labels()[1]]
    if add_lines:
        legend_items += line_items
        legend_labels += [rename_map.get(m, m) for m in line_order]

    # if len(method_order) < 7:  # combined legend
    #   ax.legend(
    #       legend_items,
    #       legend_labels,
    #       loc='center left', bbox_to_anchor=(1, 0.5))

    if figname:
        filename = f"{figname}_{pname}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(filename)
    plt.show()

    if not no_legend:  # separate legend
        fig_leg = plt.figure(figsize=(legend_width, 0.1))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")

        leg = ax_leg.legend(
            legend_items, legend_labels, loc="center", ncol=num_columns, mode="expand"
        )

        if figname:
            filename = f"{figname}_legend.pdf"
            fig_leg.savefig(filename, format="pdf", bbox_inches="tight")
            print(filename)
        plt.show()


def draw_bar(
    df,
    bar_width,
    legend_width,
    index_order,
    column_order,
    index,
    columns,
    method_axis,
    color_fn,
    hatch_fn,
    xlabel,
    title,
    add_lines,
    pnames,
    figsize,
    rename_map=rename_map,
    ylim=(0, 0.8),
    figname=None,
    no_legend=False,
    title_suffix="",
):
    for pname in pnames:
        draw_simple_bar(
            df=df,
            pname=pname,
            bar_width=bar_width,
            legend_width=legend_width,
            index_order=index_order,
            column_order=column_order,
            index=index,
            columns=columns,
            method_axis=method_axis,
            color_fn=color_fn,
            hatch_fn=hatch_fn,
            xlabel=xlabel,
            title=title,
            title_suffix=title_suffix,
            add_lines=add_lines,
            figsize=figsize,
            matrix=None,
            add_lines_fn=None,
            ylim=ylim,
            figname=figname,
            no_legend=no_legend,
        )


def get_matrix_for_table(
    df,
    pnames,
    concat_axis=0,
    transpose=True,
    index="category",
    columns="method",
    index_order=category_order,
    column_order=base_method_order,
    compare_along="index",
    *args,
    **kwargs,
):
    df_dict = dict()
    print(index_order)
    print(column_order)
    for pname in pnames:
        df_dict[rename_map[pname]] = get_formatted_matrix(
            df,
            pname,
            index,
            columns,
            index_order,
            column_order,
            compare_along=compare_along,
            hightlight=True,
        )
    matrix = pd.concat(df_dict, axis=concat_axis)

    if transpose:
        return matrix.transpose()
    return matrix


def assert_same_set(df, metric, column):
    assert set(df[column].unique()) == set(
        metric[column].unique()
    ), f"{df[column].unique()} != {metric[column].unique()}"


def draw_stripplot(
    metric,
    df=None,
    method_order=base_method_order,
    index="task",
    index_order=task_order,
    figsize=(6.4, 5.4),
    annot_size=10,
    pnames=["acc", "f1"],
    markersize=5,
    legend_width=12,
    alpha=0.7,
    jitter=0,
    xlabel="",
    ylabel="",
    title="",
    figname="",
    no_udynamic=True,
    *args,
    **kwargs,
):
    for pname in pnames:
        fig, ax = plt.subplots(figsize=figsize)

        sns.stripplot(
            x=metric[index],
            y=metric[pname],
            order=index_order,
            color="lightgray",
            linewidth=0,
            alpha=0.3,
            zorder=1,
        )

        if df is not None:
            for i, idx in enumerate(index_order):
                for method in method_order:
                    subdf = filter_dataframe(
                        df, {"method": method, index: idx}, no_udynamic=no_udynamic
                    )

                    if index != "model":
                        assert_same_set(subdf, metric, "model")
                    if index != "task":
                        assert_same_set(subdf, metric, "task")
                    if "combn" in subdf.columns:
                        assert_same_set(subdf, metric, "combn")

                    x = i
                    if jitter:
                        x += np.random.uniform(-jitter, jitter)
                    ax.plot(
                        x,
                        subdf[pname].mean(),
                        color=method_color_map[method],
                        marker=method_marker_map.get(method, "o"),
                        markersize=markersize,
                        linestyle="None",
                        alpha=alpha,
                        label=method if i == 0 else None,
                        zorder=2,
                    )

        ax.set_xticklabels([rename_map[i] for i in index_order], rotation=0)

        ax.xaxis.grid(False)
        ax.set_xlabel(rename_map.get(xlabel, xlabel))
        if pname == "acc":
            ax.set_ylabel("Accuracy of Prompt")
        elif pname == "f1":
            ax.set_ylabel("F1 of Prompt")
        plt.title(title)

        if len(method_order) < 10:
            ax.legend(
                ax.get_legend_handles_labels()[0],
                [rename_map[m] for m in ax.get_legend_handles_labels()[1]],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        if figname:
            filename = f"{figname}_{pname}.pdf"
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            print(filename)
        plt.show()

    if not len(method_order) < 10:
        legend_items = ax.get_legend_handles_labels()[0]
        legend_labels = [rename_map[m] for m in ax.get_legend_handles_labels()[1]]

        fig_leg = plt.figure(figsize=(legend_width, 0.1))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")

        num_columns = len(legend_items) // 2 + len(legend_items) % 2
        leg = ax_leg.legend(
            legend_items, legend_labels, loc="center", ncol=num_columns, mode="expand"
        )

        if figname:
            filename = f"{figname}_legend.pdf"
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            print(filename)

    plt.show()


def preprocess_template_df(df):
    remap = {
        "ag_news": "agnews",
        "glue/sst2": "sst2",
        "imdb": "imdb",
        "newspop": "newspop",
        "glue/rte": "glue-rte",
        "super_glue/cb": "cb",
        "super_glue/copa": "copa",
        "hellaswag": "hellaswag",
        "tweet_eval/emotion": "tweet-emotion",
        "tweet_eval/irony": "tweet-irony",
        "SetFit/sst5": "sst5",
        "piqa": "piqa",
        "story_cloze/2016": "storycloze",
    }
    df["task"] = df["task"].apply(lambda x: remap[x])

    dfs = []
    for task in ["piqa", "copa", "hellaswag", "storycloze"]:
        newdf = df[df["task"] == task].copy()
        newdf.loc[:, "task"] = "U_" + task
        dfs.append(newdf)
    df = pd.concat([df] + dfs)
    return df


def get_metric_filename(**option):
    metric_type = option.pop("metric_type")
    prefix = {
        "metric": "",
        "ometric": "o",
        "imetric": "i",
    }[metric_type]

    return get_dict_key(option) + f".{prefix}metric.csv"


def get_metric_filepath(**option):
    save_dir = "../metrics"
    return os.path.join(save_dir, get_metric_filename(**option))


def safe_read_csv(filepath):
    if os.path.exists(filepath):
        ...
    return pd.read_csv(filepath)
    # return None


def preprocess_metric(metric, token="all"):
    if metric is None:
        return None

    if "Unnamed: 0" in metric.columns:
        metric = metric.drop("Unnamed: 0", axis=1)

    for col in ["LE", "PPL"]:
        if col in metric.columns:
            metric[f"-{col}"] = -metric[col]

    if token == "all":
        metric = metric.rename(columns={"MI": "MIa"})

    metric["category"] = np.where(
        metric["task"].isin(["cb", "tweet-emotion", "newspop", "tweet-irony", "sst5"]),
        "unbalanced",
        np.where(
            metric["task"].isin(["glue-rte", "agnews", "imdb", "sst2"]),
            "balanced",
            np.where(
                metric["task"].str.startswith("U_"),
                "udynamic",
                np.where(
                    metric["task"].isin(["piqa", "storycloze", "hellaswag", "copa"]),
                    "dynamic",
                    "other",
                ),
            ),
        ),
    )
    return metric


def preprocess_ometric(ometric):
    if ometric is None:
        return None

    if "Unnamed: 0" in ometric.columns:
        ometric = ometric.drop("Unnamed: 0", axis=1)
    ometric["task"] = ometric["task"].str.replace("OU_", "U_")
    ometric["task"] = ometric["task"].str.replace("O_", "")
    ometric.columns = [col.replace("_O", "") for col in ometric.columns]

    ometric = preprocess_metric(ometric, token="first")
    return ometric


def preprocess_prompt_column(df, setup):
    if setup in ["v12", "v2"]:
        df["prompt"] = df["prompt"].apply(
            lambda x: x.replace("verbalizer12", "prompt").replace(
                "verbalizer2", "prompt"
            )
        )
    elif setup in ["few"]:
        df["prompt"] = df["prompt"].apply(
            lambda x: x.replace("fewshot__", "")
            .replace("shot__set", "_")
            .replace("__permute", "_")
        )
    return df


def get_template_df(setup):
    if setup in ["few"]:
        filepath = "../all_templates/all_templates_fewshot.csv"
    else:
        filepath = "../all_templates/all_templates.csv"

    df = pd.read_csv(filepath)
    df = df.rename({"prompt_name": "prompt"}, axis=1)
    df = preprocess_template_df(df)
    return df


def get_metric(
    metric_type, setup, cali_type, cali_norm_type, do_filter, add_template_info=False
):

    df = safe_read_csv(
        get_metric_filepath(
            metric_type=metric_type,
            setup=setup,
            cali_type=cali_type,
            cali_norm_type=cali_norm_type,
            do_filter=do_filter,
        )
    )
    if df is None:
        return None

    df = preprocess_prompt_column(df, setup)

    if metric_type == "ometric":
        process_fn = preprocess_ometric
    else:
        process_fn = preprocess_metric

    df = process_fn(df)

    if add_template_info:
        template_df = get_template_df(setup)
        df = df.merge(
            template_df[["task", "prompt", "template", "answer_choices"]],
            on=["task", "prompt"],
            how="left",
        )

    df.loc[:, "setup"] = setup
    df.loc[:, "cali_type"] = cali_type
    df.loc[:, "cali_norm_type"] = cali_norm_type
    df.loc[:, "do_filter"] = do_filter

    return df


def get_metric_dict(option):
    metric_dict = dict()
    for metric_type in ["metric", "imetric", "ometric"]:
        metric_dict[metric_type] = get_metric(metric_type, **option)
    return metric_dict


def get_metric_dict_few(option):
    metric_dict = dict()
    for metric_type in ["metric", "imetric"]:
        metric_dict[metric_type] = get_metric(metric_type, **option)
    return metric_dict


def draw_simple_heatmap(
    matrix,
    pname,
    pval_matrix=None,
    significance_level=0.05,
    index="task",
    columns="model",
    higher_better=True,
    index_order=task_order,
    column_order=model_order,
    figsize=(6.4, 5.4),
    annot_size=10,
    xlabel="",
    ylabel="",
    title="",
    title_suffix="",
    figname="",
    rename_map=rename_map,
    vmin_vmax=None,
    already_formatted=False,
    cmap=None,
    axhlines=[],
    axvlines=[],
    *args,
    **kwargs,
):
    if not cmap:
        if higher_better:
            cmap = "RdBu"
        else:
            cmap = "RdBu_r"

    if not figsize:
        x_size = (8 / 11) * len(matrix.columns)
        y_size = (5.5 / 14) * len(matrix.index)
        figsize = (x_size, y_size)

    plt.figure(figsize=figsize)

    if rename_map and not already_formatted:
        matrix = matrix.rename(columns=rename_map, index=rename_map)

    vmin, vmax = None, None
    if vmin_vmax:
        vmin, vmax = vmin_vmax

    # Create a new matrix with formatted numbers and stars for significant p-values
    if pval_matrix is not None:
        annot_matrix = matrix.copy()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                p_value = pval_matrix.iloc[i, j]
                star = "$^*$" if p_value <= significance_level else ""
                annot_matrix.iloc[i, j] = f"{value:.2f}{star}"
    else:
        annot_matrix = matrix.applymap(lambda x: f"{x:.2f}")

    ax = sns.heatmap(
        matrix,
        annot=annot_matrix,
        cmap=cmap,
        annot_kws={"size": annot_size},
        vmin=vmin,
        vmax=vmax,
        fmt="",
    )

    plt.xlabel(rename_map.get(xlabel, xlabel))
    plt.ylabel(rename_map.get(ylabel, ylabel))
    if title == True:
        plt.title(rename_map.get(pname, pname) + title_suffix)
    else:
        plt.title(rename_map.get(title, title) + title_suffix)

    for line in axhlines:
        ax.axhline(y=line, linewidth=2, color="white")

    for line in axvlines:
        ax.axvline(x=line, linewidth=2, color="white")

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    if figname:
        filename = f"{figname}_{pname}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(filename)
    plt.show()


def draw_heatmap(
    df,
    index="task",
    columns="model",
    higher_better=True,
    index_order=task_order,
    column_order=model_order,
    figsize=(6.4, 5.4),
    annot_size=10,
    pnames=["acc", "f1"],
    xlabel="",
    ylabel="",
    title="",
    title_suffix="",
    figname="",
    vmin_vmax=None,
    *args,
    **kwargs,
):
    for pname in pnames:
        assert not ("task" not in [index, columns] and "avg" in df["task"])

        matrix = get_formatted_matrix(
            df=df,
            value=pname,
            index=index,
            columns=columns,
            index_order=index_order,
            column_order=column_order,
        )
        if "_corr" in pname:
            pval_matrix = get_formatted_matrix(
                df=df,
                value=pname.replace("_corr", "_pval"),
                index=index,
                columns=columns,
                index_order=index_order,
                column_order=column_order,
            )
        else:
            pval_matrix = None

        draw_simple_heatmap(
            matrix=matrix,
            pval_matrix=pval_matrix,
            pname=pname,
            index=index,
            columns=columns,
            higher_better=higher_better,
            index_order=index_order,
            column_order=column_order,
            figsize=figsize,
            annot_size=annot_size,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            title_suffix=title_suffix,
            figname=figname,
            vmin_vmax=vmin_vmax,
            already_formatted=True,
            axhlines=[],
            axvlines=[],
            *args,
            **kwargs,
        )


def draw_simple_scatter(
    metric,
    x_column,
    y_columns,
    title="",
    xtitle=None,
    figsize=None,
    rename_map=None,
    show_corr=True,
    figname=None,
    alpha=0.5,
):
    if rename_map is None:
        rename_map = {}

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(y_columns))

    print(len(metric[x_column]))

    for i, y_column in enumerate(y_columns):
        sns.scatterplot(
            data=metric,
            x=x_column,
            y=y_column,
            color=colors[i],
            label=rename_map.get(y_column, y_column),
            alpha=alpha,
            legend=False,
        )
        sns.regplot(data=metric, x=x_column, y=y_column, scatter=False, color=colors[i])

        # Find the index of the maximum x_column value for the current y_column
        max_x_idx = metric[x_column].idxmax()

        # Highlight the point with the maximum x_column value
        sns.scatterplot(
            data=metric.loc[[max_x_idx]],
            x=x_column,
            y=y_column,
            color=colors[i],
            marker="X",
            s=100,
            legend=False,
        )

        if "prompt" in metric:
            # Get the x and y values of the point you want to annotate
            x_value = metric.loc[max_x_idx, x_column]
            y_value = metric.loc[max_x_idx, y_column]

            # Add the annotation to the plot
            plt.text(
                x_value,
                y_value,
                f"{metric.loc[max_x_idx, 'prompt']} ({y_value:.4f})",
                fontsize=8,
                horizontalalignment="right",
                verticalalignment="bottom",
            )

    if len(y_columns) == 2:
        plt.ylabel("Accuracy, F1 of Prompt")
    else:
        plt.ylabel(" / ".join([rename_map.get(y_col, y_col) for y_col in y_columns]))
    if not xtitle:
        plt.xlabel(rename_map.get(x_column, x_column))
    else:
        plt.xlabel(xtitle)
    plt.title(title)

    if figname:
        filename = f"{figname}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        print(filename)
    plt.show()
