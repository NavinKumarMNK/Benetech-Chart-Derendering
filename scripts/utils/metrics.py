import numpy as np
import pandas as pd
from polyleven import levenshtein 
from typing  import List, Union, Tuple, Dict
import re


PROMPT_TOKEN = "<|BOS|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

SEPARATOR_TOKENS = [
    PROMPT_TOKEN,
    X_START,
    X_END,
    Y_START,
    Y_END,
]

LINE_TOKEN =  "<line>" 
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
SCATTER_TOKEN = "<scatter>"
DOT_TOKEN = "<dot>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    SCATTER_TOKEN,
    DOT_TOKEN,
]

NEW_TOKENS = SEPARATOR_TOKENS + CHART_TYPE_TOKENS

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    @brief   : Calculate the Root Mean Square Error (RMSE) between the 
                true and predicted values.
    @args    :
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.
    @returns : float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    @brief   : Calculate the sigmoid of a value.
    @args    :
        x (float): The value.
    @returns : float: The sigmoid of the value.
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    @brief   : Calculate the normalized RMSE between the true and predicted values.
    @args    :
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.
    @returns : float: The normalized RMSE.
    @note    : The normalized RMSE is calculated as the RMSE between the true and predicted values
                divided by the RMSE between the true values and the mean of the true values. 
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    @brief   : Calculate the normalized Levenshtein score between the true and predicted values.
    @args    :
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.
    @returns : float: The normalized Levenshtein score.
    @note    : The normalized Levenshtein score is calculated as the Levenshtein distance between the
                true and predicted values divided by the sum of the lengths of the true values.
    """
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true: List[Union[float, str]], y_pred: List[Union[float, str]]) -> float:
    """
    @brief   : Calculate the score between the true and predicted values.
    @args    :
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.
    @returns : float: The score.
    @note    : The score is calculated as the normalized RMSE if the true and predicted values are
                floats, or the normalized Levenshtein score if the true and predicted values are
                strings.
    """
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    @brief   : Calculate the Benetech score between the ground truth and predicted values.
    @args    :
        ground_truth (pd.DataFrame): The ground truth values.
        predictions (pd.DataFrame): The predicted values.
    @returns : float: The Benetech score.
    @note    : The Benetech score is calculated as the mean of the scores between the true and
                predicted values for each chart type.
    """

    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score


def string2triplet(pred_string: str) -> Tuple[str, List[str], List[str]]:
    """
    @brief   : Convert a prediction string to a triplet of chart type, x values, and y values.
    @args    :
        pred_string (str): The prediction string.
    @returns : Tuple[str, List[str], List[str]]: The chart type, x values, and y values.
    @note    : The prediction string is of the form:
                    <chart_type><x_values><y_values>
                where <chart_type> is one of the chart types, <x_values> is a semicolon-separated
                list of x values, and <y_values> is a semicolon-separated list of y values.
    """
    
    chart_type = "line"
    for tok in CHART_TYPE_TOKENS:
        if tok in pred_string:
            chart_type = tok.strip("<>")

    pred_string = re.sub(r"<one>", "1", pred_string)

    x = pred_string.split(X_START)[1].split(X_END)[0].split(";")
    y = pred_string.split(Y_START)[1].split(Y_END)[0].split(";")

    if len(x) == 0 or len(y) == 0:
        return chart_type, [], []

    min_length = min(len(x), len(y))

    x = x[:min_length]
    y = y[:min_length]

    return chart_type, x, y


def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame) -> Dict[str, float]:
    """
    @brief   : Calculate the Benetech score between the ground truth and predicted values.
    @args    :
        val_outputs (List[str]): The predicted values.
        val_ids (List[str]): The IDs of the predicted values.
        gt_df (pd.DataFrame): The ground truth values.
    @returns : Dict[str, float]: The Benetech score and the score for each chart type.
    @note    : The Benetech score is calculated as the mean of the scores between the true and
                predicted values for each chart type.
    """
    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
            pred_triplets.append(("line", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )

    overall_score, chart_type2score = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )

    return {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }