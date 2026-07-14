def compare_model_performance(candidate_metrics, champion_metrics):
    """
    So sánh metrics của Candidate Model với Champion Model.
    Candidate thắng nếu MAPE thấp hơn và R2 cao hơn.
    """
    candidate_mape = candidate_metrics["mape"]
    candidate_r2 = candidate_metrics["r2"]

    champion_mape = champion_metrics["mape"]
    champion_r2 = champion_metrics["r2"]

    # Logic so sánh: Candidate phải tốt hơn Champion ở cả 2 phương diện
    candidate_win = candidate_mape < champion_mape and candidate_r2 > champion_r2

    return bool(candidate_win)
