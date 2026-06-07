from demand_forecast.data.hierarchy import ForecastHierarchy


def test_get_level_series_item_store(synthetic_sales):
    h = ForecastHierarchy(synthetic_sales)
    base = h.get_level_series("item_store")
    # 10 unique item×store combinations in fixture
    assert base["unique_id"].nunique() == 10


def test_get_level_series_state(synthetic_sales):
    h = ForecastHierarchy(synthetic_sales)
    state_df = h.get_level_series("state")
    assert set(state_df["unique_id"].unique()) == {"CA", "TX"}


def test_get_level_series_total(synthetic_sales):
    h = ForecastHierarchy(synthetic_sales)
    total_df = h.get_level_series("total")
    assert total_df["unique_id"].nunique() == 1


def test_summing_matrix_shape(synthetic_sales):
    h = ForecastHierarchy(synthetic_sales)
    s_mat = h.get_summing_matrix()
    n_base = h.get_level_series("item_store")["unique_id"].nunique()
    n_all = sum(
        h.get_level_series(lvl)["unique_id"].nunique()
        for lvl in ForecastHierarchy.LEVELS
    )
    assert s_mat.shape == (n_all, n_base)


def test_summing_matrix_aggregation_correctness(synthetic_sales):
    """S @ base_totals must equal the grand total."""
    h = ForecastHierarchy(synthetic_sales)
    s_mat = h.get_summing_matrix()
    base = h.get_level_series("item_store")
    base_totals = (
        base.groupby("unique_id")["y"].sum().reindex(h.base_series_order_).values
    )
    projected = s_mat @ base_totals

    total_sum = float(synthetic_sales["sales"].sum())
    total_row_idx = h.all_series_order_.index("total")
    assert abs(projected[total_row_idx] - total_sum) < 1e-6


def test_coherence_check_on_real_data_is_zero(synthetic_sales):
    """Real aggregated data is coherent by construction."""
    h = ForecastHierarchy(synthetic_sales)
    forecasts = {lvl: h.get_level_series(lvl) for lvl in ForecastHierarchy.LEVELS}
    err = h.coherence_check(forecasts)
    assert err < 1e-6


def test_coherence_check_detects_incoherence(synthetic_sales):
    h = ForecastHierarchy(synthetic_sales)
    forecasts = {lvl: h.get_level_series(lvl) for lvl in ForecastHierarchy.LEVELS}
    forecasts["item_store"] = forecasts["item_store"].copy()
    forecasts["item_store"]["y"] = forecasts["item_store"]["y"] * 100
    err = h.coherence_check(forecasts)
    assert err > 0.01
