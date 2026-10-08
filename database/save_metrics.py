from sqlalchemy import text

from database.postgres_sql import get_engine


def _format_metric_list(value) -> str | None:
    """
    Convert a metric that may be either a list or a string
    into a PostgreSQL-friendly string.
    """

    if value is None:
        return None

    if isinstance(value, list):
        return "\n".join(str(item) for item in value)

    return str(value)


def save_metrics(
    company: str,
    year: int,
    metrics: dict
) -> None:
    """
    Save extracted financial metrics to PostgreSQL.

    Args:
        company: Company name.
        year: Fiscal year.
        metrics: Extracted KPI dictionary.
    """

    engine = get_engine()

    query = """
    INSERT INTO financial_metrics (
        company,
        year,
        revenue,
        net_income,
        operating_income,
        cash_flow,
        total_assets,
        total_liabilities,
        risk_factors,
        growth_drivers
    )
    VALUES (
        :company,
        :year,
        :revenue,
        :net_income,
        :operating_income,
        :cash_flow,
        :total_assets,
        :total_liabilities,
        :risk_factors,
        :growth_drivers
    )
    """

    # Support both Pydantic aliases and Python field names.
    params = {
        "company": company,
        "year": year,

        "revenue": (
            metrics.get("Revenue")
            if metrics.get("Revenue") is not None
            else metrics.get("revenue")
        ),

        "net_income": (
            metrics.get("Net Income")
            if metrics.get("Net Income") is not None
            else metrics.get("net_income")
        ),

        "operating_income": (
            metrics.get("Operating Income")
            if metrics.get("Operating Income") is not None
            else metrics.get("operating_income")
        ),

        "cash_flow": (
            metrics.get("Cash Flow from Operating Activities")
            if metrics.get("Cash Flow from Operating Activities") is not None
            else metrics.get("cash_flow")
        ),

        "total_assets": (
            metrics.get("Total Assets")
            if metrics.get("Total Assets") is not None
            else metrics.get("total_assets")
        ),

        "total_liabilities": (
            metrics.get("Total Liabilities")
            if metrics.get("Total Liabilities") is not None
            else metrics.get("total_liabilities")
        ),

        "risk_factors": _format_metric_list(
            metrics.get("Top Risk Factors")
            if metrics.get("Top Risk Factors") is not None
            else metrics.get("risk_factors")
        ),

        "growth_drivers": _format_metric_list(
            metrics.get("Top Growth Drivers")
            if metrics.get("Top Growth Drivers") is not None
            else metrics.get("growth_drivers")
        )
    }

    with engine.begin() as connection:
        connection.execute(text(query), params)

    print(
        f"Successfully saved metrics for {company} {year}"
    )


if __name__ == "__main__":
    sample_metrics = {
        "Revenue": "$391,035",
        "Net Income": "$93,736",
        "Operating Income": "$123,216",
        "Cash Flow from Operating Activities": "$118,254",
        "Total Assets": "$364,980",
        "Total Liabilities": "$308,030",
        "Top Risk Factors": [
            "Macroeconomic conditions including inflation, "
            "interest rates, and currency fluctuations could "
            "materially impact results.",

            "High competition with aggressive pricing, "
            "short product life cycles, and rapid "
            "technological changes.",

            "Dependence on single or limited sources for "
            "certain components, with potential supply shortages.",

            "Exposure to foreign exchange rate fluctuations "
            "impacting sales and margins.",

            "Legal and regulatory challenges, including "
            "significant tax disputes such as the State Aid Decision."
        ],
        "Top Growth Drivers": [
            "Increased Services revenue from advertising, "
            "App Store, and cloud services.",

            "Higher Mac sales driven by increased laptop demand.",

            "Continued strong iPhone sales performance."
        ]
    }

    save_metrics(
        company="Apple",
        year=2024,
        metrics=sample_metrics
    )