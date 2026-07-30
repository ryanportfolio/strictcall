"""External API tool: currency conversion via the Frankfurter API (no key needed).

Point balances in the warehouse redeem at a fixed USD value; this tool lets the
agent answer questions like "what is that worth in EUR?". HTTP retries happen at
the transport layer; anything else becomes a structured ToolError.
"""

import httpx
from langchain_core.tools import StructuredTool

from strictcall.contracts import FxRateInput, FxRateResult, ToolError

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"

FX_RATE_DESCRIPTION = (
    "Get the latest exchange rate between two ISO 4217 currencies "
    "(European Central Bank reference rates). Returns JSON with base, target, "
    "rate, and the as_of date."
)


def make_fx_tool(client: httpx.Client | None = None) -> StructuredTool:
    http = client or httpx.Client(
        transport=httpx.HTTPTransport(retries=2),
        timeout=httpx.Timeout(10.0),
    )

    def run_fx_rate(base: str, target: str) -> str:
        if base == target:
            return ToolError(
                error="base and target are the same currency; the rate is 1 by definition."
            ).model_dump_json()
        try:
            response = http.get(FRANKFURTER_URL, params={"base": base, "symbols": target})
            response.raise_for_status()
            payload = response.json()
            result = FxRateResult(
                base=payload["base"],
                target=target,
                rate=payload["rates"][target],
                as_of=payload["date"],
            )
            return result.model_dump_json()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return ToolError(
                error=f"{type(exc).__name__}: {exc}",
                hint="Use ISO 4217 codes covered by the ECB reference rates, e.g. USD, EUR, GBP.",
            ).model_dump_json()

    return StructuredTool.from_function(
        func=run_fx_rate,
        name="fx_rate",
        description=FX_RATE_DESCRIPTION,
        args_schema=FxRateInput,
    )
