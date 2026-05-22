# MahkiVision Core Agents

## 1. Market Structure Agent
Determines whether the asset is bullish, bearish, or mixed.

Uses:
- price
- MA20
- MA50
- RSI
- daily change
- volatility

Output:
- Market Bias
- Structure Quality
- Plain-English explanation

## 2. Multi-Timeframe Alignment Agent
Checks whether short-term and long-term timeframes agree.

Uses:
- 15m
- 1h
- 4h
- 1d

Output:
- Bullish Alignment
- Bearish Alignment
- Mixed Alignment
- Conflicting Timeframes

## 3. Setup Quality Agent
Scores how strong the current setup is.

Uses:
- market bias
- signal score
- RSI
- risk level
- multi-timeframe alignment

Output:
- Setup Quality
- Setup Grade
- Setup Type

## 4. Risk Protection Agent
Warns when the setup is dangerous or stretched.

Uses:
- RSI
- volatility
- negative signals
- warning signs
- timeframe conflict

Output:
- Risk Level
- Warning Signs
- What Could Go Wrong

## 5. Timing & Confirmation Agent
Helps users avoid entering too early or too late.

Uses:
- RSI
- price vs moving averages
- momentum status
- confirmation signals

Output:
- Timing Status
- What to Watch
- Confirmation Needed

## 6. AI Explanation Agent
Turns all technical data into plain English.

Uses:
- full market context
- setup quality
- risk level
- market bias
- alignment
- timing status

Output:
- AI Read
- Setup Review
- Human Review Checklist