import pandas as pd
import numpy as np
from typing import Dict, Any

class SignalGenerator:
    """
    Generate unified trading signals from multiple technical indicators
    """
    
    def __init__(self):
        self.signal_types = ['STRONG SELL', 'SELL', 'NEUTRAL', 'BUY', 'STRONG BUY']
    
    def normalize_score(self, value: float, min_val: float = -1, max_val: float = 1) -> float:
        """
        Normalize a score to be between -1 and 1
        
        Args:
            value: Value to normalize
            min_val: Minimum value
            max_val: Maximum value
        
        Returns:
            Normalized score
        """
        if pd.isna(value):
            return 0
        return np.clip(value, min_val, max_val)
    
    def calculate_rsi_score(self, rsi: float, levels: Dict[str, float]) -> float:
        """
        Calculate RSI-based signal score
        
        Args:
            rsi: RSI value
            levels: Dictionary with 'overbought' and 'oversold' levels
        
        Returns:
            Score between -1 (oversold/bullish) and 1 (overbought/bearish)
        """
        if pd.isna(rsi):
            return 0
        
        overbought = levels['overbought']
        oversold = levels['oversold']
        
        if rsi >= overbought:
            # Overbought - bearish signal
            return (rsi - overbought) / (100 - overbought)
        elif rsi <= oversold:
            # Oversold - bullish signal
            return (rsi - oversold) / oversold - 1
        else:
            # Neutral zone
            mid_point = (overbought + oversold) / 2
            if rsi > mid_point:
                return (rsi - mid_point) / (overbought - mid_point) * 0.5
            else:
                return (rsi - mid_point) / (mid_point - oversold) * -0.5
    
    def calculate_macd_score(self, macd: float, signal: float, histogram: float) -> float:
        """
        Calculate MACD-based signal score
        
        Args:
            macd: MACD line value
            signal: Signal line value
            histogram: MACD histogram value
        
        Returns:
            Score between -1 (bearish) and 1 (bullish)
        """
        if pd.isna(macd) or pd.isna(signal) or pd.isna(histogram):
            return 0
        
        # MACD above signal line is bullish
        crossover_score = 1 if macd > signal else -1
        
        # Increasing histogram is bullish
        histogram_score = np.tanh(histogram * 10)  # Normalize large values
        
        # MACD above zero is bullish
        zero_line_score = 1 if macd > 0 else -1
        
        # Combine scores with weights
        total_score = (crossover_score * 0.5 + histogram_score * 0.3 + zero_line_score * 0.2)
        
        return self.normalize_score(total_score)
    
    def calculate_ma_score(self, price: float, ma_short: float, ma_long: float) -> float:
        """
        Calculate Moving Average-based signal score
        
        Args:
            price: Current price
            ma_short: Short-term moving average
            ma_long: Long-term moving average
        
        Returns:
            Score between -1 (bearish) and 1 (bullish)
        """
        if pd.isna(price) or pd.isna(ma_short) or pd.isna(ma_long):
            return 0
        
        # Price position relative to MAs
        if price > ma_short > ma_long:
            return 1  # Strong bullish
        elif price > ma_long > ma_short:
            return 0.5  # Mild bullish
        elif ma_short > price > ma_long:
            return 0.25  # Weak bullish
        elif ma_long > price > ma_short:
            return -0.25  # Weak bearish
        elif ma_long > ma_short > price:
            return -0.5  # Mild bearish
        elif price < ma_short < ma_long:
            return -1  # Strong bearish
        else:
            return 0  # Neutral/Mixed
    
    def calculate_bb_score(self, price: float, upper: float, middle: float, lower: float) -> float:
        """
        Calculate Bollinger Bands-based signal score
        
        Args:
            price: Current price
            upper: Upper Bollinger Band
            middle: Middle Bollinger Band (SMA)
            lower: Lower Bollinger Band
        
        Returns:
            Score between -1 (oversold/bullish) and 1 (overbought/bearish)
        """
        if pd.isna(price) or pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return 0
        
        # Position within bands
        if price >= upper:
            # Above upper band - overbought
            return 1
        elif price <= lower:
            # Below lower band - oversold
            return -1
        else:
            # Within bands - normalize position
            band_width = upper - lower
            if band_width == 0:
                return 0
            
            position = (price - middle) / (band_width / 2)
            return self.normalize_score(position)
    
    def calculate_stochastic_score(self, k: float, d: float) -> float:
        """
        Calculate Stochastic-based signal score
        
        Args:
            k: %K value
            d: %D value
        
        Returns:
            Score between -1 (oversold/bullish) and 1 (overbought/bearish)
        """
        if pd.isna(k) or pd.isna(d):
            return 0
        
        # Overbought/Oversold levels
        if k >= 80 and d >= 80:
            return 1  # Overbought
        elif k <= 20 and d <= 20:
            return -1  # Oversold
        elif k > d and k > 50:
            return 0.5  # Bullish momentum
        elif k < d and k < 50:
            return -0.5  # Bearish momentum
        else:
            # Normalize to -1 to 1 range
            avg_stoch = (k + d) / 2
            return (avg_stoch - 50) / 50
    
    def get_individual_scores(self, row: pd.Series, weights: Dict[str, float], rsi_levels: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate individual indicator scores for a single row
        
        Args:
            row: DataFrame row with indicator values
            weights: Indicator weights
            rsi_levels: RSI overbought/oversold levels
        
        Returns:
            Dictionary of individual scores
        """
        scores = {}
        
        # RSI Score
        scores['rsi'] = self.calculate_rsi_score(float(row['RSI']) if pd.notna(row['RSI']) else 50, rsi_levels)
        
        # MACD Score
        scores['macd'] = self.calculate_macd_score(
            float(row['MACD']) if pd.notna(row['MACD']) else 0,
            float(row['MACD_Signal']) if pd.notna(row['MACD_Signal']) else 0,
            float(row['MACD_Histogram']) if pd.notna(row['MACD_Histogram']) else 0
        )
        
        # Moving Average Score
        scores['ma'] = self.calculate_ma_score(
            float(row['Close']) if pd.notna(row['Close']) else 0,
            float(row['MA_Short']) if pd.notna(row['MA_Short']) else 0,
            float(row['MA_Long']) if pd.notna(row['MA_Long']) else 0
        )
        
        # Bollinger Bands Score
        scores['bb'] = self.calculate_bb_score(
            float(row['Close']) if pd.notna(row['Close']) else 0,
            float(row['BB_Upper']) if pd.notna(row['BB_Upper']) else 0,
            float(row['BB_Middle']) if pd.notna(row['BB_Middle']) else 0,
            float(row['BB_Lower']) if pd.notna(row['BB_Lower']) else 0
        )
        
        # Stochastic Score
        scores['stochastic'] = self.calculate_stochastic_score(
            float(row['Stoch_K']) if pd.notna(row['Stoch_K']) else 50,
            float(row['Stoch_D']) if pd.notna(row['Stoch_D']) else 50
        )
        
        return scores
    
    def calculate_higher_tf_trend(self, higher_df: pd.DataFrame) -> pd.Series:
        """
        Calculate higher timeframe trend direction
        
        Args:
            higher_df: Higher timeframe DataFrame with OHLCV data
        
        Returns:
            Series with trend direction (-1: bearish, 0: neutral, 1: bullish)
        """
        if higher_df.empty:
            return pd.Series(dtype=float)
        
        # Calculate EMAs for trend detection
        ema20 = higher_df['Close'].ewm(span=20).mean()
        ema50 = higher_df['Close'].ewm(span=50).mean()
        
        # Trend conditions
        bullish_trend = (ema20 > ema50) & (higher_df['Close'] > ema20)
        bearish_trend = (ema20 < ema50) & (higher_df['Close'] < ema20)
        
        trend = pd.Series(0, index=higher_df.index)
        trend[bullish_trend] = 1
        trend[bearish_trend] = -1
        
        return trend
    
    def generate_unified_signals(self, df: pd.DataFrame, weights: Dict[str, float], 
                               thresholds: Dict[str, float], rsi_levels: Dict[str, float],
                               higher_df: pd.DataFrame = None, trend_weight: float = 0.3) -> Dict[str, pd.Series]:
        """
        Generate unified trading signals from multiple indicators with optional MTF analysis
        
        Args:
            df: DataFrame with price and indicator data
            weights: Dictionary of indicator weights
            thresholds: Signal thresholds
            rsi_levels: RSI overbought/oversold levels
            higher_df: Higher timeframe DataFrame for trend analysis
            trend_weight: Weight for higher timeframe trend influence
        
        Returns:
            Dictionary containing signal scores, types, and strengths
        """
        signals = {
            'score': [],
            'signal': [],
            'strength': []
        }
        
        # Calculate higher timeframe trend if provided
        higher_trend = None
        if higher_df is not None and not higher_df.empty:
            higher_trend = self.calculate_higher_tf_trend(higher_df)
        
        for idx, row in df.iterrows():
            # Calculate individual scores
            individual_scores = self.get_individual_scores(row, weights, rsi_levels)
            
            # Calculate weighted total score
            total_score = sum(individual_scores[indicator] * weights[indicator] 
                            for indicator in individual_scores.keys())
            
            # Normalize by total weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                total_score /= total_weight
            
            # Apply higher timeframe trend filter
            if higher_trend is not None and len(higher_trend) > 0:
                # Find the closest higher timeframe point
                try:
                    trend_value = higher_trend.loc[higher_trend.index <= idx].iloc[-1] if len(higher_trend.loc[higher_trend.index <= idx]) > 0 else 0
                    
                    # Adjust signal based on trend
                    if trend_value > 0:  # Bullish trend
                        if total_score > 0:
                            total_score = total_score * (1 + trend_weight)
                        else:
                            total_score = total_score * (1 - trend_weight * 0.5)
                    elif trend_value < 0:  # Bearish trend
                        if total_score < 0:
                            total_score = total_score * (1 + trend_weight)
                        else:
                            total_score = total_score * (1 - trend_weight * 0.5)
                except:
                    pass  # Skip trend adjustment if index matching fails
            
            # Determine signal type based on thresholds
            if total_score >= thresholds['strong_buy']:
                signal_type = 'STRONG BUY'
            elif total_score >= thresholds['buy']:
                signal_type = 'BUY'
            elif total_score <= thresholds['strong_sell']:
                signal_type = 'STRONG SELL'
            elif total_score <= thresholds['sell']:
                signal_type = 'SELL'
            else:
                signal_type = 'NEUTRAL'
            
            # Calculate signal strength (0 to 1)
            signal_strength = abs(total_score)
            
            signals['score'].append(total_score)
            signals['signal'].append(signal_type)
            signals['strength'].append(signal_strength)
        
        return {
            'score': pd.Series(signals['score'], index=df.index),
            'signal': pd.Series(signals['signal'], index=df.index),
            'strength': pd.Series(signals['strength'], index=df.index)
        }
    
    def get_signal_summary(self, signals: pd.Series) -> Dict[str, Any]:
        """
        Get summary statistics for signals
        
        Args:
            signals: Series of signal types
        
        Returns:
            Dictionary with signal statistics
        """
        signal_counts = signals.value_counts()
        total_signals = len(signals)
        
        summary = {
            'total_signals': total_signals,
            'signal_distribution': signal_counts.to_dict(),
            'signal_percentages': (signal_counts / total_signals * 100).to_dict(),
            'most_common_signal': signal_counts.index[0] if not signal_counts.empty else 'NEUTRAL',
            'signal_changes': (signals != signals.shift(1)).sum()
        }
        
        return summary
    
    def detect_signal_divergence(self, df: pd.DataFrame, price_col: str = 'Close', 
                               signal_col: str = 'Signal_Score', window: int = 20) -> pd.Series:
        """
        Detect divergence between price and signal
        
        Args:
            df: DataFrame with price and signal data
            price_col: Column name for price
            signal_col: Column name for signal score
            window: Lookback window for divergence detection
        
        Returns:
            Series indicating divergence points
        """
        price = df[price_col]
        signal = df[signal_col]
        
        # Calculate rolling correlation
        correlation = price.rolling(window=window).corr(signal)
        
        # Detect divergence (negative correlation)
        divergence = correlation < -0.5
        
        return divergence
