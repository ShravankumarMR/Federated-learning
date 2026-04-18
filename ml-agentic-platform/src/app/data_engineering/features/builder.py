import pandas as pd


class FeatureBuilder:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        if "timestamp" in transformed.columns:
            transformed["timestamp"] = pd.to_datetime(transformed["timestamp"], errors="coerce")
        return transformed
