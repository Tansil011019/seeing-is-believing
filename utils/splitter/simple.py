import pandas as pd
from sklearn.model_selection import train_test_split
from base.splitter import BaseSplitter
import logging

logger = logging.getLogger(__name__)

class SimpleSplitter(BaseSplitter):
    def __init__(
        self,
        test_size: 0.2,
        random_state: 42,
        stratify = True
    ): 
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
    
    def split(self, df, features, target):
        X = df[features]
        y = df[target]

        stratify_param = y if self.stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        logger.info(f"Performed simple train/val split with test_size={self.test_size}, random_state={self.random_state}, stratify={self.stratify}")
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val