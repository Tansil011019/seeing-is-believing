import logging
from sklearn.model_selection import GroupShuffleSplit
from base.splitter import BaseSplitter

logger = logging.getLogger(__name__)

class GroupSplitter(BaseSplitter):
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        group_col = None,
        n_splits: int = 1
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.group_col = group_col
        self.n_splits = n_splits

    def split(self, df, features, target):
        X = df[features]
        y = df[target]

        if self.group_col is None:
            raise ValueError("group_col must be specified for GroupSplitter.")
        
        gss = GroupShuffleSplit(n_splits=self.n_splits, test_size=self.test_size, random_state=self.random_state)

        train_idx, val_idx = next(gss.split(X, y, groups=df[self.group_col]))

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Performed group-based train/val split with test_size={self.test_size}, random_state={self.random_state}, group_column={self.group_col}")
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        return X_train, X_val, y_train, y_val