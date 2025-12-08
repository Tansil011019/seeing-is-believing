import logging
import joblib

logger = logging.getLogger(__name__)

class FitTrainer:
    def __init__(
        self,
        model,
        use_eval_set = False,
        early_stopping_rounds = 50, 
        verbose = False
    ):
        self.model = model
        self.use_eval_set = use_eval_set
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_params = {}
        if self.use_eval_set and X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = self.verbose

            if self.early_stopping_rounds > 0:
                fit_params['early_stopping_rounds'] = self.early_stopping_rounds
        else: 
            logger.warning("Validation set not provided or use_eval_set is False; proceeding without eval_set.")

        logger.info(f"Fitting model: {self.model.__class__.__name__} with parameters: {fit_params}")
        
        self.model.fit(
            X_train, 
            y_train, 
            **fit_params
        )
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")