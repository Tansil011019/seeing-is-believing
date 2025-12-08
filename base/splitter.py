class BaseSplitter:
    def split(self, df, features, target):
        raise NotImplementedError("Subclasses should implement this method.")