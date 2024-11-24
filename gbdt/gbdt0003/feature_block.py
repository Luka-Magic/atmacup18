from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import List

class AbstractBaseBlock:
    """
    https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/
    """

    def __init__(self) -> None:
        pass

    def fit(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        # return self.transform(input_df)
        raise NotImplementedError()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


def run_block(input_df: pd.DataFrame, blocks: List[AbstractBaseBlock], is_fit):
    output_df = pd.DataFrame()
    for block in blocks:
        name = block.__class__.__name__

        if is_fit:
            # print(f'fit: {name}')
            _df = block.fit(input_df)
        else:
            # print(f'transform: {name}')
            _df = block.transform(input_df)

        # print(f'concat: {name}')
        output_df = pd.concat([output_df, _df], axis=1)
    return output_df

class NumericBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].copy()
        return output_df

class LabelEncodingBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col
        self.encoder = LabelEncoder()

    def fit(self, input_df):
        # return self.transform(input_df)

        self.encoder.fit(input_df[self.col])
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()

        # output_df[self.col] = self.encoder.fit_transform(input_df[self.col])

        # self.encoder.fit(input_df[self.col])
        output_df[self.col] = self.encoder.transform(input_df[self.col])
        return output_df.add_suffix('@le')

class CountEncodingBlock(AbstractBaseBlock):
    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        self.val_count_dict = {}
        self.val_count = input_df[self.col].value_counts()
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].map(self.val_count)
        return output_df.add_suffix('@ce')
    

class AggBlock(AbstractBaseBlock):
    def __init__(self, group_col: str, target_columns: List[str], agg_columns: List[str]) -> None:
        super().__init__()
        self.group_col = group_col
        self.target_columns = target_columns
        self.agg_columns = agg_columns

    def fit(self, input_df):
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.DataFrame()
        group_df = input_df.groupby(self.group_col)
        new_columns = []
        for agg_column in self.agg_columns:
            agg_df = group_df[self.target_columns].agg(agg_column).astype(float).add_suffix(f'@{agg_column}')
            new_columns += agg_df.columns.tolist()
            output_df = pd.concat([output_df, agg_df], axis=1)
        output_df = pd.merge(input_df, output_df, left_on=self.group_col, right_index=True, how='left')
        return output_df[new_columns]