from river import stream
import numpy as np
from . import base


class LendingClub(base.FileDataset):
    """Lending Club.

    This dataset contains features from loans that are accepted or not.

    """

    def __init__(self):
        super().__init__(
            n_samples=13231176,
            n_features=19,
            task=base.BINARY_CLF,
            filename="lending_club.csv.zip",
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="rejected",
            converters={
                "dti": float,
                "emp_length": lambda x: 0 if x == '' else float(x),
                "loan_amnt": float,
                "rejected": int,
                "purpose_Business": lambda x: x == "1",
                "purpose_car": lambda x: x == "1",
                "purpose_credit_card": lambda x: x == "1",
                "purpose_debt_consolidation": lambda x: x == "1",
                "purpose_home_improvement": lambda x: x == "1",
                "purpose_house": lambda x: x == "1",
                "purpose_major_purchase": lambda x: x == "1",
                "purpose_medical": lambda x: x == "1",
                "purpose_moving": lambda x: x == "1",
                "purpose_other": lambda x: x == "1",
                "purpose_renewable_energy": lambda x: x == "1",
                "purpose_small_business": lambda x: x == "1",
                "purpose_vacation": lambda x: x == "1",
                "purpose_wedding": lambda x: x == "1"
            },
            # parse_dates={"issue_d": "%Y-%m-%d"},
            drop=["id", "issue_d"],
        )
