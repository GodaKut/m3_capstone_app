from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import pandas as pd
import numpy as np

class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, installments_payments, bureau, application_prev , POS_CASH_balance, selected_columns = None):
        """
        Custom transformer for creating the following features:
            - installments_payments:
                AVG_INSTALMENT_DEBT - the average difference between the payed ammount and instalment (AMT_PAYMENT- AMT_INSTALMENT)
                AVG_INSTALMENT_DELAY - the average ammount of days the installemnt was late to be payed (DAYS_ENTRY_PAYMENT- DAYS_INSTALMENT)
            - bureau:
                AMT_CREDIT_SUM_STD - the standard deviation of the ammount of credit
                AMT_CREDIT_SUM_DEBT_AVG - the average ammount of debt
                AMT_CREDIT_SUM_OVERDUE_FLAG - flags if there is any ammount overdue
                DEBT_TO_CREDIT_AVG - average debt to credit ratio calculated by AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM
                CREDIT_ACTIVE_COUNT - the count of active credits
                CREDIT_DAY_OVERDUE_AVG - the average number of days past due on Credit Bureau credit at the time of application for related loan
                CNT_CREDIT_PROLONG_AVG - the average number of times credits were prolonged
            - application_prev:
                W_CREDIT_APPROVAL_RATIO - the weighted average ratio of the ammount of credit applied vs granted after evaluation. The weight helps include relevance and is calculated based on DAYS_DECISION column (the ammount of days has passed between current and previous application decision date)
                W_ANUITY_GROWTH - the increase/decrease in anuity between current and latest previous application which is weighted based on DAYS_DECISION
                W_CREDIT_GROWTH - the increase/decrease in credit amount between current and latest previous application which is weighted based on DAYS_DECISION
                CASH_LOAN_COUNT - count of cash type previous applications
                REVOLVING_LOAN_COUNT - count of revolving type previous applications
            - POS_CASH_balance:
                SK_DPD_AVG - the average of days past due of client previous applications
                SK_DPD_STD - the standard deviation of days past due of client previous applications
       
        """
        self.installments_payments = installments_payments
        self.bureau = bureau
        self.application_prev = application_prev
        self.POS_CASH_balance = POS_CASH_balance
        self.selected_columns = selected_columns
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame)-> pd.DataFrame:
        '''
           In this section the mentioned features are created.
           In addition, the Name feature is removed since deemed not useful.

           Parameters:
            - X: DataFrame which will have the columns added to 
            - installments_payments: dataframe used to create the new columns
            - bureau: dataframe used to create the new columns
            - application_prev: dataframe used to create the new columns
            - POS_CASH_balance: dataframe used to create the new columns

            Returns:
            - Transformed DataFrame with new engineered columns. 
        '''
        #  Features creation from installments_payments 
        if (self.selected_columns == None) or ("AVG_INSTALMENT_DEBT" in self.selected_columns):
            avg_instalment_debt = (
                self.installments_payments.assign(
                    INSTALMENT_DEBT=lambda df: df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]
                )
                .groupby("SK_ID_CURR")["INSTALMENT_DEBT"]
                .mean()
                .rename("AVG_INSTALMENT_DEBT")
            )
            X = X.join(avg_instalment_debt, on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("AVG_INSTALMENT_DELAY" in self.selected_columns):
            avg_instalment_delay = (
                self.installments_payments.assign(
                    INSTALMENT_DEBT=lambda df: df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
                )
                .groupby("SK_ID_CURR")["INSTALMENT_DEBT"]
                .mean()
                .rename("AVG_INSTALMENT_DELAY")
            )
            X = X.join(avg_instalment_delay, on="SK_ID_CURR")
        # Features creation from bureau
        if (self.selected_columns == None) or ("AMT_CREDIT_SUM_STD" in self.selected_columns):
            amt_credit_sum_std = (
                self.bureau
                .groupby("SK_ID_CURR")["AMT_CREDIT_SUM"]
                .std() 
                .rename("AMT_CREDIT_SUM_STD") 
            )
            X = X.join(amt_credit_sum_std, on="SK_ID_CURR")
        if (self.selected_columns == None) or ("AMT_CREDIT_SUM_DEBT_AVG" in self.selected_columns):
            amt_credit_sum_debt_avg = (
                self.bureau
                .groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"]
                .std() 
                .rename("AMT_CREDIT_SUM_DEBT_AVG") 
            )
            X = X.join(amt_credit_sum_debt_avg, on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("AMT_CREDIT_SUM_OVERDUE_FLAG" in self.selected_columns):
            amt_credit_sum_overdue_flag = (
                self.bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"]
                .apply(lambda x: int((x > 0).any()))
                .rename("AMT_CREDIT_SUM_OVERDUE_FLAG")
            )
            X = X.join(amt_credit_sum_overdue_flag, on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("DEBT_TO_CREDIT_AVG" in self.selected_columns):
            debt_to_credit_avg = (
                self.bureau.assign(
                    DEBT_TO_CREDIT=lambda df: df["AMT_CREDIT_SUM_DEBT"].fillna(0) / df["AMT_CREDIT_SUM"].replace(0, np.nan)
                )
                .groupby("SK_ID_CURR")["DEBT_TO_CREDIT"]
                .mean()
                .rename("DEBT_TO_CREDIT_AVG")
            )
            X = X.join(debt_to_credit_avg, on="SK_ID_CURR")

        if (self.selected_columns == None) or ("CREDIT_ACTIVE_COUNT" in self.selected_columns):
            credit_active_count = (
                self.bureau[self.bureau["CREDIT_ACTIVE"] == "Active"]
                .groupby("SK_ID_CURR")["CREDIT_ACTIVE"]
                .count()
                .rename("CREDIT_ACTIVE_COUNT")
            )
            X = X.join(credit_active_count, on="SK_ID_CURR")

        if (self.selected_columns == None) or ("CREDIT_DAY_OVERDUE_AVG" in self.selected_columns):
            credit_day_overdue_avg =  (
                self.bureau
                .groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"]
                .mean() 
                .rename("CREDIT_DAY_OVERDUE_AVG") 
            )
            X = X.join(credit_day_overdue_avg, on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("CNT_CREDIT_PROLONG_AVG" in self.selected_columns):
            cnt_credit_prolong_avg =  (
                self.bureau
                .groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"]
                .mean() 
                .rename("CNT_CREDIT_PROLONG_AVG") 
            )
            X = X.join(cnt_credit_prolong_avg, on="SK_ID_CURR")
        # Features creation from application_prev
        if (self.selected_columns == None) or ("W_CREDIT_APPROVAL_DIFF" in self.selected_columns):
            w_credit_approval_diff = (
                self.application_prev
                .assign(
                    CREDIT_APPROVAL_DIFF=lambda df: df["AMT_APPLICATION"] - df["AMT_CREDIT"],
                    WEIGHT=lambda df: 1 / (df["DAYS_DECISION"].abs() + 1)
                )
                .groupby("SK_ID_CURR").apply(
                    lambda group: (group["CREDIT_APPROVAL_DIFF"] * group["WEIGHT"]).mean()
                )
                .rename("W_CREDIT_APPROVAL_DIFF")
            )
            X = X.join(w_credit_approval_diff, on="SK_ID_CURR")
    
        most_recent_application = (
            self.application_prev
            .sort_values("DAYS_DECISION", ascending=False)
            .groupby("SK_ID_CURR")
            .head(1) 
        )
        if (self.selected_columns == None) or ("W_ANNUITY_GROWTH" in self.selected_columns):
            w_annuity_growth = (
                most_recent_application[["SK_ID_CURR", "AMT_ANNUITY", "DAYS_DECISION"]]
                .rename(columns={"AMT_ANNUITY": "ANNUITY_prev", "DAYS_DECISION": "DAYS_DECISION_prev"})
                .merge(
                    X[["AMT_ANNUITY"]],
                    left_on="SK_ID_CURR",
                    right_index=True,
                    how="left"
                )
                .assign(
                    ANNUITY_GROWTH=lambda df: df["AMT_ANNUITY"] - df["ANNUITY_prev"],
                    WEIGHT=lambda df: 1 / (df["DAYS_DECISION_prev"].abs() + 1),
                    W_ANNUITY_GROWTH=lambda df: df["ANNUITY_GROWTH"] * df["WEIGHT"]
                )
            )
            X = X.join(w_annuity_growth[["SK_ID_CURR", "W_ANNUITY_GROWTH"]].set_index("SK_ID_CURR"), on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("W_CREDIT_GROWTH" in self.selected_columns):
            w_credit_growth = (
                most_recent_application[["SK_ID_CURR", "AMT_CREDIT", "DAYS_DECISION"]]
                .rename(columns={"AMT_CREDIT": "CREDIT_prev", "DAYS_DECISION": "DAYS_DECISION_prev"})
                .merge(
                    X[["AMT_CREDIT"]],
                    left_on="SK_ID_CURR",
                    right_index=True,
                    how="left"
                )
                .assign(
                    CREDIT_GROWTH=lambda df: df["AMT_CREDIT"] - df["CREDIT_prev"],
                    WEIGHT=lambda df: 1 / (df["DAYS_DECISION_prev"].abs() + 1),
                    W_CREDIT_GROWTH=lambda df: df["CREDIT_GROWTH"] * df["WEIGHT"]
                )
            )
            X = X.join(w_credit_growth[["SK_ID_CURR", "W_CREDIT_GROWTH"]].set_index("SK_ID_CURR"), on="SK_ID_CURR")

        if (self.selected_columns == None) or ("CASH_LOAN_COUNT" in self.selected_columns):
            cash_loan_count = (
                self.application_prev[self.application_prev["NAME_CONTRACT_TYPE"] == "Cash loans"]
                .groupby("SK_ID_CURR")["NAME_CONTRACT_TYPE"]
                .count()
                .rename("CASH_LOAN_COUNT")
            )
            X = X.join(cash_loan_count, on="SK_ID_CURR")

        if (self.selected_columns == None) or ("REVOLVING_LOAN_COUNT" in self.selected_columns):
            revolving_loan_count = (
                self.application_prev[self.application_prev["NAME_CONTRACT_TYPE"] == "Revolving loans"]
                .groupby("SK_ID_CURR")["NAME_CONTRACT_TYPE"]
                .count()
                .rename("REVOLVING_LOAN_COUNT")
            )
            X = X.join(revolving_loan_count, on="SK_ID_CURR")
        # Features creation from POS_CASH_balance
        if (self.selected_columns == None) or ("SK_DPD_AVG" in self.selected_columns):
            sk_dpd_avg = (
                self.POS_CASH_balance
                .groupby("SK_ID_CURR")["SK_DPD"]
                .mean() 
                .rename("SK_DPD_AVG") 
            )
            X = X.join(sk_dpd_avg, on="SK_ID_CURR")
        
        if (self.selected_columns == None) or ("SK_DPD_STD" in self.selected_columns):
            sk_dpd_std = (
                self.POS_CASH_balance
                .groupby("SK_ID_CURR")["SK_DPD"]
                .std() 
                .rename("SK_DPD_STD") 
            )
            X = X.join(sk_dpd_std, on="SK_ID_CURR")
        
     
        return X

class DropUnselectedColumns(BaseEstimator, TransformerMixin):
    def __init__(self, rank_keep_columns: set, VIF_drop : list = None):
        """
        Custom transformer for drop columns in a DataFrame if  
        they are not in the passe list of useful columns.

        Parameters:
        - rank_keep_columns: list of column names keep in the DataFrame.
        """
        self.rank_keep_columns = rank_keep_columns
        self.VIF_drop = VIF_drop


    def fit(self, X, y=None):
        """
        This transformer does not learn anything from the data,
        so we simply return self.
        """
        return self

    def transform(self, X):
        """
        Removes the columns of the input DataFrame keeping only the passed list.

        Parameters:
        - X: DataFrame that will be modified.

        Returns:
        - Transformed DataFrame with only columns names passed.
        """
        if self.VIF_drop != None:
            self.rank_keep_columns = [x for x in self.rank_keep_columns if x not in self.VIF_drop]
        
        return  X[self.rank_keep_columns]


class EncodeOrganizationType(BaseEstimator, TransformerMixin):
    def  __init__(self):
        """
        Custom transformer for encoding ORGANIZATION_TYPE column in a DataFrame.
        Since it has 58 unique values it is necesarry to use a mix of ordinal 
        and one-hot encoding stratedies

        Parameters:
        - feature_names: list of new column names to assign to the DataFrame.
        - to_round_cols: list of column to round.
        """
        self.column_order_ = None

    def fit(self, X, y=None):
        """
        This transformer learns the column order
        so it can be ensured to be the same.
        """
        self.column_order_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame)-> pd.DataFrame:
        """
        Encodes ORGANIZATION_TYPE columns the following way:
            1) If type has 1 word it encodes it as one-hot 
            2) If there are several words then it must contain
        area and the numeric type. We separate thode parts.
            3) one-hot encode the are and keep type as ordinal encoding

        Parameters:
        - X: DataFrame whose columns are to be renamed.

        Returns:
        - Transformed DataFrame with samller number of one-hot encodined columns.
        """
        one_word_type = []
        industry_level = []
        for type in X[X["ORGANIZATION_TYPE"].notna()][
            "ORGANIZATION_TYPE"
        ].unique():
            if len(type.split()) < 3:
                one_word_type.append(type)
            else:
                industry_level.append(type.lower().split("type")[0])
        
        for value in one_word_type:
            col_name = f"ORGANIZATION_TYPE_{value.strip().replace(':', '').replace(' ', '_')}"
            X[col_name] = (X["ORGANIZATION_TYPE"] == value).astype(int)

        for key in industry_level:
            col_name = f"ORGANIZATION_TYPE_{key.strip().replace(':', '').replace(' ', '_')}"  # Create unique column names
            X[col_name] = X["ORGANIZATION_TYPE"].apply(
            lambda x: int(x.lower().split('type')[-1].strip()) if key in x.lower() and 'type' in x.lower() else 0
        )

        if self.column_order_ is not None:
            X = X.reindex(columns=self.column_order_, fill_value=0)
        
        X.drop(columns=['ORGANIZATION_TYPE'], inplace=True)
        return X


class ReorderColumns(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_step):
        """
        Custom transformer for renaming columns in a DataFrame after 
        ordinal encoding and imputation since the names become X1, X2...
        Round the columns to integer values for those columns that had 
        a value imputed since float is imputed instead of integer.

        Parameters:
        - feature_names: list of new column names to assign to the DataFrame.
        - to_round_cols: list of column to round.
        """
        self.encoder_step = encoder_step


    def fit(self, X, y=None):
        """
        This transformer does not learn anything from the data,
        so we simply return self.
        """
        return self

    def transform(self, X):
        """
        Renames the columns of the input DataFrame to the specified feature names.

        Parameters:
        - X: DataFrame whose columns are to be renamed.

        Returns:
        - Transformed DataFrame with renamed columns.
        """
        if hasattr(self.encoder_step, "column_order_") and self.encoder_step.column_order_:
            X = X.reindex(columns=self.encoder_step.column_order_, fill_value=0)
        return X


class DecodeCatOrdEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_cols, cat_ord_encoder_step, categories):
        """
        Custom transformer for double-checking the name of the columns after one-hot 
        encoding has an integer prefix.

        Parameters:
        - col_list: list of column that were one-hot encoded
        """
        self.one_hot_enc_cols = one_hot_enc_cols
        self.cat_ord_encoder_step = cat_ord_encoder_step
        self.categories = categories

    def fit(self, X, y=None):
        """
        This transformer does not learn anything from the data,
        so we simply return self.
        """
        return self

    def transform(self, X):
        """
        Renames the columns of the input DataFrame to the specified feature names.

        Parameters:
        - X: DataFrame whose columns are to be renamed.

        Returns:
        - Transformed DataFrame with renamed columns.
        """
        for col, cat in zip(self.one_hot_enc_cols, self.categories):
            max_cat = len(cat)-1
            X[col] = X[col].round().astype(np.int32)
            X[col] = X[col].clip(0, max_cat).astype(np.int32)

        for name, transformer, columns in self.cat_ord_encoder_step.transformers_:
            if hasattr(transformer, "inverse_transform"):
                X[self.one_hot_enc_cols] = transformer.inverse_transform(X[self.one_hot_enc_cols])
        
        for col in self.one_hot_enc_cols:
            X[col] = X[col].str.replace("/","or").str.replace(" ","_").str.replace("-","_").str.replace(",","")
        
        for col in X.columns:
            if (len(X[col].unique()) == 2) and (X[col].dtype != 'object'):
                X[col] =  X[col].astype(np.int32)
            if (len(X[col].unique()) == 1) and ((X[col].unique() == 0) or (X[col].unique() == 1)):
                X[col] =  X[col].astype(np.int32)

        return X



