"""Preprocessing utilities for AnnData objects."""

from typing import Optional

import pandas as pd
from anndata import AnnData
from sklearn.preprocessing import StandardScaler


class AnnDataStandardScaler:
    """Standardize numerical columns of ``adata.obs`` using sklearn's StandardScaler.

    Categorical and object-typed columns are detected automatically and left
    unchanged.  The API mirrors the familiar sklearn fit / transform pattern.

    Parameters
    ----------
    **scaler_kwargs
        Keyword arguments forwarded to :class:`sklearn.preprocessing.StandardScaler`
        (e.g. ``with_mean``, ``with_std``).

    Attributes
    ----------
    numeric_cols_ : list[str]
        Names of ``adata.obs`` columns that were identified as numerical and
        will be (or have been) scaled.
    categorical_cols_ : list[str]
        Names of ``adata.obs`` columns that are categorical / non-numeric and
        are left unchanged.
    scaler_ : sklearn.preprocessing.StandardScaler
        The fitted scaler instance.

    Examples
    --------
    >>> scaler = AnnDataStandardScaler()
    >>> adata_scaled = scaler.fit_transform(adata)          # returns a copy
    >>> scaler.fit_transform(adata, inplace=True)            # modifies in place
    """

    def __init__(self, **scaler_kwargs):
        self._scaler_kwargs = scaler_kwargs

    def fit(self, adata: AnnData) -> "AnnDataStandardScaler":
        """Fit the scaler on the numerical columns of ``adata.obs``.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix whose ``.obs`` DataFrame is inspected.

        Returns
        -------
        self
        """
        obs = adata.obs
        self.numeric_cols_ = [
            c for c in obs.columns if pd.api.types.is_numeric_dtype(obs[c])
        ]
        self.categorical_cols_ = [
            c for c in obs.columns if c not in self.numeric_cols_
        ]
        self.scaler_ = StandardScaler(**self._scaler_kwargs).fit(
            obs[self.numeric_cols_]
        )
        return self

    def transform(self, adata: AnnData, inplace: bool = False) -> Optional[AnnData]:
        """Scale the numerical columns of ``adata.obs``.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix to transform.
        inplace : bool, optional
            If ``True``, modify ``adata.obs`` directly and return ``None``.
            If ``False`` (default), return a copy of ``adata`` with scaled
            ``.obs``.

        Returns
        -------
        AnnData or None
            Modified copy when ``inplace=False``; ``None`` when ``inplace=True``.
        """
        if not hasattr(self, "scaler_"):
            raise RuntimeError(
                "This AnnDataStandardScaler instance is not fitted yet. "
                "Call fit() before transform()."
            )

        print(
            f"Scaled numerical variables     : "
            f"{', '.join(self.numeric_cols_) if self.numeric_cols_ else '(none)'}"
        )
        print(
            f"Unchanged categorical variables: "
            f"{', '.join(self.categorical_cols_) if self.categorical_cols_ else '(none)'}"
        )

        scaled_values = self.scaler_.transform(adata.obs[self.numeric_cols_])
        scaled_df = pd.DataFrame(
            scaled_values,
            index=adata.obs.index,
            columns=self.numeric_cols_,
        )

        if inplace:
            adata.obs[self.numeric_cols_] = scaled_df
            return None

        new_obs = adata.obs.copy()
        new_obs[self.numeric_cols_] = scaled_df
        result = adata.copy()
        result.obs = new_obs
        return result

    def fit_transform(self, adata: AnnData, inplace: bool = False) -> Optional[AnnData]:
        """Fit the scaler and transform ``adata.obs`` in one step.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix to fit and transform.
        inplace : bool, optional
            Passed through to :meth:`transform`.

        Returns
        -------
        AnnData or None
            See :meth:`transform`.
        """
        return self.fit(adata).transform(adata, inplace=inplace)
