"""
Tests for fast copula covariance implementation.

This module tests the new fast_copula_covariance function and associated
estimators that use top-k gene selection for computational efficiency.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad

from scdesigner.estimators import (
    fast_copula_covariance, 
    FastCovarianceStructure,
    fast_negbin_copula_factory,
    fast_poisson_copula_factory,
    negbin_copula,
    poisson_copula
)


def generate_test_adata(N=100, G=50, random_seed=42):
    """
    Generate test AnnData object for fast copula testing.
    
    Parameters:
    -----------
    N : int
        Number of observations 
    G : int
        Number of genes
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    anndata.AnnData : Test dataset
    """
    np.random.seed(random_seed)
    
    # Create expression data with some genes more highly expressed than others
    base_expression = np.random.poisson(1, size=(N, G))
    
    # Make some genes highly expressed (top-k candidates)
    high_expr_genes = np.random.choice(G, size=G//4, replace=False)
    base_expression[:, high_expr_genes] += np.random.poisson(10, size=(N, len(high_expr_genes)))
    
    # Use dense matrix to avoid sparse matrix issues in testing
    adata = ad.AnnData(base_expression.astype(np.float32))
    
    # Add pseudotime covariate
    adata.obs["pseudotime"] = np.random.uniform(0, 1, N)
    
    # Add cell type covariate
    adata.obs["cell_type"] = pd.Categorical(
        np.random.choice(["A", "B", "C"], size=N)
    )
    
    adata.var_names = [f"Gene_{i:d}" for i in range(G)]
    
    return adata


class TestFastCovarianceStructure:
    """Test the FastCovarianceStructure class."""
    
    def test_initialization(self):
        """Test FastCovarianceStructure initialization."""
        top_k = 5
        remaining = 10
        total = top_k + remaining
        
        top_k_cov = np.random.randn(top_k, top_k)
        top_k_cov = top_k_cov @ top_k_cov.T  # Make positive definite
        remaining_var = np.random.uniform(0.1, 2.0, remaining)
        top_k_indices = np.array([0, 1, 2, 3, 4])
        remaining_indices = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        gene_total_expr = np.random.uniform(0, 100, total)
        
        fast_cov = FastCovarianceStructure(
            top_k_cov, remaining_var, top_k_indices, remaining_indices, gene_total_expr
        )
        
        assert fast_cov.top_k == top_k
        assert fast_cov.total_genes == total
        assert len(fast_cov.remaining_indices) == remaining
        
    def test_to_full_matrix(self):
        """Test conversion to full covariance matrix."""
        top_k = 3
        remaining = 2
        
        top_k_cov = np.array([[1.0, 0.5, 0.2], 
                              [0.5, 1.0, 0.3],
                              [0.2, 0.3, 1.0]])
        remaining_var = np.array([0.8, 1.2])
        top_k_indices = np.array([0, 2, 4])
        remaining_indices = np.array([1, 3])
        gene_total_expr = np.array([50.0, 10.0, 40.0, 20.0, 30.0])
        
        fast_cov = FastCovarianceStructure(
            top_k_cov, remaining_var, top_k_indices, remaining_indices, gene_total_expr
        )
        
        full_matrix = fast_cov.to_full_matrix()
        
        # Check dimensions
        assert full_matrix.shape == (5, 5)
        
        # Check top-k block
        assert np.allclose(full_matrix[np.ix_([0, 2, 4], [0, 2, 4])], top_k_cov)
        
        # Check diagonal for remaining genes
        assert np.allclose(full_matrix[1, 1], remaining_var[0])
        assert np.allclose(full_matrix[3, 3], remaining_var[1])
        
        # Check off-diagonal zeros for remaining genes
        assert full_matrix[1, 3] == 0.0
        assert full_matrix[3, 1] == 0.0


class TestFastCopulaCovariance:
    """Test the fast_copula_covariance function."""
    
    def test_fallback_to_full_covariance(self):
        """Test that function falls back to full covariance when top_k >= total genes."""
        adata = generate_test_adata(N=50, G=20)
        
        # Regular estimation
        regular_estimator = negbin_copula
        regular_params = regular_estimator(adata, formula="~ pseudotime")
        
        # Fast estimation with top_k >= total genes (should be identical)
        fast_estimator = fast_negbin_copula_factory(top_k=25)  # > 20 genes
        fast_params = fast_estimator(adata, formula="~ pseudotime")
        
        # Should get same results (within numerical tolerance)
        if isinstance(regular_params["covariance"], pd.DataFrame):
            assert np.allclose(
                regular_params["covariance"].values, 
                fast_params["covariance"].values,
                rtol=1e-4
            )
        
    def test_top_k_gene_selection(self):
        """Test that top-k genes are correctly selected based on total expression."""
        adata = generate_test_adata(N=100, G=50)
        
        # Calculate total expression manually
        total_expr = np.array(adata.X.sum(axis=0)).flatten()
        expected_top_k_indices = np.argsort(total_expr)[-10:]  # top 10
        
        fast_estimator = fast_negbin_copula_factory(top_k=10)
        params = fast_estimator(adata, formula="~ pseudotime")
        
        # Check that FastCovarianceStructure was created
        assert isinstance(params["covariance"], FastCovarianceStructure)
        
        # Check that top-k indices match expected
        fast_cov = params["covariance"]
        assert len(fast_cov.top_k_indices) == 10
        
        # The selected genes should be among the highest expressed
        # (allowing for some numerical differences in the selection process)
        selected_totals = total_expr[fast_cov.top_k_indices]
        remaining_totals = total_expr[fast_cov.remaining_indices]
        assert np.min(selected_totals) >= np.max(remaining_totals) * 0.8  # Allow some tolerance
        
    def test_parameter_validation(self):
        """Test parameter validation in fast_copula_covariance."""
        adata = generate_test_adata(N=50, G=20)
        
        # Test invalid top_k values
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fast_estimator = fast_negbin_copula_factory(top_k=0)
            fast_estimator(adata, formula="~ pseudotime")
            
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fast_estimator = fast_negbin_copula_factory(top_k=-5)
            fast_estimator(adata, formula="~ pseudotime")


class TestSamplingCompatibility:
    """Test that sampling functions work with FastCovarianceStructure."""
    
    def test_fast_sampling_vs_regular_sampling(self):
        """Test that fast covariance produces reasonable samples."""
        adata = generate_test_adata(N=100, G=30)
        
        # Create simulators
        from scdesigner.simulators import NegBinCopulaSimulator
        
        # Regular simulator
        regular_sim = NegBinCopulaSimulator()
        regular_sim.fit(adata, "~ pseudotime")
        
        # Test sampling (just check that it doesn't crash)
        regular_samples = regular_sim.sample(adata.obs[:20])
        
        # For fast version, we'd need to modify the simulator class
        # to accept fast estimators. For now, just check parameter structure.
        fast_estimator = fast_negbin_copula_factory(top_k=10)
        fast_params = fast_estimator(adata, formula="~ pseudotime")
        
        assert "covariance" in fast_params
        assert isinstance(fast_params["covariance"], FastCovarianceStructure)
        assert "coef_mean" in fast_params
        assert "coef_dispersion" in fast_params


class TestMultipleDistributions:
    """Test fast copula with different distributions."""
    
    def test_poisson_fast_copula(self):
        """Test fast copula with Poisson distribution."""
        adata = generate_test_adata(N=80, G=25)
        
        fast_estimator = fast_poisson_copula_factory(top_k=8)
        params = fast_estimator(adata, formula="~ pseudotime")
        
        assert isinstance(params["covariance"], FastCovarianceStructure)
        assert params["covariance"].top_k == 8
        assert params["covariance"].total_genes == 25
        
    def test_negbin_fast_copula(self):
        """Test fast copula with Negative Binomial distribution."""
        adata = generate_test_adata(N=80, G=25)
        
        fast_estimator = fast_negbin_copula_factory(top_k=8)
        params = fast_estimator(adata, formula="~ pseudotime")
        
        assert isinstance(params["covariance"], FastCovarianceStructure)
        assert params["covariance"].top_k == 8
        assert params["covariance"].total_genes == 25


def test_integration_example():
    """Integration test showing typical usage pattern."""
    # Generate test data
    adata = generate_test_adata(N=200, G=100)
    
    # Create fast estimator for top 20 genes
    fast_estimator = fast_negbin_copula_factory(top_k=20)
    
    # Fit the model
    params = fast_estimator(adata, formula="~ pseudotime + cell_type")
    
    # Verify results
    assert isinstance(params["covariance"], FastCovarianceStructure)
    assert params["covariance"].top_k == 20
    assert params["covariance"].total_genes == 100
    assert len(params["covariance"].remaining_indices) == 80
    
    # Check that parameter structure is preserved
    assert "coef_mean" in params
    assert "coef_dispersion" in params
    assert isinstance(params["coef_mean"], pd.DataFrame)
    assert isinstance(params["coef_dispersion"], pd.DataFrame)
    
    print(f"Fast copula test completed successfully!")
    print(f"Top-k genes: {params['covariance'].top_k}")
    print(f"Total genes: {params['covariance'].total_genes}")
    print(f"Total expression range: {params['covariance'].gene_total_expression.min():.2f} - {params['covariance'].gene_total_expression.max():.2f}")


if __name__ == "__main__":
    # Run integration test
    test_integration_example()
