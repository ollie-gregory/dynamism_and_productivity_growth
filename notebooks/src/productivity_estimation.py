import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

class ProductivityEstimator:
    def __init__(self, df, id_col='company_id', year_col='year', 
                 output_col='sale', capital_col='ppent', 
                 labor_col='emp', materials_col='cogs',
                 naics_col='naics',
                 max_iterations=1000, tolerance=1e-6):

        self.df = df.copy()
        self.id_col = id_col
        self.year_col = year_col
        self.output_col = output_col
        self.capital_col = capital_col
        self.labor_col = labor_col
        self.materials_col = materials_col
        self.naics_col = naics_col
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Sort dataframe by firm and year for correct lagging
        self.df = self.df.sort_values([id_col, year_col])
        
        # Take logs of the main variables (replace with 0.00001 if log(0) or log(negative))
        self.df['y'] = np.log(np.maximum(self.df[output_col], 0.001))
        self.df['k'] = np.log(np.maximum(self.df[capital_col], 0.001))
        self.df['l'] = np.log(np.maximum(self.df[labor_col], 0.001))
        self.df['m'] = np.log(np.maximum(self.df[materials_col], 0.001))
        
        # Initialise parameters
        self._initialise_parameters()
        self.rho = 0.8
        
        # Create lagged variables for k, l, m, y
        self._create_lags()
        
    def _initialise_parameters(self, verbose=False):
        # Use OLS to initialise parameters
        try:
            # Check if there's enough data for OLS
            valid_data = self.df[['k', 'l', 'm', 'y']].dropna()

            if len(valid_data) < 5:  # Minimum observations needed for regression
                # Fall back to default values
                self.alpha_k = 0.3
                self.alpha_l = 0.4
                self.alpha_m = 0.3
                return

            X = sm.add_constant(valid_data[['k', 'l', 'm']])
            y = valid_data['y']

            # Run OLS regression
            model = sm.OLS(y, X)
            results = model.fit()

            # Set initial values based on OLS coefficients
            self.alpha_k = max(0.05, min(0.6, results.params['k']))
            self.alpha_l = max(0.05, min(0.7, results.params['l']))
            self.alpha_m = max(0.05, min(0.7, results.params['m']))

            if verbose:
                print(f"Initial parameters from OLS: alpha_k={self.alpha_k:.4f}, alpha_l={self.alpha_l:.4f}, alpha_m={self.alpha_m:.4f}")

        except Exception as e:
            print(f"Error in parameter initialization: {e}")
            print("Using default parameter values.")
            self.alpha_k = 0.3
            self.alpha_l = 0.4
            self.alpha_m = 0.3
    
    def _create_lags(self):
        # Group by firm ID and create lags
        for var in ['k', 'l', 'm', 'y']:
            self.df[f'{var}_lag'] = self.df.groupby(self.id_col)[var].shift(1)
    
    def _calculate_productivity(self):
        # Calculate productivity residuals
        self.df['a'] = self.df['y'] - (
            self.alpha_k * self.df['k'] + 
            self.alpha_l * self.df['l'] + 
            self.alpha_m * self.df['m']
        )
    
    def _estimate_ar_process(self):
        # drop the NaN values for a and a_lag
        temp_df = self.df.dropna(subset=['a', 'a_lag'])

        # Estimate AR(1) process for productivity
        model = sm.OLS(temp_df['a'], sm.add_constant(temp_df['a_lag']))
        results = model.fit()
        self.rho = results.params.iloc[1]

        # Calculate eta for all rows where a_lag exists
        mask = ~self.df['a_lag'].isna()
        self.df.loc[mask, 'eta'] = self.df.loc[mask, 'a'] - (results.params['const'] + self.rho * self.df.loc[mask, 'a_lag'])

        return results

    def _check_moment_conditions(self):
        # drop NaN values for correlation calculations
        temp_df = self.df.dropna(subset=['eta', 'k_lag', 'l_lag', 'm_lag'])

        return {
            'k_lag': stats.pearsonr(temp_df['eta'], temp_df['k_lag']),
            'l_lag': stats.pearsonr(temp_df['eta'], temp_df['l_lag']),
            'm_lag': stats.pearsonr(temp_df['eta'], temp_df['m_lag'])
        }
    
    def _update_parameters(self, step_size=0.01):

        corrs = self._check_moment_conditions()
        
        corr_k = corrs['k_lag'][0]
        corr_l = corrs['l_lag'][0]
        corr_m = corrs['m_lag'][0]
        
        # Update parameters based on the sign and magnitude of correlations
        delta_k = -step_size * corr_k
        delta_l = -step_size * corr_l
        delta_m = -step_size * corr_m
        
        # Ensure sum of coefficients stays within reasonable range (0.8 to 1.2)
        old_sum = self.alpha_k + self.alpha_l + self.alpha_m
        new_sum = old_sum + delta_k + delta_l + delta_m
        
        if new_sum < 0.8 or new_sum > 1.2:
            scaling_factor = old_sum / new_sum
            delta_k *= scaling_factor
            delta_l *= scaling_factor
            delta_m *= scaling_factor
        
        # Update parameters
        self.alpha_k += delta_k
        self.alpha_l += delta_l
        self.alpha_m += delta_m
        
        # Ensure all parameters remain positive
        self.alpha_k = max(0.001, self.alpha_k)
        self.alpha_l = max(0.001, self.alpha_l)
        self.alpha_m = max(0.001, self.alpha_m)
        
        # Calculate parameter changes for convergence check
        param_changes = abs(delta_k) + abs(delta_l) + abs(delta_m)
        return param_changes, corrs
    
    def estimate(self, verbose=False):
        # Initialise variables to track iterations
        iterations = 0
        converged = False
        
        results_history = []
        
        while not converged and iterations < self.max_iterations:
            # Calculate productivity with current parameters
            self._calculate_productivity()
            
            # Update a_lag
            self.df['a_lag'] = self.df.groupby(self.id_col)['a'].shift(1)
            
            # Estimate AR(1) process
            self._estimate_ar_process()
            
            # Update parameters and check convergence
            param_changes, correlations = self._update_parameters()
            
            # Record iteration results
            iter_results = {
                'iteration': iterations + 1,
                'alpha_k': self.alpha_k,
                'alpha_l': self.alpha_l,
                'alpha_m': self.alpha_m,
                'rho': self.rho,
                'corr_k': correlations['k_lag'][0],
                'corr_l': correlations['l_lag'][0],
                'corr_m': correlations['m_lag'][0],
                'p_value_k': correlations['k_lag'][1],
                'p_value_l': correlations['l_lag'][1],
                'p_value_m': correlations['m_lag'][1],
                'param_changes': param_changes
            }
            results_history.append(iter_results)
            
            if verbose and (iterations % 5 == 0 or iterations < 5):
                print(f"Iteration {iterations+1}:")
                print(f"  Parameters: alpha_k={self.alpha_k:.4f}, alpha_l={self.alpha_l:.4f}, alpha_m={self.alpha_m:.4f}, rho={self.rho:.4f}")
                print(f"  Correlations: k_lag={correlations['k_lag'][0]:.4f} (p={correlations['k_lag'][1]:.4f}), "
                      f"l_lag={correlations['l_lag'][0]:.4f} (p={correlations['l_lag'][1]:.4f}), "
                      f"m_lag={correlations['m_lag'][0]:.4f} (p={correlations['m_lag'][1]:.4f})")
                print(f"  Parameter changes: {param_changes:.6f}")
                print()
            
            # Check for convergence - all correlations should be close to zero and parameters should not be changing much
            if (param_changes < self.tolerance or
                (abs(correlations['k_lag'][0]) < 0.05 and
                 abs(correlations['l_lag'][0]) < 0.05 and
                 abs(correlations['m_lag'][0]) < 0.05)):
                converged = True
            
            iterations += 1
        
        # Final calculation of productivity
        self._calculate_productivity()
        
        if verbose:
            if converged:
                print(f"Converged after {iterations} iterations.")
            else:
                print(f"Maximum iterations ({self.max_iterations}) reached without convergence.")
            
            print("\nFinal parameter estimates:")
            print(f"  alpha_k = {self.alpha_k:.4f}")
            print(f"  alpha_l = {self.alpha_l:.4f}")
            print(f"  alpha_m = {self.alpha_m:.4f}")
            print(f"  rho = {self.rho:.4f}")
        
        productivity_df = self.df[[self.id_col, self.year_col, 'a', 'eta']].copy()
        productivity_df['alpha_k'] = self.alpha_k
        productivity_df['alpha_l'] = self.alpha_l
        productivity_df['alpha_m'] = self.alpha_m
        
        results = {
            'params': {
                'alpha_k': self.alpha_k,
                'alpha_l': self.alpha_l,
                'alpha_m': self.alpha_m,
                'rho': self.rho
            },
            'converged': converged,
            'iterations': iterations,
            'history': pd.DataFrame(results_history),
            'productivity': productivity_df
        }
        
        return results
    
    def get_productivity_measures(self):
        return self.df[[self.id_col, self.year_col, 'a', 'eta']].copy()