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
        
        # Take logs of the main variables (replace with 0.001 if log(0) or log(negative))
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
    
    def _update_parameters_gmm(self):

        # Drop rows with missing values for required variables
        valid_data = self.df.dropna(subset=['eta', 'k_lag', 'l_lag', 'm_lag'])

        if len(valid_data) < 10:
            print("Warning: Not enough valid observations for GMM update")
            return 0, {}

        # Define GMM objective function (sum of squared moment conditions)
        def gmm_objective(params):
            alpha_k, alpha_l, alpha_m = params

            # Recalculate residuals with new parameter estimates
            a_temp = valid_data['y'] - (alpha_k * valid_data['k'] + 
                                       alpha_l * valid_data['l'] + 
                                       alpha_m * valid_data['m'])

            # AR(1) regression to get innovations
            X = sm.add_constant(valid_data['a_lag'])
            ar_model = sm.OLS(a_temp, X)
            ar_results = ar_model.fit()
            rho_temp = ar_results.params.iloc[1]
            eta_temp = a_temp - (ar_results.params.iloc[0] + rho_temp * valid_data['a_lag'])

            # Calculate moment conditions (orthogonality to lagged inputs)
            g1 = eta_temp * valid_data['k_lag']
            g2 = eta_temp * valid_data['l_lag']
            g3 = eta_temp * valid_data['m_lag']

            # Sum of squared moment conditions
            g = np.array([np.mean(g1), np.mean(g2), np.mean(g3)])
            return np.sum(g**2)

        # Initial parameter values
        x0 = np.array([self.alpha_k, self.alpha_l, self.alpha_m])

        bounds = [(0, 100), (0, 100), (0, 100)]

        # Additional constraint: sum of parameters should be close to 1 (constant returns to scale)
        def constraint(params):
            return params.sum() - 1.0

        constraints = {'type': 'eq', 'fun': constraint}

        try:
            # Minimize GMM objective function
            from scipy.optimize import minimize
            result = minimize(gmm_objective, x0, method='SLSQP', 
                              bounds=bounds, constraints=constraints)

            if result.success:
                # Calculate parameter changes
                new_params = result.x
                param_changes = np.sum(np.abs(new_params - x0))

                # Update parameters
                self.alpha_k, self.alpha_l, self.alpha_m = new_params

                # Recalculate productivity with new parameters
                self._calculate_productivity()

                # Update a_lag for next iteration
                self.df['a_lag'] = self.df.groupby(self.id_col)['a'].shift(1)

                # Calculate correlations for reporting
                correlations = self._check_moment_conditions()

                return param_changes, correlations
            else:
                print(f"GMM optimization failed: {result.message}")
                return 0, self._check_moment_conditions()

        except Exception as e:
            print(f"Error in GMM update: {e}")
            return 0, self._check_moment_conditions()
    
    def estimate(self, verbose=True, use_gmm=True):

        # Initialize variables to track iterations
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

            # Update parameters using either GMM or the original correlation method
            if use_gmm:
                param_changes, correlations = self._update_parameters_gmm()
            else:
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

            # Check for convergence - all correlations should be close to zero and parameters should not change much
            if (param_changes < self.tolerance and 
                abs(correlations['k_lag'][0]) < 0.05 and
                abs(correlations['l_lag'][0]) < 0.05 and
                abs(correlations['m_lag'][0]) < 0.05):
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
        productivity_df['rho'] = self.rho

        # Return final parameters and productivity estimates
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