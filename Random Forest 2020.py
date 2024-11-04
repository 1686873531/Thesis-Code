import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

total_consumption_2020 = {
    'January': np.array([3521, 5609, 5261, 2987, 2415, 5859, 5467, 5376, 5295, 5665, 4335, 2747, 4920, 5113, 5880, 5174, 5421, 3210, 4799, 6880, 6292, 6846, 6946, 7285, 5034, 4564, 5699, 6524, 6600, 6351, 5888]),
    'February': np.array([3925, 4804, 7150, 5430, 4710, 4449, 5106, 3044, 2586, 5932, 6078, 5349, 4891, 5200, 3837, 4568, 6297, 6787, 6846, 5760, 5692, 2529, 2548, 5652, 5926, 6663, 6042, 4556, 3659]),
    'March': np.array([4058, 5091, 5148, 5238, 5236, 5142, 4297, 4182, 5042, 5093, 5171, 5176, 5026, 4174, 4125, 4810, 4717, 4714, 4818, 4639, 4134, 4133, 4853, 4830, 4759, 4662, 4466, 3734, 3766, 4235, 4120]),
    'April': np.array([4175, 4214, 4102, 3652, 3532, 3844, 3787, 3896, 3811, 3689, 3578, 3462, 3661, 3766, 3877, 3912, 3934, 3676, 3632, 3962, 3867, 3935, 3913, 3824, 3664, 3568, 3961, 4033, 4172, 4241]),
    'May': np.array([4361, 3941, 3919, 4036, 4387, 4396, 4509, 4459, 4147, 4107, 4660, 4518, 6643, 7346, 1693, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 7161, 6741]),
    'June': np.array([7610, 6223, 4778, 4804, 4694, 4694, 4332, 5112, 4433, 3005, 2967, 2948, 2724, 2734, 2921, 3002, 3087, 2983, 2949, 2696, 2702, 2947, 3050, 3106, 3298, 3123, 2696, 2699, 3193, 3218]),
    'July': np.array([3103, 3302, 3152, 2502, 2618, 3482, 3563, 3241, 3113, 3118, 2730, 2679, 3004, 3357, 3182, 3137, 3002, 2552, 2601, 3088, 2913, 2951, 3156, 3139, 2734, 2691, 3021, 2974, 3074, 3057, 2972]),
    'August': np.array([2980, 2912, 3150, 3111, 2982, 3080, 3004, 2625, 2508, 3331, 3204, 3220, 3324, 3118, 2705, 2499, 2909, 2919, 2871, 2863, 2814, 2526, 2565, 3138, 4129, 4995, 4557, 3958, 3577, 3611, 3937]),
    'September': np.array([4049, 4039, 4113, 4041, 3847, 3892, 4059, 4208, 4250, 4191, 4006, 3624, 3884, 4110, 4235, 4257, 4300, 4081, 3902, 3890, 4021, 6009, 5381, 4577, 5297, 3706, 3630, 3793, 3770, 3829]),
    'October': np.array([3980, 4398, 3784, 3763, 3733, 4718, 4596, 4660, 6389, 3776, 3783, 4797, 4664, 3937, 4315, 4854, 3754, 3745, 4965, 5150, 5023, 4955, 5317, 4745, 3764, 5788, 5232, 5074, 5189, 5141, 3978]),
    'November': np.array([3694, 5641, 5002, 4940, 6124, 4629, 4790, 3862, 4467, 4677, 4470, 4971, 5468, 4844, 6229, 7303, 6989, 5863, 5602, 5766, 3749, 3981, 6880, 6153, 5417, 4893, 5802, 4544, 4653, 6899]),
    'December': np.array([8495, 7813, 7239, 7394, 5679, 5907, 8609, 5302, 4643, 4831, 4334, 2826, 2831, 4887, 6979, 7107, 6965, 6417, 4229, 4478, 6656, 6603, 2286, 1469, 1258, 1273, 302, 849, 765, 987, 1100])
}

# Historical total energy consumption data (kWh)
total_consumption_2020 = {
    'January': np.array([3521, 5609, 5261, 2987, 2415, 5859, 5467, 5376, 5295, 5665, 4335, 2747, 4920, 5113, 5880, 5174, 5421, 3210, 4799, 6880, 6292, 6846, 6946, 7285, 5034, 4564, 5699, 6524, 6600, 6351, 5888]),
    'February': np.array([3925, 4804, 7150, 5430, 4710, 4449, 5106, 3044, 2586, 5932, 6078, 5349, 4891, 5200, 3837, 4568, 6297, 6787, 6846, 5760, 5692, 2529, 2548, 5652, 5926, 6663, 6042, 4556, 3659]),
    'March': np.array([4058, 5091, 5148, 5238, 5236, 5142, 4297, 4182, 5042, 5093, 5171, 5176, 5026, 4174, 4125, 4810, 4717, 4714, 4818, 4639, 4134, 4133, 4853, 4830, 4759, 4662, 4466, 3734, 3766, 4235, 4120]),
    'April': np.array([4175, 4214, 4102, 3652, 3532, 3844, 3787, 3896, 3811, 3689, 3578, 3462, 3661, 3766, 3877, 3912, 3934, 3676, 3632, 3962, 3867, 3935, 3913, 3824, 3664, 3568, 3961, 4033, 4172, 4241]),
    'May': np.array([4361, 3941, 3919, 4036, 4387, 4396, 4509, 4459, 4147, 4107, 4660, 4518, 6643, 7346, 1693, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 5017, 7161, 6741]),
    'June': np.array([7610, 6223, 4778, 4804, 4694, 4694, 4332, 5112, 4433, 3005, 2967, 2948, 2724, 2734, 2921, 3002, 3087, 2983, 2949, 2696, 2702, 2947, 3050, 3106, 3298, 3123, 2696, 2699, 3193, 3218]),
    'July': np.array([3103, 3302, 3152, 2502, 2618, 3482, 3563, 3241, 3113, 3118, 2730, 2679, 3004, 3357, 3182, 3137, 3002, 2552, 2601, 3088, 2913, 2951, 3156, 3139, 2734, 2691, 3021, 2974, 3074, 3057, 2972]),
    'August': np.array([2980, 2912, 3150, 3111, 2982, 3080, 3004, 2625, 2508, 3331, 3204, 3220, 3324, 3118, 2705, 2499, 2909, 2919, 2871, 2863, 2814, 2526, 2565, 3138, 4129, 4995, 4557, 3958, 3577, 3611, 3937]),
    'September': np.array([4049, 4039, 4113, 4041, 3847, 3892, 4059, 4208, 4250, 4191, 4006, 3624, 3884, 4110, 4235, 4257, 4300, 4081, 3902, 3890, 4021, 6009, 5381, 4577, 5297, 3706, 3630, 3793, 3770, 3829]),
    'October': np.array([3980, 4398, 3784, 3763, 3733, 4718, 4596, 4660, 6389, 3776, 3783, 4797, 4664, 3937, 4315, 4854, 3754, 3745, 4965, 5150, 5023, 4955, 5317, 4745, 3764, 5788, 5232, 5074, 5189, 5141, 3978]),
    'November': np.array([3694, 5641, 5002, 4940, 6124, 4629, 4790, 3862, 4467, 4677, 4470, 4971, 5468, 4844, 6229, 7303, 6989, 5863, 5602, 5766, 3749, 3981, 6880, 6153, 5417, 4893, 5802, 4544, 4653, 6899]),
    'December': np.array([8495, 7813, 7239, 7394, 5679, 5907, 8609, 5302, 4643, 4831, 4334, 2826, 2831, 4887, 6979, 7107, 6965, 6417, 4229, 4478, 6656, 6603, 2286, 1469, 1258, 1273, 302, 849, 765, 987, 1100])
}


# simulated outdoor temperature data for Brisbane
outdoor_temperature = {
    'January': [30 + np.random.uniform(-1, 1) for _ in range(31)],  
    'February': [30 + np.random.uniform(-1, 1) for _ in range(29)],
    'March': [28 + np.random.uniform(-1, 1) for _ in range(31)],
    'April': [25 + np.random.uniform(-1, 1) for _ in range(30)],
    'May': [22 + np.random.uniform(-1, 1) for _ in range(31)],
    'June': [20 + np.random.uniform(-1, 1) for _ in range(30)],  
    'July': [18 + np.random.uniform(-1, 1) for _ in range(31)],
    'August': [20 + np.random.uniform(-1, 1) for _ in range(31)],
    'September': [24 + np.random.uniform(-1, 1) for _ in range(30)],
    'October': [27 + np.random.uniform(-1, 1) for _ in range(31)],
    'November': [29 + np.random.uniform(-1, 1) for _ in range(30)],
    'December': [31 + np.random.uniform(-1, 1) for _ in range(31)]
}

# Humidity levels for each month
humidity_levels = {
    'January': [75 + np.random.uniform(-2, 3) for _ in range(31)],  
    'February': [78 + np.random.uniform(-2, 3) for _ in range(29)],
    'March': [72 + np.random.uniform(-2, 3) for _ in range(31)],
    'April': [65 + np.random.uniform(-3, 3) for _ in range(30)],
    'May': [60 + np.random.uniform(-3, 3) for _ in range(31)],
    'June': [55 + np.random.uniform(-3, 3) for _ in range(30)],   
    'July': [52 + np.random.uniform(-3, 3) for _ in range(31)],
    'August': [55 + np.random.uniform(-3, 3) for _ in range(31)],
    'September': [60 + np.random.uniform(-3, 3) for _ in range(30)],
    'October': [65 + np.random.uniform(-3, 3) for _ in range(31)],
    'November': [70 + np.random.uniform(-3, 3) for _ in range(30)],
    'December': [75 + np.random.uniform(-3, 3) for _ in range(31)]
}

# Air conditioner usage percentage for each month
seasonal_ac_percentage = {
    'January': [0.60 + np.random.uniform(-0.03, 0.03) for _ in range(31)],  
    'February': [0.60 + np.random.uniform(-0.03, 0.03) for _ in range(29)],
    'March': [0.50 + np.random.uniform(-0.03, 0.03) for _ in range(31)],
    'April': [0.40 + np.random.uniform(-0.03, 0.03) for _ in range(30)],
    'May': [0.30 + np.random.uniform(-0.03, 0.03) for _ in range(31)],
    'June': [0.20 + np.random.uniform(-0.03, 0.03) for _ in range(30)],  
    'July': [0.20 + np.random.uniform(-0.03, 0.03) for _ in range(31)],
    'August': [0.25 + np.random.uniform(-0.03, 0.03) for _ in range(31)],
    'September': [0.35 + np.random.uniform(-0.03, 0.03) for _ in range(30)],
    'October': [0.45 + np.random.uniform(-0.03, 0.03) for _ in range(31)],
    'November': [0.55 + np.random.uniform(-0.03, 0.03) for _ in range(30)],
    'December': [0.60 + np.random.uniform(-0.03, 0.03) for _ in range(31)]
}

occupancy_data = {
    'January': np.array([0.80 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'February': np.array([0.75 + np.random.uniform(-0.05, 0.05) for _ in range(29)]),
    'March': np.array([0.70 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'April': np.array([0.60 + np.random.uniform(-0.05, 0.05) for _ in range(30)]),
    'May': np.array([0.50 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'June': np.array([0.40 + np.random.uniform(-0.05, 0.05) for _ in range(30)]),
    'July': np.array([0.35 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'August': np.array([0.45 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'September': np.array([0.60 + np.random.uniform(-0.05, 0.05) for _ in range(30)]),
    'October': np.array([0.70 + np.random.uniform(-0.05, 0.05) for _ in range(31)]),
    'November': np.array([0.75 + np.random.uniform(-0.05, 0.05) for _ in range(30)]),
    'December': np.array([0.80 + np.random.uniform(-0.05, 0.05) for _ in range(31)])
}

# Temperature set points for each month
monthly_temp_set_points = {
    'January': 24,
    'February': 24,
    'March': 23,
    'April': 22,
    'May': 22,
    'June': 21,
    'July': 21,
    'August': 22,
    'September': 23,
    'October': 24,
    'November': 24,
    'December': 24
}

def calculate_temp_factor(temp_set_point, outdoor_temp, humidity):
    # Calculate the influence coefficient of temperature difference
    temp_diff = outdoor_temp - temp_set_point
    if temp_diff > 0:
        temp_factor = 1 - 0.0002 * temp_diff
    else:
        temp_factor = 1 + 0.0002 * abs(temp_diff)

    # Calculate the humidity influence coefficient
    if humidity > 50:
        humidity_factor = 1 - 0.0005 * (humidity - 50)
    else:
        humidity_factor = 1 + 0.0005 * (50 - humidity)

    return temp_factor * humidity_factor

# Data standardization
scaler = StandardScaler()

# Initializing PCA dimensionality reduction
pca = PCA(n_components=3)  # three important principal components

# The random forest model was trained with PCA dimensionality reduction and fixed parameters
def train_consumption_model(consumption_data, temperature_data, humidity_data, ac_percentage, days, occupancy_level):
    X = np.array([temperature_data, humidity_data, ac_percentage, days, occupancy_level]).T

    # Print the input data shape
    print(f"Training data shape: {X.shape}")
    
    # Data standardization
    X_scaled = scaler.fit_transform(X)
    y = np.array(consumption_data)

    # Dimensionality reduction using PCA
    X_pca = pca.fit_transform(X_scaled)

    # Random forest model with fixed parameters
    model = RandomForestRegressor(random_state=42)

    # Perform a simplified grid search
    param_grid = {
        'n_estimators': [25],   # Only use a fixed number of trees 25
        'max_depth': [10, 15],   # The depth of the tree is chosen 10 or 15
    }

    # Debug the output to see the number of parameter combinations for the grid search
    print(f"Parameter grid size: {len(param_grid['n_estimators']) * len(param_grid['max_depth'])}")

    # Cross validation is set to 3 to reduce computation
    grid_search = GridSearchCV(model, param_grid, cv=3)

    print("Starting grid search with simplified parameter grid...")  # debug
    
    try:
        grid_search.fit(X_pca, y)
        print(f"Grid search completed successfully with best params: {grid_search.best_params_}")
    except Exception as e:
        print(f"Grid search failed: {e}")
    
    # Returns the tuned model
    return grid_search.best_estimator_

# Prediction function: Make predictions using data that has been dimensionally reduced by PCA
def simulate_consumption(model, temp, humidity, ac_percentage, day, occupancy_level):
    X_simulate = np.array([temp, humidity, ac_percentage, day, occupancy_level]).reshape(1, -1)
    X_scaled = scaler.transform(X_simulate)

    # Data dimensionality was reduced using PCA
    X_pca = pca.transform(X_scaled)

    print(f"Simulating for day {day} with temperature {temp} and occupancy {occupancy_level}")  # debug
    return model.predict(X_pca)[0]

# Main cycle: Simulation and validation on a monthly basis 

total_yearly_original_simulated = 0
total_yearly_adjusted_simulated = 0
total_yearly_savings = 0
real_consumptions = []  
simulated_consumptions = []  

# Suppose you use loaded monthly data for energy consumption projections
for month in total_consumption_2020:
    print(f"Simulating for {month}:")
    days = np.arange(1, len(total_consumption_2020[month]) + 1)
    
    # Training model
    model = train_consumption_model(total_consumption_2020[month], 
                                    outdoor_temperature[month], 
                                    humidity_levels[month], 
                                    seasonal_ac_percentage[month], 
                                    days, 
                                    occupancy_data[month])

    total_original_simulated = 0
    total_adjusted_simulated = 0
    total_savings = 0

    for i, real_consumption in enumerate(total_consumption_2020[month]):
        # Raw power consumption simulation
        simulated_value = simulate_consumption(model, 
                                               outdoor_temperature[month][i], 
                                               humidity_levels[month][i], 
                                               seasonal_ac_percentage[month][i], 
                                               i + 1, 
                                               occupancy_data[month][i])
        total_original_simulated += simulated_value

        # Calculate the adjusted power consumption
        temp_factor = calculate_temp_factor(monthly_temp_set_points[month], outdoor_temperature[month][i], humidity_levels[month][i])
        adjusted_simulated_value = simulated_value * temp_factor
        total_adjusted_simulated += adjusted_simulated_value

        # Calculate the amount of electricity saved
        savings = simulated_value - adjusted_simulated_value
        total_savings += savings

        total_yearly_original_simulated += total_original_simulated
        total_yearly_adjusted_simulated += total_adjusted_simulated
        total_yearly_savings += total_savings

        # Save real and simulated values for later error analysis
        real_consumptions.append(real_consumption)
        simulated_consumptions.append(simulated_value)

        # Output a comparison of the original simulation, the adjusted simulation, and the actual data
        print(f"Day {i+1}: Real = {real_consumption:.2f} kWh, Original Simulated = {simulated_value:.2f} kWh, Adjusted Simulated = {adjusted_simulated_value:.2f} kWh, Savings = {savings:.2f} kWh")
        time.sleep(0.1)
    savings_percentage = (total_savings / total_original_simulated) * 100
    print(f"\nTotal original simulated for {month}: {total_original_simulated:.2f} kWh")
    print(f"Total adjusted simulated for {month}: {total_adjusted_simulated:.2f} kWh")
    print(f"Total savings for {month}: {total_savings:.2f} kWh ({savings_percentage:.2f}% savings)\n")

# Calculate the annual savings percentage
yearly_savings_percentage = (total_yearly_savings / total_yearly_original_simulated) * 100
print(f"\nTotal original simulated for the year: {total_yearly_original_simulated:.2f} kWh")
print(f"Total adjusted simulated for the year: {total_yearly_adjusted_simulated:.2f} kWh")
print(f"Total savings for the year: {total_yearly_savings:.2f} kWh ({yearly_savings_percentage:.2f}% savings)")

# error analysis
mae = mean_absolute_error(real_consumptions, simulated_consumptions)
mse = mean_squared_error(real_consumptions, simulated_consumptions)
print(f"Mean Absolute Error (MAE): {mae:.2f} kWh")
print(f"Mean Squared Error (MSE): {mse:.2f} kWh^2")

# Plot real vs simulated consumption
plt.figure(figsize=(10, 6))
plt.plot(real_consumptions, label='Real Consumption (2020)', color='blue', marker='.')
plt.plot(simulated_consumptions, label='RandomForestRegression Simulated Consumption', color='red', linestyle='--', marker='.')
plt.title('Comparison of Real vs RandomForestRegression Simulated Electricity Consumption (2020)')
plt.xlabel('Days')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# total_simulated_consumption = np.sum(simulated_consumptions)  # Sum of all simulated consumption
# total_savings = total_simulated_consumption - np.sum(real_consumptions)  # Difference between simulated and real gives savings

# plt.figure(figsize=(6, 6))
# plt.pie([total_yearly_original_simulated - total_yearly_savings, total_yearly_savings], 
#         labels=['Simulated Consumption', 'Savings'], 
#         autopct='%1.1f%%', 
#         colors=['blue', 'red'])
# plt.title('Annual Savings as a Percentage of Total Simulated Consumption')
# plt.show()