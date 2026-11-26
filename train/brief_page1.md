# A CPU weather twin for monsoon scenarios

Airavat predicts six physical weather fields on a 32 × 36 grid. Three daily input states initialize a residual U-Net that predicts deviations from a fitted seasonal cycle. Static terrain and a land mask are bundled with the model. The CPU Engine steps forward one day at a time and accepts an intervention on the last input state.

## Drift and spatial detail

Multi-step training exposes the network to its own forecasts. The rain head learns in log space with penalties for missed heavy rain and loss of spatial variance. Held-out mean RMSE was 25.7%, 28.1%, and 25.8% below persistence at days 1, 4, and 8 across 721 windows. Day-eight spatial variance ratios on the five event proxies ranged from 0.54 to 1.09.

## Counterfactual dynamics

Branch combines a learned response with wind-driven transport, lateral mixing, and a damped pressure and momentum adjustment. A spatial support limit applies to both response terms on day one. On ten archive diagnostic windows, the kit's propagation check passed for 9 Bay of Bengal shocks and 10 Arabian Sea shocks. All ten zero and subprecision shocks matched their unperturbed forecasts exactly.

## Where the model can fail

The five event proxies scored 28.69 out of 45 for forecast skill. Day-eight mean skill versus the stronger baseline was −2.0%. The model missed observed heavy-rain cells at days four and eight in the winter disturbance proxy. The hidden 2023–2024 cases are unverified. The deployment checkpoint uses the full public archive. Forecast skill is measured on a separate checkpoint fit only on the earlier years.
