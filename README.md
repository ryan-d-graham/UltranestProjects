# Bayesian Inverse Geolocation (Compass Resection)

*Locate your own coordinates when all you have is a compass and a clear view of a few known landmarks.*

This small script demonstrates how to turn noisy bearing observations into a **posterior probability cloud** using Bayesian inference and nested sampling. The result is an interactive heat-map that shows *where you probably are*, rather than a single point estimate.

---

## Why it matters

| Use-case | Why bearings only help |
| --- | --- |
| **Search & Rescue** | Find a responder’s location after GPS failure or jamming. |
| **Field surveying** | Quick ground-truth checks without laying out reflectors or total stations. |
| **Wild-life / sensor tracking** | Back-solve the position of a tag that can only broadcast heading. |
| **Backup navigation** | Emergency fixes on land or at sea when every other nav aid fails. |

Anywhere you can see at least two landmarks whose coordinates you know, this method can give you an immediate position estimate—together with its uncertainty.

---

## How the script works (one-minute version)

1. **Bearings → Likelihood**  
   For each landmark, the difference between the *observed* bearing and the *predicted* bearing is treated as a wrapped-normal error  
   \\[
     \Delta\theta \sim \mathcal N(0,\sigma^2)\quad\bmod 360^\circ
   \\]

2. **Loose prior**  
   A uniform rectangle (plus a small margin) is drawn around all chosen landmarks.

3. **Nested sampling with UltraNest**  
   UltraNest explores the latitude-longitude space and returns equally-weighted samples from the posterior distribution.

4. **Visualisation**  
   The posterior samples are pushed into Folium as a heat layer; beacon markers and the maximum-a-posteriori (MAP) point are overlaid.

> More landmarks or tighter compass accuracy → a smaller, sharper probability cloud.

---

## Quick-start

```bash
pip install numpy scipy ultranest geographiclib folium
python bayesian_resection.py

# Landmarks (lat, lon) in decimal degrees
beacons = np.array([
    [lat1, lon1],  # Landmark A
    [lat2, lon2],  # Landmark B
    ...
])

# Your measured bearings (degrees), same order as 'beacons'
theta_data = np.array([bearingA, bearingB, ...])

sigma = 10.0  # Compass 1-σ noise


