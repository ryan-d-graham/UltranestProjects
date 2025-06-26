#!/usr/bin/env python3
"""
Bayesian Resectioning with Observer‑Relative Bearings and Interactive Web Map

Requires:
    pip install numpy scipy ultranest geographiclib folium
"""

import os
import shutil
import threading
import http.server
import socketserver
import webbrowser

import numpy as np
from geographiclib.geodesic import Geodesic
import ultranest as un
import scipy.stats as st
import folium
from folium.plugins import HeatMap
from folium.raster_layers import WmsTileLayer

# Toggles: Control what markers to display.
plot_gt_loc   = False   # Plot the true (measured) observer location.
plot_mean_loc = False    # Plot the posterior mean location.
plot_map_loc  = True    # Plot the maximum a posteriori (MAP) location.

# 1. Beacon positions in decimal degrees (lat, lon)
beacons = np.array([
    [32.401561, -100.714005],  # Wolf stadium
    [32.412594, -100.713310],  # Munoz trucking
    [32.40361,  -100.71279],   # Water tower
    [32.41195,  -100.71327],   # Radio tower
])
# Select which beacons to use (by index)
beacons_trunc = beacons[[2, 3], :]  # currently using the water tower & radio tower

# 2. True observer location (for simulation/testing)
x_gt = np.array([32.408577, -100.721543])  # Mrs G’s

# 3. Geodesic bearing function (observer → beacon)
geod = Geodesic.WGS84
def bearing(lat_lon, beacons):
    """
    Compute forward azimuths (°) from the observer (lat_lon) to each beacon.
    Returns a NumPy array of angles wrapped to [0,360).
    """
    lat, lon = lat_lon
    az = [geod.Inverse(lat, lon, b_lat, b_lon)['azi1']
          for b_lat, b_lon in beacons]
    az = np.array(az)
    return np.mod(az + 360, 360)

# 4. Simulate noisy compass readings (observer‐relative bearings)
sigma = 10.0  # Compass noise in degrees.
theta_true = bearing(x_gt, beacons_trunc)
# For testing, use the measured bearings from your military compass:
theta_data = np.array([115, 60])
theta_data = np.mod(theta_data, 360)
print("Observed bearings (°):", np.round(theta_data, 2), "\n")

# 5. Automatic prior bounds around the beacons (since we're using field measurements)
margin = 0.05  # ~5 km buffer
all_lats = beacons_trunc[:, 0]
all_lons = beacons_trunc[:, 1]
lat_lo, lat_hi = all_lats.min() - margin, all_lats.max() + margin
lon_lo, lon_hi = all_lons.min() - margin, all_lons.max() + margin
print(f"Sampling latitude  in [{lat_lo:.5f}, {lat_hi:.5f}]")
print(f"Sampling longitude in [{lon_lo:.5f}, {lon_hi:.5f}]\n")

# 6. Prior transform & Likelihood
def PriorTransform(cube):
    lat = cube[0] * (lat_hi - lat_lo) + lat_lo
    lon = cube[1] * (lon_hi - lon_lo) + lon_lo
    return np.array([lat, lon])

def LogLikelihood(params):
    theta_pred = bearing(params, beacons_trunc)
    delta = (theta_data - theta_pred + 180) % 360 - 180
    return -0.5 * np.sum((delta / sigma) ** 2) - len(delta) * 0.5 * np.log(2 * np.pi * sigma ** 2)

# 7. Reset UltraNest log_dir to force a fresh run.
log_dir = 'ultranest_tx_run'
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

# 8. Run Nested Sampling
sampler = un.NestedSampler(
    ["lat", "lon"], LogLikelihood, PriorTransform,
    num_live_points=200,
    log_dir=log_dir,
    resume='overwrite'
)
results = sampler.run()

# 9. Extract posterior samples (equally weighted)
samples = results['samples']  # shape (n_samples, 2)
lat_samps = samples[:, 0]
lon_samps = samples[:, 1]
print(f"Posterior mean → lat = {lat_samps.mean():.5f}, lon = {lon_samps.mean():.5f}\n")

# Compute the MAP location using a kernel density estimate.
kde = st.gaussian_kde(np.vstack([lon_samps, lat_samps]))
densities = kde(np.vstack([lon_samps, lat_samps]))
map_index = np.argmax(densities)
map_lat = lat_samps[map_index]
map_lon = lon_samps[map_index]
print(f"MAP location → lat = {map_lat:.5f}, lon = {map_lon:.5f}\n")

# 10. Build interactive Folium map
center = [lat_samps.mean(), lon_samps.mean()]
m = folium.Map(
    location=center,
    zoom_start=14,
    tiles='OpenStreetMap',
    attr='© OpenStreetMap contributors'
)

# Additional base layers
folium.TileLayer('Stamen Toner', name='Toner').add_to(m)
folium.TileLayer('CartoDB positron', name='Positron').add_to(m)

# USGS Topo WMS overlay
WmsTileLayer(
    url='https://basemap.nationalmap.gov/arcgis/services/USGSTopo/MapServer/WMSServer',
    name='USGS Topo',
    layers='0',
    fmt='image/png',
    transparent=True,
    attribution='USGS'
).add_to(m)

# Posterior heatmap layer
HeatMap(
    list(zip(lat_samps, lon_samps)),
    radius=6,
    blur=12,
    name='Posterior Heat'
).add_to(m)

# Beacon markers
for lat, lon in beacons_trunc:
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color='blue', icon='flag'),
        popup=f'Beacon ({lat:.5f}, {lon:.5f})'
    ).add_to(m)

# True location marker
if plot_gt_loc:
    folium.CircleMarker(
        location=x_gt.tolist(),
        radius=6,
        color='red',
        fill=True,
        fill_color='red',
        popup='True Location'
    ).add_to(m)

# Posterior Mean location marker
if plot_mean_loc:
    mean_lat = lat_samps.mean()
    mean_lon = lon_samps.mean()
    folium.CircleMarker(
        location=[mean_lat, mean_lon],
        radius=5,
        color='black',
        fill=True,
        fill_color='black',
        popup='Mean Location'
    ).add_to(m)

# MAP location marker
if plot_map_loc:
    folium.CircleMarker(
        location=[map_lat, map_lon],
        radius=5,
        color='purple',
        fill=True,
        fill_color='purple',
        popup='MAP Location'
    ).add_to(m)

# Layer control and save map
folium.LayerControl(collapsed=False).add_to(m)
m.save('observer_heatmap.html')
print("Interactive map saved to observer_heatmap.html")

# 11. Serve the map over HTTP to avoid Edge file:// errors
def serve_map(port=8000):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f"Serving at http://localhost:{port}/observer_heatmap.html")
        httpd.serve_forever()

threading.Thread(target=serve_map, daemon=True).start()
webbrowser.open('http://localhost:8000/observer_heatmap.html')
