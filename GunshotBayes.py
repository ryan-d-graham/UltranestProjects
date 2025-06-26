import numpy as np
from geographiclib.geodesic import Geodesic
import ultranest as un
from scipy.stats import gaussian_kde
import folium
from folium.plugins import HeatMap
import threading, http.server, socketserver, webbrowser

# ---------------------------
# 1.  Domain‑level constants
# ---------------------------
SOUND_SPEED = 343.0      # m/s  (assumed known & constant)
SIGMA_T     = 1.0e-4     # s    (100 µs timing error)

# ---------------------------
# 2.  Helper functions
# ---------------------------
WGS84 = Geodesic.WGS84  # ellipsoid object (module‑level reuse)

def geodesic_distance(coord_a, coord_b):
    """Vincenty distance on WGS‑84 ellipsoid (metres)."""
    lat_a, lon_a = coord_a
    lat_b, lon_b = coord_b
    return WGS84.Inverse(lat_a, lon_a, lat_b, lon_b)["s12"]


def travel_times(source, mic_positions, c=SOUND_SPEED):
    """Return one propagation time per microphone (seconds)."""
    return np.array([
        geodesic_distance(source, mic) / c for mic in mic_positions
    ])


def predict_delay_matrix(source, mic_positions, c=SOUND_SPEED):
    """Skew‑symmetric matrix of predicted TDOAs (s)."""
    t = travel_times(source, mic_positions, c)
    return t[:, None] - t[None, :]


# ---------------------------
# 3.  Test geometry & synthetic data
# ---------------------------
# Microphone array (roughly 200 m square in downtown area)
MIC_POS = np.array([
    [41.8800, -87.6300],  # SW corner
    [41.8800, -87.6280],  # SE
    [41.8820, -87.6280],  # NE
    [41.8820, -87.6300],  # NW
])
N = MIC_POS.shape[0]

# Ground‑truth source (inside the square, slightly north‑east)
TRUE_SRC = np.array([41.8810, -87.6290])

# Build the noise‑free delay matrix and inject i.i.d. Gaussian noise
np.random.seed(5)
D_clean = predict_delay_matrix(TRUE_SRC, MIC_POS)
noise   = np.random.normal(scale=SIGMA_T, size=(N, N))
noise   = np.tril(noise,  k=-1)       # keep lower triangle
noise  -= noise.T                     # reflect to upper & change sign
np.fill_diagonal(noise, 0.0)
D_obs = D_clean + noise

# ---------------------------
# 4.  Bayesian inference
# ---------------------------
# Prior rectangle: 0.001° (~110 m) margin around the array
lat_lo, lat_hi = MIC_POS[:, 0].min() - 0.001, MIC_POS[:, 0].max() + 0.001
lon_lo, lon_hi = MIC_POS[:, 1].min() - 0.001, MIC_POS[:, 1].max() + 0.001

MASK = np.tril(np.ones((N, N), dtype=bool), k=-1)  # lower‑triangular bool mask


def log_likelihood(params):
    """Return log L(D_obs | params) using Frobenius norm on masked residual."""
    lat, lon = params
    pred = predict_delay_matrix((lat, lon), MIC_POS)
    res  = (D_obs - pred)[MASK]
    return -0.5 * np.sum(res ** 2) / SIGMA_T ** 2


def prior_transform(cube):
    """Unit‑cube -> latitude/longitude rectangle."""
    lat = cube[0] * (lat_hi - lat_lo) + lat_lo
    lon = cube[1] * (lon_hi - lon_lo) + lon_lo
    return np.array([lat, lon])

sampler = un.ReactiveNestedSampler(
    ["latitude", "longitude"],
    log_likelihood,
    prior_transform,
    log_dir="run_tdoa_geo_test",
    resume="overwrite",
)

results  = sampler.run()
posterior_samples = results["samples"]  # shape (S, 2)

# Posterior summary
print("True source lat/lon:", TRUE_SRC)
print("Posterior mean lat/lon:", posterior_samples.mean(axis=0))
print("Posterior std‑dev (deg):", posterior_samples.std(axis=0))

# MAP estimate via kernel density
kde     = gaussian_kde(posterior_samples.T)
f_vals  = kde(posterior_samples.T)
map_idx = np.argmax(f_vals)
map_est = posterior_samples[map_idx]
print("Approx. MAP estimate lat/lon:", map_est)

# ---------------------------
# 5.  Folium map with HTTP‑served auto‑open
# ---------------------------
center = posterior_samples.mean(axis=0).tolist()
map_obj = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap",
                     attr="© OpenStreetMap contributors")
HeatMap(posterior_samples, radius=6, blur=10, name="Posterior Heat").add_to(map_obj)
for lat, lon in MIC_POS:
    folium.CircleMarker([lat, lon], radius=3, color="blue", fill=True, name="Mic").add_to(map_obj)
folium.CircleMarker(TRUE_SRC.tolist(), radius=4, color="red", fill=True, popup="True", name="True").add_to(map_obj)
folium.CircleMarker(map_est.tolist(), radius=4, color="purple", fill=True, popup="MAP", name="MAP").add_to(map_obj)
folium.LayerControl(collapsed=False).add_to(map_obj)

HTML_FILE = "tdoa_heatmap.html"
map_obj.save(HTML_FILE)
print(f"Interactive map saved to {HTML_FILE}")

# 6.  Serve the map over HTTP and open in default browser

def serve_map(port=8010):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f"Serving at http://localhost:{port}/{HTML_FILE}")
        httpd.serve_forever()

threading.Thread(target=serve_map, daemon=True).start()
webbrowser.open(f"http://localhost:8010/{HTML_FILE}")
