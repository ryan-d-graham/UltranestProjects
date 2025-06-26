import numpy as np
import ultranest as un
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button, TextBox

# -------------------------------
# Global Model Parameters
# -------------------------------
Q = 1.0          # Total injected mass (kg)
L = 10.0         # Domain length (m)
D = 0.5          # Diffusion coefficient (m^2/s)
v = 0.5          # Advection velocity (m/s)
N_max = 50       # Number of terms in the series expansion
sigma = 0.1      # Standard deviation of measurement noise

# Global variables to store the "true" injection site and saved measurement data.
# saved_data is a list of tuples: ((x, t), u_obs)
true_injection_site = None  
saved_data = []  
posterior_samples = None

# -------------------------------
# Forward Model (Dirichlet BC using Sine Series)
# -------------------------------
def u_model_np(x, t, x0, N_max=N_max):
    """
    Compute the forward model prediction for the advection-diffusion equation 
    with homogeneous Dirichlet boundary conditions using a sine series expansion.

    The solution is given by:
    
      u(x,t;x0) = (2Q/L) * exp((v/(2D))*(x-x0)) *
                  sum_{n=1}^{N_max} [ sin(nπx0/L) sin(nπx/L)
                  exp(-[D*(nπ/L)^2 + v^2/(4D)]*t) ].
    
    Parameters:
      - x: spatial coordinate (m); scalar or 1D array.
      - t: time (s); scalar.
      - x0: injection site (scalar).
      - N_max: number of series terms.
      
    Returns:
      - u: predicted contaminant concentration; same shape as x.
    """
    x = np.atleast_1d(x)
    n = np.arange(1, N_max + 1)  # shape: (N_max,)
    # Compute sine terms for x: shape (len(x), N_max)
    sin_x = np.sin((n * np.pi * x[:, None]) / L)
    # Compute sine terms for injection site x0: shape (N_max,)
    sin_x0 = np.sin((n * np.pi * x0) / L)
    # Compute exponential decay for each mode: shape (N_max,)
    decay = np.exp(- (D * (n * np.pi / L)**2 + v**2 / (4 * D)) * t)
    # Sum over modes (axis=1)
    series = np.sum(sin_x * sin_x0 * decay, axis=1)
    u = (2 * Q / L) * np.exp((v / (2 * D)) * (x.flatten() - x0)) * series
    if u.size == 1:
        return u.item()
    return u

# -------------------------------
# Interactive Figure for Setting Injection Site and Measurements
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, bottom=0.70)

# Create spatial grid for plotting
x_vals = np.linspace(0, L, 400)
initial_disp_t = 1.0      # Initial display time for forward model
initial_inj = 3.0         # Initial injection site value

# Plot the initial forward model curve.
fwd_line, = ax.plot(x_vals, u_model_np(x_vals, initial_disp_t, initial_inj), lw=2)
ax.set_xlim(0, L)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t;x0)')
ax.set_title("Set Injection Site and Measurements")

# Create scatter objects for the measurements (each with a distinct color)
meas1_scatter = ax.scatter([], [], c='red', s=80, zorder=5, label='Measurement 1')
meas2_scatter = ax.scatter([], [], c='green', s=80, zorder=5, label='Measurement 2')
meas3_scatter = ax.scatter([], [], c='blue', s=80, zorder=5, label='Measurement 3')
ax.legend()

# -------------------------------
# Sliders for Injection Site and Display Time
# -------------------------------
ax_inj = plt.axes([0.15, 0.62, 0.7, 0.03])
slider_inj = Slider(ax_inj, "Injection Site x0", 0.0, L, valinit=initial_inj)

ax_disp_t = plt.axes([0.15, 0.57, 0.7, 0.03])
slider_disp_t = Slider(ax_disp_t, "Display Time t", 0.0, 10.0, valinit=initial_disp_t)

def update_forward(val):
    """Update forward model curve when display time or injection site changes."""
    t_disp = slider_disp_t.val
    inj = slider_inj.val
    fwd_line.set_ydata(u_model_np(x_vals, t_disp, inj))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    
slider_disp_t.on_changed(update_forward)
slider_inj.on_changed(update_forward)

# -------------------------------
# Text Boxes for Measurement Inputs (2 rows by 3 columns)
# -------------------------------
# Top row: positions for measurements
ax_meas1_x = plt.axes([0.15, 0.48, 0.2, 0.04])
text_meas1_x = TextBox(ax_meas1_x, "", initial="6.0")
ax_meas1_x.set_title("Meas 1 x", fontsize=10)

ax_meas2_x = plt.axes([0.40, 0.48, 0.2, 0.04])
text_meas2_x = TextBox(ax_meas2_x, "", initial="8.0")
ax_meas2_x.set_title("Meas 2 x", fontsize=10)

ax_meas3_x = plt.axes([0.65, 0.48, 0.2, 0.04])
text_meas3_x = TextBox(ax_meas3_x, "", initial="9.0")
ax_meas3_x.set_title("Meas 3 x", fontsize=10)

# Bottom row: times for measurements
ax_meas1_t = plt.axes([0.15, 0.42, 0.2, 0.04])
text_meas1_t = TextBox(ax_meas1_t, "", initial="1.3")
ax_meas1_t.set_title("Meas 1 t", fontsize=10)

ax_meas2_t = plt.axes([0.40, 0.42, 0.2, 0.04])
text_meas2_t = TextBox(ax_meas2_t, "", initial="1.5")
ax_meas2_t.set_title("Meas 2 t", fontsize=10)

ax_meas3_t = plt.axes([0.65, 0.42, 0.2, 0.04])
text_meas3_t = TextBox(ax_meas3_t, "", initial="1.6")
ax_meas3_t.set_title("Meas 3 t", fontsize=10)

# -------------------------------
# Text Boxes for Additional Forward Model Parameters
# -------------------------------
# Inputs for flow velocity, diffusion and number of terms. Arrange in a single row.
ax_flow = plt.axes([0.15, 0.35, 0.2, 0.04])
text_flow = TextBox(ax_flow, "", initial=str(v))
ax_flow.set_title("Flow velocity v", fontsize=10)

ax_diff = plt.axes([0.40, 0.35, 0.2, 0.04])
text_diff = TextBox(ax_diff, "", initial=str(D))
ax_diff.set_title("Diffusion D", fontsize=10)

ax_terms = plt.axes([0.65, 0.35, 0.2, 0.04])
text_terms = TextBox(ax_terms, "", initial=str(N_max))
ax_terms.set_title("Number of Terms N_max", fontsize=10)

# -------------------------------
# Buttons: Save, Run UltraNest, Show Posterior
# -------------------------------
ax_save = plt.axes([0.10, 0.20, 0.25, 0.05])
button_save = Button(ax_save, "Save", hovercolor='0.975')

ax_run = plt.axes([0.37, 0.20, 0.25, 0.05])
button_run = Button(ax_run, "Run P(x₀ | D)", hovercolor='0.975')

ax_show = plt.axes([0.64, 0.20, 0.25, 0.05])
button_show = Button(ax_show, "Show P(x₀ | D)", hovercolor='0.975')

def save_callback(event):
    global true_injection_site, saved_data, v, D, N_max
    # Get current injection site from the slider.
    inj = slider_inj.val
    true_injection_site = inj
    
    # Parse additional forward model parameters from text boxes.
    try:
        v_new = float(text_flow.text.strip())
        D_new = float(text_diff.text.strip())
        terms_new = int(text_terms.text.strip())
    except Exception as e:
        print("Error parsing forward model parameters:", e)
        return
    # Update global parameters.
    v = v_new
    D = D_new
    N_max = terms_new
    
    # Parse measurement values from text boxes.
    try:
        m1_x = float(text_meas1_x.text.strip())
        m1_t = float(text_meas1_t.text.strip())
        m2_x = float(text_meas2_x.text.strip())
        m2_t = float(text_meas2_t.text.strip())
        m3_x = float(text_meas3_x.text.strip())
        m3_t = float(text_meas3_t.text.strip())
    except Exception as e:
        print("Error parsing measurement inputs:", e)
        return
    
    # Create measurement observations with added noise.
    saved_data = []
    for x_meas, t_meas in ((m1_x, m1_t), (m2_x, m2_t), (m3_x, m3_t)):
        u_obs = u_model_np(x_meas, t_meas, inj) + np.random.normal(0, sigma)
        saved_data.append(((x_meas, t_meas), u_obs))
        print(f"Saved measurement: x = {x_meas:.2f}, t = {t_meas:.2f}, u_obs = {u_obs:.4f}")
    
    # Update the measurement markers using each measurement's own time.
    meas1_scatter.set_offsets(np.c_[[saved_data[0][0][0]], [u_model_np(saved_data[0][0][0], saved_data[0][0][1], inj)]])
    meas2_scatter.set_offsets(np.c_[[saved_data[1][0][0]], [u_model_np(saved_data[1][0][0], saved_data[1][0][1], inj)]])
    meas3_scatter.set_offsets(np.c_[[saved_data[2][0][0]], [u_model_np(saved_data[2][0][0], saved_data[2][0][1], inj)]])
    fig.canvas.draw_idle()

button_save.on_clicked(save_callback)

def run_sampler(event):
    global posterior_samples, saved_data, true_injection_site
    if not saved_data:
        print("No measurements saved. Please save the measurements before running the sampler.")
        return
        
    data_for_un = saved_data.copy()
    
    def local_log_likelihood(params):
        x0 = params[0]
        total_error = 0.0
        for (x, t), u_obs in data_for_un:
            pred = u_model_np(x, t, x0)
            total_error += (pred - u_obs)**2
        return - total_error / (2 * sigma**2)
    
    def local_prior_transform(cube):
        return [cube[0] * L]
    
    print("Running UltraNest sampler...")
    output_folder = "ultranest_results"
    sampler = un.ReactiveNestedSampler(
        ["x0"],
        local_log_likelihood,
        local_prior_transform,
        log_dir=output_folder,
        resume='overwrite'
    )
    sampler.run(min_num_live_points=400)
    posterior_samples = sampler.results["samples"][:, 0]
    print("Sampler complete.")
    
button_run.on_clicked(run_sampler)

def show_posterior(event):
    global posterior_samples, true_injection_site
    if posterior_samples is None:
        print("No posterior available. Run the sampler first.")
        return
    plt.figure(figsize=(8, 4))
    sns.kdeplot(posterior_samples, clip=(0, L), bw_adjust=0.5)
    plt.xlim(0, L)
    plt.xlabel("$x_0$ (m)")
    plt.ylabel("Density")
    plt.title("KDE Estimate of Posterior for Injection Site $x_0$")
    plt.axvline(x=true_injection_site, color='red', linestyle='--', linewidth=2, label='$x_0^{true}$')
    plt.legend()
    plt.show()
    
button_show.on_clicked(show_posterior)

plt.show()
