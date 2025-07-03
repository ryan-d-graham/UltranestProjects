# Bayesian Inverse Problems with **UltraNest**  
*Tiny skeleton projects that turn minimal data into **full posterior distributions** — uncertainty and all.*

---

## 1 · A very small primer on Bayesian inversion  

Given data **D** and parameters **θ**, Bayesian inference rewrites the inverse problem  

\[
\text{“find }\theta\text{ given }D\text{”}
\]

as the *posterior* probability density  

\[
p(\theta \mid D)
      \;=\;
      \frac{p(D \mid \theta)\,p(\theta)}
           {\displaystyle
            \int p(D \mid \theta)\,p(\theta)\,d\theta},
      \qquad
      \text{with } 
      p(D \mid \theta) = \mathcal L(\theta)
      \text{ (likelihood), } 
      p(\theta)\text{ (prior).}
\]

Sampling \(p(\theta|D)\) yields an **ensemble of solutions** instead of a single best-fit.  
This repository uses **nested sampling** (via [UltraNest]) to produce those samples and, when helpful, renders them as heat-maps or kernel-density curves.

---

## 2 · Projects in the repo (all “minimal-Viable”)  

| Script | Inverse question | Likelihood \(\mathcal L(\theta)\) | Typical application(s) |
|--------|------------------|------------------------------------|------------------------|
| `bayesian_resection.py` | Where am **I**?  (unknown lat/lon from magnetic bearings) | Wrapped–normal on each bearing:<br/> \(\displaystyle \log\mathcal L = -\tfrac12\sum_i\bigl(\tfrac{\Delta\theta_i}{\sigma}\bigr)^2\) | Search-and-rescue fixes, emergency land nav, quick survey checks |
| `advection_diffusion_inverse.py` | Where along the pipe did the pulse enter?  (1-D source position \(x_0\)) | Gaussian on concentrations:<br/> \(\displaystyle \log\mathcal L = -\tfrac{1}{2\sigma^2}\sum_i (u_\text{obs}-u_\text{pred})^2\) | River-spill forensics, groundwater tracers, industrial leak localisation |
| `tdoa_geolocation.py` | Where is the sound source?  (lat/lon from TDOA matrix at 4 mics) | Frobenius norm on residual delay matrix:<br/> \(\displaystyle \log\mathcal L = -\tfrac{1}{2\sigma_t^{2}}\lVert D_\text{obs}-D_\text{pred}\rVert_F^{2}\) | Urban gunshot detection, wildlife bioacoustics, machinery knock diagnostics |

> **Status:** All three are *skeletons*. They run end-to-end with synthetic data but lack nice I/O, error handling, and in-depth documentation. Pull requests welcome!

---

## 3 · Quick install & run

```bash
git clone https://github.com/your-handle/ultranest-inverse-demos.git
cd ultranest-inverse-demos
python -m venv .venv && source .venv/bin/activate   # optional
pip install numpy scipy ultranest geographiclib folium matplotlib seaborn

# edit beacon coordinates & your compass bearings near the top
python bayesian_resection.py
# → opens  http://localhost:8000/observer_heatmap.html  with posterior heat-map

python advection_diffusion_inverse.py
# → GUI window: move “Injection Site x₀” slider, enter three (x,t) pairs
#   click  Save → Run P(x₀|D) → Show P(x₀|D)

# (optional) replace MIC_POS and D_obs with your own array & delay matrix
python tdoa_geolocation.py
# → opens  http://localhost:8010/tdoa_heatmap.html  with posterior heat-map

# Edit beacon coordinates & compass bearings near the top of the file
python bayesian_resection.py
# → opens  http://localhost:8000/observer_heatmap.html  with a posterior heat-map

python advection_diffusion_inverse.py
# → GUI window:
#   • move “Injection Site x₀” slider
#   • enter three (x,t) pairs
#   • click  Save → Run P(x₀ | D) → Show P(x₀ | D)

# (Optional) replace MIC_POS and D_obs with your own array & delay matrix
python tdoa_geolocation.py
# → opens  http://localhost:8010/tdoa_heatmap.html  with a posterior heat-map