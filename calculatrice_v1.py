import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

mu0 = 4 * np.pi * 1e-7

def coil_positions_linear(N, d_m):
    idx = np.arange(N, dtype=float)
    return (idx - (N - 1)/2) * d_m

def NI_from_Bcoil(R_m, B_coil):
    return B_coil * 2 * R_m / mu0

def B_loop_on_axis(z, z0, R, NI):
    dz = z - z0
    return mu0 * NI * R**2 / (2 * (R**2 + dz**2)**1.5)

def B_total_on_axis(z, coil_z, R, NI):
    Bz = np.zeros_like(z)
    for z0 in coil_z:
        Bz += B_loop_on_axis(z, z0, R, NI)
    return Bz

def find_z_for_target(z, Bz, target, zmin=0.0, zmax=None):
    if zmax is None:
        zmax = np.max(z)
    mask = (z >= zmin) & (z <= zmax)
    if not np.any(mask):
        return None
    zz = z[mask]
    BB = Bz[mask]
    i = np.argmin(np.abs(BB - target))
    return zz[i], BB[i]

# -------- UI --------
st.set_page_config(layout="wide")
st.title("Calculatrice de champ magnétique – affichage en cm")

col_left, col_right = st.columns([1, 2])

# -------- Colonne gauche : paramètres + positions bobines EN BAS --------
with col_left:
    st.header("Paramètres (cm)")

    R_cm = st.slider("Rayon de bobine R (cm)", 1.0, 30.0, 6.9, 0.1)
    d_cm = st.slider("Distance entre bobines d (cm)", 1.0, 60.0, 21.6, 0.1)
    N = st.slider("Nombre de bobines", 1, 12, 2)

    B_coil = st.slider("Champ au centre d'une bobine B_coil (T)", 0.0, 2.0, 0.27, 0.01)
    target = st.slider("Champ cible B_target (T)", 0.0, 2.0, 0.20, 0.01)
    zspan_cm = st.slider("Fenêtre d'affichage en z (cm)", 5.0, 200.0, 30.0, 1.0)

# -------- Conversions cm -> m --------
R = R_cm / 100
d = d_cm / 100
zspan = zspan_cm / 100

# -------- Calculs --------
coil_z = coil_positions_linear(N, d)
NI = NI_from_Bcoil(R, B_coil)

z = np.linspace(-zspan/2, zspan/2, 3001)
Bz = B_total_on_axis(z, coil_z, R, NI)

zmax_search = np.max(coil_z)
res = find_z_for_target(z, Bz, target, 0.0, zmax_search)

# -------- Colonne droite : GRAPHE EN HAUT + reste en dessous --------
with col_right:
    st.header("Résultats")

    # Graphe en premier (à la place du tableau)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(z * 100, Bz, label="B(z)")
    ax.axhline(target, linestyle="--", label="B_target")
    if res:
        z_star, _ = res
        ax.axvline(z_star * 100, linestyle="--")
    for z0 in coil_z:
        ax.axvline(z0 * 100, color="gray", alpha=0.25)

    ax.set_xlabel("z (cm)")
    ax.set_ylabel("Champ magnétique B (T)")
    ax.set_title("Champ magnétique sur l’axe (affichage en cm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Ensuite NI + message target
    st.write(f"**NI par bobine** : `{NI:.3e} A·tours`")
    if res:
        z_star, B_star = res
        st.success(f"B ≈ {target:.3f} T à z ≈ {z_star*100:.2f} cm")
    else:
        st.warning("Champ cible non trouvé dans la zone centrale")

# -------- Retour colonne gauche : Positions des bobines EN BAS --------
with col_left:
    st.subheader("Positions des bobines (cm)")
    st.write(np.round(coil_z * 100, 2))
